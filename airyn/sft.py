"""
Airyn SFT (Supervised Fine-Tuning) script.
Fine-tunes a pretrained Airyn checkpoint on instruction-following data.

Usage:
  python airyn/sft.py                                    # defaults
  python airyn/sft.py --epochs 3 --lr 1e-4              # tune hyperparams
  python airyn/sft.py --checkpoint checkpoints/my.pt     # specific checkpoint
"""

from __future__ import annotations

import os
os.environ.setdefault("USE_LIBUV", "0")

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from airyn.train import (
    GPT,
    CastedLinear,
    Hyperparameters,
    SwiGLU,
    restore_low_dim_params_to_fp32,
)

# Chat template — simple text markers the model learns to follow.
SYSTEM_PROMPT = "You are Airyn, a helpful, intelligent, and concise AI assistant. You answer questions clearly and accurately."

def format_example(instruction: str, input_text: str, output: str) -> str:
    """Format a single training example with the Airyn chat template."""
    parts = [f"### System:\n{SYSTEM_PROMPT}\n"]
    if input_text.strip():
        parts.append(f"### User:\n{instruction}\n{input_text}\n")
    else:
        parts.append(f"### User:\n{instruction}\n")
    parts.append(f"### Assistant:\n{output}")
    return "\n".join(parts)


def load_and_tokenize_dataset(enc: tiktoken.Encoding, max_seq_len: int, max_examples: int = 0):
    """Load alpaca-cleaned from HuggingFace and tokenize."""
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    print("Loading dataset yahma/alpaca-cleaned...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")

    eot = enc._special_tokens["<|endoftext|>"]
    all_sequences = []
    skipped = 0

    examples = list(ds)
    if max_examples > 0:
        examples = examples[:max_examples]

    for ex in examples:
        text = format_example(ex["instruction"], ex["input"], ex["output"])
        tokens = enc.encode(text) + [eot]
        if len(tokens) <= max_seq_len:
            # Pad to max_seq_len for simple batching
            padded = tokens + [eot] * (max_seq_len - len(tokens))
            all_sequences.append((padded, len(tokens)))
        else:
            skipped += 1

    print(f"Tokenized {len(all_sequences)} examples ({skipped} skipped as too long)")
    return all_sequences


def main():
    parser = argparse.ArgumentParser(description="Airyn SFT")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-examples", type=int, default=0, help="Max examples (0=all)")
    parser.add_argument("--val-split", type=float, default=0.02, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = Hyperparameters()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        import glob
        ckpts = sorted(glob.glob("checkpoints/*.pt"), key=os.path.getmtime)
        if not ckpts:
            print("No checkpoint found. Run pretraining first.")
            sys.exit(1)
        ckpt_path = ckpts[-1]

    # Load model
    print(f"Loading model from {ckpt_path}...")
    ffn_factory = None
    if hparams.ffn_type == "swiglu":
        ffn_factory = lambda dim, mlp_mult: SwiGLU(dim, mlp_mult)
    model = GPT(
        vocab_size=hparams.vocab_size,
        num_layers=hparams.num_layers,
        model_dim=hparams.model_dim,
        num_heads=hparams.num_heads,
        num_kv_heads=hparams.num_kv_heads,
        mlp_mult=hparams.mlp_mult,
        tie_embeddings=hparams.tie_embeddings,
        tied_embed_init_std=hparams.tied_embed_init_std,
        logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base,
        qk_gain_init=hparams.qk_gain_init,
        ffn_factory=ffn_factory,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")

    # Tokenize dataset
    enc = tiktoken.get_encoding("gpt2")
    seq_len = hparams.train_seq_len  # 1024
    data = load_and_tokenize_dataset(enc, seq_len, args.max_examples)

    # Train/val split
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_split))
    val_data = data[:n_val]
    train_data = data[n_val:]
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")

    # Optimizer — simple AdamW, lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Training loop
    total_steps = args.epochs * (len(train_data) // args.batch_size)
    print(f"Training for {args.epochs} epochs, {total_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 80)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        epoch_tokens = 0

        for i in range(0, len(train_data) - args.batch_size + 1, args.batch_size):
            batch = train_data[i : i + args.batch_size]
            # Build input/target tensors with loss masking
            input_ids = torch.zeros(args.batch_size, seq_len, dtype=torch.long, device=device)
            target_ids = torch.full((args.batch_size, seq_len), -100, dtype=torch.long, device=device)

            for j, (tokens, real_len) in enumerate(batch):
                t = torch.tensor(tokens, dtype=torch.long)
                input_ids[j] = t[:seq_len]
                # Only compute loss on assistant response tokens
                text = enc.decode(tokens[:real_len])
                assistant_marker = "### Assistant:\n"
                marker_pos = text.find(assistant_marker)
                if marker_pos >= 0:
                    prefix_tokens = len(enc.encode(text[:marker_pos + len(assistant_marker)]))
                    # Target: predict tokens after the assistant marker
                    target_ids[j, prefix_tokens - 1 : real_len - 1] = t[prefix_tokens : real_len]
                else:
                    # Fallback: loss on all tokens
                    target_ids[j, :real_len - 1] = t[1:real_len]

            # Forward — we can't use model(x, y) directly since it uses reduction="mean"
            # and doesn't support -100 masking. So we compute loss manually.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tok_emb = model.tok_emb(input_ids)
                h = F.rms_norm(tok_emb, (tok_emb.size(-1),))
                x0 = h
                skips = []
                for bi in range(model.num_encoder_layers):
                    h = model.blocks[bi](h, x0)
                    skips.append(h)
                for bi in range(model.num_decoder_layers):
                    if skips:
                        h = h + model.skip_weights[bi].to(dtype=h.dtype)[None, None, :] * skips.pop()
                    h = model.blocks[model.num_encoder_layers + bi](h, x0)
                h = model.final_norm(h)
                if model.tie_embeddings:
                    logits = F.linear(h, model.tok_emb.weight)
                else:
                    logits = model.lm_head(h)
                logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tokens = (target_ids != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            step += 1

            if step % args.log_every == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                elapsed = time.time() - t0
                print(f"epoch:{epoch+1}/{args.epochs} step:{step}/{total_steps} "
                      f"loss:{loss.item():.4f} avg_loss:{avg_loss:.4f} "
                      f"time:{elapsed:.0f}s", flush=True)

            if args.save_every > 0 and step % args.save_every == 0:
                save_path = f"checkpoints/airyn_sft_step{step}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  Saved {save_path}", flush=True)

        # End of epoch — validation
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.no_grad():
            for i in range(0, len(val_data) - args.batch_size + 1, args.batch_size):
                batch = val_data[i : i + args.batch_size]
                input_ids = torch.zeros(args.batch_size, seq_len, dtype=torch.long, device=device)
                target_ids = torch.full((args.batch_size, seq_len), -100, dtype=torch.long, device=device)
                for j, (tokens, real_len) in enumerate(batch):
                    t = torch.tensor(tokens, dtype=torch.long)
                    input_ids[j] = t[:seq_len]
                    text = enc.decode(tokens[:real_len])
                    assistant_marker = "### Assistant:\n"
                    marker_pos = text.find(assistant_marker)
                    if marker_pos >= 0:
                        prefix_tokens = len(enc.encode(text[:marker_pos + len(assistant_marker)]))
                        target_ids[j, prefix_tokens - 1 : real_len - 1] = t[prefix_tokens : real_len]
                    else:
                        target_ids[j, :real_len - 1] = t[1:real_len]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tok_emb = model.tok_emb(input_ids)
                    h = F.rms_norm(tok_emb, (tok_emb.size(-1),))
                    x0 = h
                    skips = []
                    for bi in range(model.num_encoder_layers):
                        h = model.blocks[bi](h, x0)
                        skips.append(h)
                    for bi in range(model.num_decoder_layers):
                        if skips:
                            h = h + model.skip_weights[bi].to(dtype=h.dtype)[None, None, :] * skips.pop()
                        h = model.blocks[model.num_encoder_layers + bi](h, x0)
                    h = model.final_norm(h)
                    if model.tie_embeddings:
                        logits = F.linear(h, model.tok_emb.weight)
                    else:
                        logits = model.lm_head(h)
                    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
                vl = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=-100,
                )
                n = (target_ids != -100).sum().item()
                val_loss_sum += vl.item() * n
                val_tokens += n

        val_loss = val_loss_sum / max(val_tokens, 1)
        print(f"=== Epoch {epoch+1} done. Val loss: {val_loss:.4f} ===", flush=True)
        model.train()

    # Save final
    os.makedirs("checkpoints", exist_ok=True)
    final_path = "checkpoints/airyn_sft_final.pt"
    torch.save(model.state_dict(), final_path)
    elapsed = time.time() - t0
    print(f"\nSFT complete. Final checkpoint: {final_path} ({elapsed:.0f}s total)")
    print(f"Test with: python airyn/generate.py --checkpoint {final_path} --prompt \"### System:\\n{SYSTEM_PROMPT}\\n\\n### User:\\nWhat is gravity?\\n\\n### Assistant:\"")


if __name__ == "__main__":
    main()
