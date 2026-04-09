"""
Airyn evaluation script — HellaSwag and LAMBADA benchmarks.

Usage:
  python airyn/eval.py                                          # both benchmarks, latest checkpoint
  python airyn/eval.py --checkpoint checkpoints/airyn_sft_final.pt  # specific checkpoint
  python airyn/eval.py --benchmarks hellaswag                   # just one
  python airyn/eval.py --benchmarks lambada                     # just one
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from airyn.train import (
    GPT,
    CastedLinear,
    Hyperparameters,
    restore_low_dim_params_to_fp32,
)


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.eval()
    return model


@torch.no_grad()
def get_logits(model: GPT, input_ids: Tensor) -> Tensor:
    """Forward pass returning full logits tensor (B, T, V)."""
    tok_emb = model.tok_emb(input_ids)
    h = F.rms_norm(tok_emb, (tok_emb.size(-1),))
    x0 = h
    skips = []
    for i in range(model.num_encoder_layers):
        h = model.blocks[i](h, x0)
        skips.append(h)
    for i in range(model.num_decoder_layers):
        if skips:
            h = h + model.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
        h = model.blocks[model.num_encoder_layers + i](h, x0)
    h = model.final_norm(h)
    if model.tie_embeddings:
        logits = F.linear(h, model.tok_emb.weight)
    else:
        logits = model.lm_head(h)
    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
    return logits.float()


# -----------------------------
# HELLASWAG
# -----------------------------

def eval_hellaswag(model: GPT, device: torch.device, max_seq_len: int = 1024) -> dict:
    """Evaluate on HellaSwag validation set (10,042 examples)."""
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")

    print("Loading HellaSwag validation set...")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    correct_norm = 0
    correct_sum = 0
    total = 0
    t0 = time.time()

    for idx, example in enumerate(ds):
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        ctx_tokens = enc.encode(ctx)

        # Tokenize each ending (space-prepended for proper BPE)
        ending_tokens_list = [enc.encode(" " + e) for e in endings]

        # For each candidate: context + ending
        candidate_losses_norm = []
        candidate_losses_sum = []

        for ending_tokens in ending_tokens_list:
            full_tokens = ctx_tokens + ending_tokens
            # Truncate from left if too long (keep ending visible)
            if len(full_tokens) > max_seq_len:
                excess = len(full_tokens) - max_seq_len
                full_tokens = full_tokens[excess:]
                # Recalculate where the ending starts
                completion_start = len(full_tokens) - len(ending_tokens)
            else:
                completion_start = len(ctx_tokens)

            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = get_logits(model, input_ids)

            # Shift: logits[t] predicts token[t+1]
            shift_logits = logits[0, :-1, :]  # (T-1, V)
            shift_targets = input_ids[0, 1:]   # (T-1,)

            per_token_loss = F.cross_entropy(shift_logits, shift_targets, reduction="none")

            # Only count loss on completion tokens
            completion_losses = per_token_loss[completion_start - 1:]  # -1 because of shift
            avg_loss = completion_losses.mean().item()
            sum_loss = completion_losses.sum().item()
            candidate_losses_norm.append(avg_loss)
            candidate_losses_sum.append(sum_loss)

        pred_norm = min(range(4), key=lambda i: candidate_losses_norm[i])
        pred_sum = min(range(4), key=lambda i: candidate_losses_sum[i])

        if pred_norm == label:
            correct_norm += 1
        if pred_sum == label:
            correct_sum += 1
        total += 1

        if (idx + 1) % 500 == 0 or idx + 1 == len(ds):
            elapsed = time.time() - t0
            print(f"  HellaSwag [{idx+1}/{len(ds)}] "
                  f"acc_norm={correct_norm/total:.4f} acc={correct_sum/total:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

    return {
        "hellaswag_acc_norm": correct_norm / total,
        "hellaswag_acc": correct_sum / total,
        "hellaswag_total": total,
    }


# -----------------------------
# LAMBADA
# -----------------------------

def eval_lambada(model: GPT, device: torch.device, max_seq_len: int = 1024) -> dict:
    """Evaluate on LAMBADA OpenAI test set (5,153 examples)."""
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")

    print("Loading LAMBADA (OpenAI) test set...")
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")

    correct = 0
    total = 0
    total_loss = 0.0
    t0 = time.time()

    for idx, example in enumerate(ds):
        text = example["text"]
        tokens = enc.encode(text)

        if len(tokens) < 2:
            continue
        if len(tokens) > max_seq_len:
            tokens = tokens[-max_seq_len:]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = get_logits(model, input_ids)

        # Last token prediction: logits at position -2 predicts token at -1
        last_logits = logits[0, -2, :]
        predicted = last_logits.argmax().item()
        target = tokens[-1]

        loss = F.cross_entropy(last_logits.unsqueeze(0), torch.tensor([target], device=device)).item()
        total_loss += loss

        if predicted == target:
            correct += 1
        total += 1

        if (idx + 1) % 500 == 0 or idx + 1 == len(ds):
            elapsed = time.time() - t0
            ppl = math.exp(total_loss / total) if total > 0 else float("inf")
            print(f"  LAMBADA [{idx+1}/{len(ds)}] "
                  f"acc={correct/total:.4f} ppl={ppl:.1f} "
                  f"({elapsed:.0f}s)", flush=True)

    avg_loss = total_loss / total if total > 0 else float("inf")
    return {
        "lambada_acc": correct / total,
        "lambada_ppl": math.exp(avg_loss),
        "lambada_total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Airyn Benchmarks")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default="hellaswag,lambada",
                        help="Comma-separated: hellaswag,lambada")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpts = sorted(glob.glob("checkpoints/*.pt"), key=os.path.getmtime)
        if not ckpts:
            print("No checkpoint found.")
            sys.exit(1)
        ckpt = ckpts[-1]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading {ckpt}...")
    model = load_model(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params on {device}")

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    results = {}

    for bench in benchmarks:
        print(f"\n{'='*60}")
        print(f"  Running: {bench.upper()}")
        print(f"{'='*60}")
        if bench == "hellaswag":
            results.update(eval_hellaswag(model, device))
        elif bench == "lambada":
            results.update(eval_lambada(model, device))
        else:
            print(f"Unknown benchmark: {bench}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {Path(ckpt).name}")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # GPT-2 124M reference
    print(f"\n  Reference (GPT-2 124M, 40B tokens):")
    print(f"    hellaswag_acc_norm: ~0.294")
    print(f"    lambada_acc:        ~0.326")
    print(f"    lambada_ppl:        ~35-40")


if __name__ == "__main__":
    main()
