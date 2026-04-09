"""
Airyn inference / evaluation script.

Usage:
  # Interactive generation
  python airyn/generate.py --prompt "The meaning of life is"

  # Longer generation
  python airyn/generate.py --prompt "Once upon a time" --max-tokens 200 --temperature 0.8

  # Compute perplexity on a string
  python airyn/generate.py --perplexity "The quick brown fox jumps over the lazy dog"

  # Use a specific checkpoint
  python airyn/generate.py --checkpoint checkpoints/my_run.pt --prompt "Hello"
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

# Import model classes from train.py
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
def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Autoregressive generation with top-k/top-p sampling."""
    seq = input_ids.clone()
    for _ in range(max_new_tokens):
        # Crop to max seq_len if needed
        x = seq if seq.size(1) <= 1024 else seq[:, -1024:]
        # Forward pass — model expects (input_ids, target_ids), we use a dummy target
        # and extract logits manually
        tok_emb = model.tok_emb(x)
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
        # Get logits for last position
        last_h = h[:, -1, :]
        if model.tie_embeddings:
            logits = F.linear(last_h, model.tok_emb.weight)
        else:
            logits = model.lm_head(last_h)
        logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
        logits = logits.float()

        # Temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
            continue

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    return seq


@torch.no_grad()
def compute_perplexity(model: GPT, input_ids: torch.Tensor) -> float:
    """Compute perplexity on a token sequence."""
    if input_ids.size(1) < 2:
        return float("inf")
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss = model(x, y)
    return math.exp(loss.item())


def find_latest_checkpoint() -> str | None:
    ckpts = sorted(glob.glob("checkpoints/*.pt"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser(description="Airyn inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--perplexity", type=str, default=None, help="Text to compute perplexity on")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt loop")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        print("No checkpoint found. Run training first or specify --checkpoint.")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading {ckpt}...")
    model = load_model(ckpt, device)
    enc = tiktoken.get_encoding("gpt2")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.1f}M params on {device}")

    if args.perplexity:
        tokens = enc.encode(args.perplexity)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        ppl = compute_perplexity(model, input_ids)
        print(f"Perplexity: {ppl:.2f} ({len(tokens)} tokens)")
        return

    def run_prompt(prompt_text: str):
        tokens = enc.encode(prompt_text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        output_ids = generate(
            model, input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        generated = enc.decode(output_ids[0].tolist())
        print(f"\n{generated}\n")

    if args.prompt:
        run_prompt(args.prompt)

    if args.interactive or (args.prompt is None and args.perplexity is None):
        print("Interactive mode (ctrl+c to exit)")
        while True:
            try:
                prompt = input(">>> ")
                if prompt.strip():
                    run_prompt(prompt.strip())
            except (KeyboardInterrupt, EOFError):
                print()
                break


if __name__ == "__main__":
    main()
