# Project Airyn

Modular LLM training harness forked from [OpenAI parameter-golf](https://github.com/openai/parameter-golf). Goal: a clean, extensible baseline for architecture experiments and scaling.

## Project structure

```
airyn/
  train.py          -- Main training script (~124M GPT-2 baseline, Muon optimizer)
  prepare_data.py   -- Downloads FineWeb-Edu and tokenizes to GPT-2 .bin shards
parameter-golf/     -- Upstream reference (read-only, do not modify)
data/fineweb10B/    -- GPT-2 tokenized shards (fineweb_train_*.bin, fineweb_val_*.bin)
venv/               -- Python 3.12 virtualenv (PyTorch nightly + CUDA 12.8)
```

## Environment

- **Always use the venv**: `venv/Scripts/python`, `venv/Scripts/pip`
- PyTorch nightly 2.12+ with CUDA 12.8 required (Blackwell sm_120 GPUs)
- Local hardware: 2x NVIDIA RTX PRO 6000 Blackwell (102.6 GB each)

## Commands

```bash
# Data prep (streams from HuggingFace, ~30min for 10B tokens)
venv/Scripts/python airyn/prepare_data.py

# Train single GPU
venv/Scripts/torchrun --nproc_per_node=1 airyn/train.py

# Train 2x GPU (local)
venv/Scripts/torchrun --nproc_per_node=2 airyn/train.py

# Override hyperparams via env vars
ITERATIONS=1000 VAL_LOSS_EVERY=100 venv/Scripts/torchrun --nproc_per_node=2 airyn/train.py

# Disable wandb for quick tests
WANDB_ENABLED=0 venv/Scripts/torchrun --nproc_per_node=1 airyn/train.py
```

## Architecture (train.py)

- **Hyperparameters**: class with env-var overrides for every setting
- **Muon optimizer**: Newton-Schulz orthogonalized updates for matrix params, Adam for scalars/embeddings
- **Block**: accepts `attn_factory` and `ffn_factory` callables to swap attention/FFN per-layer
- **GPT**: U-Net style skip connections (encoder half stores, decoder half consumes)
- **Data**: modded-nanogpt shard format (magic 20240520, uint16 tokens, 256-int header)
- **Distributed**: DDP with gradient accumulation (8 // world_size micro-steps)

## Key design decisions

- GPT-2 tokenizer (vocab 50304 = 50257 padded to 128) instead of SentencePiece
- Full MHA (num_kv_heads = num_heads = 12) for baseline, not GQA
- relu^2 MLP activation (from modded-nanogpt), will swap to SwiGLU later
- Logit softcap at 30.0 (tanh-based)
- CastedLinear keeps weights fp32, casts to bf16 at matmul time
- torch.compile with fullgraph=True on model, and on Newton-Schulz

## Conventions

- All code changes go in `airyn/`, never modify `parameter-golf/`
- Hyperparameters are always overridable via environment variables
- Validation is cross-entropy loss on fineweb val split (no BPB)
- wandb logging is on by default (project "airyn"), disable with WANDB_ENABLED=0
- Checkpoints save to `checkpoints/{run_id}.pt`
- Logs save to `logs/{run_id}.txt`

## Roadmap

1. **Baseline** (current): ~124M param GPT-2 class model, Muon, fineweb10B
2. **SwiGLU**: swap MLP via ffn_factory
3. **HellaSwag eval**: eval_hellaswag flag
4. **Scaling experiments**: larger models, longer training, multi-node
