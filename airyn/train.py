"""
Airyn Training Harness -- Step 1 Baseline
Forked from OpenAI parameter-golf/train_gpt.py.
Trains a ~124M parameter GPT-2 class transformer on fineweb10B with Muon optimizer.

Usage:
  torchrun --nproc_per_node=1 airyn/train.py          # single GPU test
  torchrun --nproc_per_node=2 airyn/train.py          # 2x RTX 6000
  torchrun --nproc_per_node=8 airyn/train.py          # 8x H100
"""

from __future__ import annotations

import os
os.environ.setdefault("USE_LIBUV", "0")  # Windows PyTorch nightly lacks libuv

import copy
import glob
import math
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# ~124M parameter baseline: 12 layers, 768 dim, 12 heads, 4x MLP
# GPT-2 tokenizer (vocab 50304 = 50257 padded to multiple of 128)
# 524k tokens/step for 5000 iterations (~2.6B tokens)

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "data/fineweb10B")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 250))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    iterations = int(os.environ.get("ITERATIONS", 5000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 50304))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 12))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Logging / eval.
    wandb_project = os.environ.get("WANDB_PROJECT", "airyn")
    wandb_enabled = bool(int(os.environ.get("WANDB_ENABLED", "1")))
    eval_hellaswag = bool(int(os.environ.get("EVAL_HELLASWAG", "0")))
    ffn_type = os.environ.get("FFN_TYPE", "swiglu")  # "relu_sq" | "swiglu" | "moe"
    n_experts = int(os.environ.get("N_EXPERTS", 8))
    n_active_experts = int(os.environ.get("N_ACTIVE_EXPERTS", 2))
    n_shared_experts = int(os.environ.get("N_SHARED_EXPERTS", 1))
    torch_compile = bool(int(os.environ.get("TORCH_COMPILE", "1")))  # 0 to disable
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 0))  # 0 = auto-compute to keep ~524k tokens per step
    ckpt_every = int(os.environ.get("CKPT_EVERY", 500))  # save checkpoint every N steps (0 = only at end)
    resume_from = os.environ.get("RESUME_FROM", "")  # path to checkpoint to resume from

# Parameter name patterns identifying control/scalar tensors (optimizer split).
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
)

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# VALIDATION
# -----------------------------

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
) -> float:
    """Compute cross-entropy validation loss over the full validation set."""
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    model.train()
    return float(val_loss.item())


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        sdpa_kwargs = dict(attn_mask=None, is_causal=True)
        if self.num_kv_heads != self.num_heads:
            sdpa_kwargs["enable_gqa"] = True
        y = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_mult: int | None = None, *, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            if mlp_mult is None:
                raise ValueError("SwiGLU requires mlp_mult or hidden_dim")
            # 2/3 factor compensates for the extra gate projection to keep ~same param count as MLP
            hidden = int(2 / 3 * mlp_mult * dim)
            hidden = ((hidden + 127) // 128) * 128  # round up to nearest multiple of 128
        else:
            hidden = hidden_dim
        if hidden <= 0:
            raise ValueError(f"SwiGLU hidden dim must be positive, got {hidden}")
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.down = CastedLinear(hidden, dim, bias=False)
        self.down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoELayer(nn.Module):
    """Mixture of Experts with sigmoid routing and auxfree load balancing."""

    def __init__(self, dim: int, expert_hidden_dim: int, n_experts: int = 8, n_active: int = 2, n_shared: int = 1):
        super().__init__()
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if n_active <= 0 or n_active > n_experts:
            raise ValueError(f"n_active must be in [1, n_experts], got n_active={n_active}, n_experts={n_experts}")
        if n_shared < 0:
            raise ValueError(f"n_shared must be non-negative, got {n_shared}")
        self.n_experts = n_experts
        self.n_active = n_active

        # Router uses sigmoid gating rather than softmax so tokens can independently score experts.
        self.router = CastedLinear(dim, n_experts, bias=False)

        # Routed experts are SwiGLU FFNs with an explicit hidden dimension.
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim=expert_hidden_dim) for _ in range(n_experts)])

        # Shared expert is always added on top of the routed expert mixture.
        self.shared_expert = SwiGLU(dim, hidden_dim=expert_hidden_dim) if n_shared > 0 else None

        # Auxfree bias is updated manually from observed routing load and should stay in fp32.
        self.register_buffer("expert_bias", torch.zeros(n_experts, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.reshape(-1, dim)

        router_logits = self.router(x_flat).float() + self.expert_bias
        scores = torch.sigmoid(router_logits)
        topk_scores, topk_indices = torch.topk(scores, k=self.n_active, dim=-1)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)

        output = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_output = self.experts[i](expert_input)
            weight_idx = (topk_indices[mask] == i).float()
            weight = (topk_weights[mask] * weight_idx).sum(dim=-1, keepdim=True).to(dtype=expert_output.dtype)
            output[mask] += weight * expert_output

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat).to(dtype=output.dtype)

        if self.training:
            with torch.no_grad():
                load = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
                for i in range(self.n_experts):
                    load[i] = (topk_indices == i).float().sum()
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    if load.device.type == "cuda" and dist.get_backend() == "gloo":
                        load_cpu = load.cpu()
                        dist.all_reduce(load_cpu, op=dist.ReduceOp.SUM)
                        load.copy_(load_cpu.to(load.device))
                    else:
                        dist.all_reduce(load, op=dist.ReduceOp.SUM)
                self.expert_bias.add_(0.001 * (load.mean() - load))

        return output.reshape(bsz, seqlen, dim)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        attn_factory: Callable[[int, int, int, float, float], nn.Module] | None = None,
        ffn_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        if attn_factory is not None:
            self.attn = attn_factory(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        if ffn_factory is not None:
            self.mlp = ffn_factory(dim, mlp_mult)
        else:
            self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        attn_factory: Callable[[int, int, int, float, float], nn.Module] | None = None,
        ffn_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    attn_factory=attn_factory,
                    ffn_factory=ffn_factory,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        # Chunked cross-entropy avoids materializing full (B*T, V) logits in fp32
        chunk_size = 4096
        total_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        weight = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        bias = None if self.tie_embeddings else (self.lm_head.bias if self.lm_head.bias is not None else None)
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i : i + chunk_size]
            t_chunk = targets[i : i + chunk_size]
            logits_proj = F.linear(x_chunk, weight.to(x_chunk.dtype), bias.to(x_chunk.dtype) if bias is not None else None)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            total_loss += F.cross_entropy(logits.float(), t_chunk, reduction="sum")
        return total_loss / targets.numel()


# -----------------------------
# TRAINING
# -----------------------------



def save_checkpoint(
    path: str,
    base_model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    step: int,
    args: Hyperparameters,
) -> None:
    """Save model, optimizer states, and training progress."""
    ckpt = {
        "model_state_dict": base_model.state_dict(),
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "step": step,
        "run_id": args.run_id,
        "train_batch_tokens": args.train_batch_tokens,
        "train_seq_len": args.train_seq_len,
        "iterations": args.iterations,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


def main() -> None:
    global zeropower_via_newtonschulz5

    args = Hyperparameters()
    model_torch_compile = args.torch_compile and args.ffn_type != "moe"
    if args.torch_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = "RANK" in os.environ and requested_world_size > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    world_size = requested_world_size if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {requested_world_size}")
    if args.grad_accum_steps > 0:
        grad_accum_steps = args.grad_accum_steps
    else:
        # Target ~524k tokens per step regardless of GPU count
        local_batch_tokens = args.train_batch_tokens // world_size
        local_batch_seqs = local_batch_tokens // args.train_seq_len
        grad_accum_steps = max(1, local_batch_seqs)
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        if sys.platform == "win32":
            dist.init_process_group(backend="gloo")
        else:
            dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    # Let PyTorch pick the best SDP backend for the GPU arch.
    # Blackwell (sm_120) may need CuDNN or math fallback.
    enable_cudnn_sdp(True)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(True)
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # SEEDS + VALIDATION DATA
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    if args.ffn_type == "moe" and args.torch_compile:
        log0("ffn_type:moe disables model torch.compile because expert routing uses dynamic control flow")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    # FFN factory based on ffn_type
    ffn_factory = None
    expert_hidden = None
    if args.ffn_type == "moe":
        swiglu_hidden = args.mlp_mult * args.model_dim
        swiglu_hidden = int(swiglu_hidden * 2 / 3)
        swiglu_hidden = swiglu_hidden - (swiglu_hidden % 128) or 128
        expert_hidden = swiglu_hidden // args.n_active_experts
        expert_hidden = expert_hidden - (expert_hidden % 64) or 64

        def ffn_factory(dim: int, _hidden_dim: int) -> nn.Module:
            return MoELayer(dim, expert_hidden, args.n_experts, args.n_active_experts, args.n_shared_experts)
    elif args.ffn_type == "swiglu":
        ffn_factory = lambda dim, mlp_mult: SwiGLU(dim, mlp_mult)
    elif args.ffn_type != "relu_sq":
        raise ValueError(f"Unknown FFN_TYPE={args.ffn_type!r}, expected 'relu_sq', 'swiglu', or 'moe'")

    base_model = GPT(
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
        ffn_factory=ffn_factory,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    for module in base_model.modules():
        if isinstance(module, MoELayer):
            module.expert_bias = module.expert_bias.float()
    if model_torch_compile:
        compiled_model = torch.compile(base_model, dynamic=False)
    else:
        compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR / TIED_EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    # --- Resume from checkpoint ---
    start_step = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        for opt, opt_state in zip(optimizers, ckpt["optimizer_states"], strict=True):
            opt.load_state_dict(opt_state)
        start_step = ckpt.get("step", 0) + 1
        if rank == 0:
            print(f"Resumed from {args.resume_from} at step {start_step}")

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    if args.ffn_type == "moe":
        expert_params_per_layer = sum(p.numel() for p in base_model.blocks[0].mlp.experts[0].parameters())
        active_expert_params = expert_params_per_layer * args.n_active_experts * len(base_model.blocks)
        shared_params = sum(p.numel() for p in base_model.blocks[0].mlp.shared_expert.parameters()) * len(base_model.blocks) if args.n_shared_experts > 0 else 0
        non_expert_params = n_params - sum(p.numel() for p in base_model.blocks.parameters()) + sum(p.numel() for p in base_model.blocks[0].attn.parameters()) * len(base_model.blocks) + sum(p.numel() for p in base_model.blocks[0].attn_norm.parameters()) * len(base_model.blocks) + sum(p.numel() for p in base_model.blocks[0].mlp_norm.parameters()) * len(base_model.blocks)
        log0(f"moe_config: {args.n_experts} experts, top-{args.n_active_experts}, {args.n_shared_experts} shared")
        log0(f"expert_hidden_dim: {expert_hidden}")
        log0(f"total_params:{n_params:,} active_approx:{n_params - (args.n_experts - args.n_active_experts) * expert_params_per_layer * len(base_model.blocks):,}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"ffn_type:{args.ffn_type} torch_compile:{model_torch_compile}")
    log0(f"attention_mode:{'gqa' if args.num_kv_heads != args.num_heads else 'mha'} num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # WANDB INIT
    # -----------------------------

    _wandb_active = False
    if args.wandb_enabled and wandb is not None and master_process:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.run_id,
                config={
                    "vocab_size": args.vocab_size,
                    "num_layers": args.num_layers,
                    "model_dim": args.model_dim,
                    "num_heads": args.num_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "mlp_mult": args.mlp_mult,
                    "train_seq_len": args.train_seq_len,
                    "train_batch_tokens": args.train_batch_tokens,
                    "iterations": args.iterations,
                    "tie_embeddings": args.tie_embeddings,
                    "matrix_lr": args.matrix_lr,
                    "scalar_lr": args.scalar_lr,
                    "ffn_type": args.ffn_type,
                    "n_params": n_params,
                    "world_size": world_size,
                },
            )
            _wandb_active = True
            log0(f"wandb: logging to project '{args.wandb_project}' run '{args.run_id}'")
        except Exception as e:
            log0(f"WARNING: wandb.init() failed: {e}. Training will continue without wandb.")
    elif args.wandb_enabled and wandb is None and master_process:
        log0("WARNING: wandb_enabled=True but wandb not installed. Skipping wandb logging.")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step: int) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if step >= warmdown_start:
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
        return 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0 and start_step == 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(start_step, args.iterations + 1):
        last_step = step == args.iterations

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens)
            tokens_seen = step * args.train_batch_tokens
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if _wandb_active:
                wandb.log({"val/loss": val_loss, "tokens_seen": tokens_seen}, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # --- Training step ---
        scale = lr_mul(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # Periodic checkpoint save
        if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0 and master_process:
            ckpt_path = f"checkpoints/{args.run_id}_step{step + 1}.pt"
            save_checkpoint(ckpt_path, base_model, optimizers, step + 1, args)
            log0(f"Checkpoint saved: {ckpt_path}")
        if distributed and args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            dist.barrier()

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step + 1 <= 10 or (step + 1) % args.train_log_every == 0)
        if should_log_train:
            tokens_seen = (step + 1) * args.train_batch_tokens
            throughput = tokens_seen / (approx_training_time_ms / 1000.0) if approx_training_time_ms > 0 else 0
            log0(
                f"step:{step + 1}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms "
                f"tok/s:{throughput:.0f}"
            )
            if args.ffn_type == "moe" and (step + 1) % 50 == 0:
                biases = base_model.blocks[0].mlp.expert_bias
                log0(f"expert_load_bias: {biases.tolist()}")
            if _wandb_active:
                current_lr = scale * args.matrix_lr
                wandb.log({
                    "train/loss": train_loss.item(),
                    "train/lr": current_lr,
                    "train/tokens_seen": tokens_seen,
                    "train/throughput_tok_per_sec": throughput,
                }, step=step + 1)

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    if master_process:
        ckpt_path = f"checkpoints/{args.run_id}.pt"
        save_checkpoint(ckpt_path, base_model, optimizers, args.iterations, args)
        log0(f"Saved final checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path)} bytes)")
    if distributed:
        dist.barrier()

    if _wandb_active:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
