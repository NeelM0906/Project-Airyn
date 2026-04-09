"""
Download and tokenize multiple pretraining & SFT datasets for Airyn.
Reuses the shard format from prepare_data.py (magic=20240520, uint16, 256-int header, ~100M tok/shard).
Uses GPT-2 tiktoken encoding.

Pretraining data -> .bin shards (parallel threads, one per dataset)
SFT data -> .jsonl files (sequential, small)

Usage:
  python airyn/prepare_data_mix.py
  python airyn/prepare_data_mix.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from pathlib import Path
from threading import Lock

import numpy as np

# ---------------------------------------------------------------------------
# Shard format (must match prepare_data.py / train.py)
# ---------------------------------------------------------------------------
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
DEFAULT_SHARD_SIZE = 100_000_000  # ~100M tokens per shard

_print_lock = Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def write_shard(path: Path, tokens: np.ndarray) -> None:
    assert tokens.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def tokenize_batch(texts: list[str]) -> np.ndarray:
    """Tokenize a batch of documents with GPT-2, prepending EOT per doc."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    all_tokens: list[int] = []
    for text in texts:
        all_tokens.append(eot)
        all_tokens.extend(enc.encode_ordinary(text))
    return np.array(all_tokens, dtype=np.uint16)


def count_existing_tokens(output_dir: Path) -> tuple[int, int]:
    """Count tokens in existing train shards. Returns (shard_count, token_count)."""
    shards = sorted(output_dir.glob("*_train_*.bin"))
    total = 0
    for f in shards:
        h = np.fromfile(f, dtype="<i4", count=3)
        if h.size >= 3:
            total += int(h[2])
    return len(shards), total


# ---------------------------------------------------------------------------
# Pretraining dataset download + tokenization
# ---------------------------------------------------------------------------

def download_and_tokenize_pretraining(
    name: str,
    hf_id: str,
    subsets: list[str] | None,
    output_dir: str,
    target_tokens: int,
    shard_size: int,
    num_workers: int,
    batch_docs: int,
) -> dict:
    """Download a pretraining dataset, tokenize with GPT-2, write .bin shards."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resume-safe: skip if shards already exist
    existing_shards, existing_tokens = count_existing_tokens(out)
    if existing_shards > 0:
        log(f"[{name}] SKIP: {existing_shards} train shards already exist "
            f"({existing_tokens / 1e9:.2f}B tokens) in {out}")
        return {
            "name": name, "dir": str(out), "train_shards": existing_shards,
            "tokens": existing_tokens, "status": "skipped",
        }

    log(f"[{name}] Starting: {hf_id} (target: {target_tokens / 1e9:.0f}B tokens)")

    token_buf = np.empty(shard_size, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    train_shard_count = 0
    val_shard_count = 0
    t0 = time.time()
    last_progress_tokens = 0

    prefix = name

    def flush_shard() -> None:
        nonlocal buf_pos, total_tokens, train_shard_count, val_shard_count
        if val_shard_count < 1:
            path = out / f"{prefix}_val_{val_shard_count:06d}.bin"
            val_shard_count += 1
        else:
            path = out / f"{prefix}_train_{train_shard_count:06d}.bin"
            train_shard_count += 1
        write_shard(path, token_buf[:buf_pos])
        total_tokens += buf_pos
        buf_pos = 0

    def ingest(chunk: np.ndarray) -> bool:
        """Ingest a token chunk into the shard buffer. Returns True if target reached."""
        nonlocal buf_pos, last_progress_tokens
        pos = 0
        while pos < len(chunk):
            space = shard_size - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos : buf_pos + n] = chunk[pos : pos + n]
            buf_pos += n
            pos += n
            if buf_pos >= shard_size:
                flush_shard()
                if total_tokens - last_progress_tokens >= 1_000_000:
                    elapsed = time.time() - t0
                    rate = total_tokens / elapsed / 1e6
                    remaining_tok = target_tokens - total_tokens
                    eta_min = remaining_tok / (rate * 1e6) / 60 if rate > 0 else float("inf")
                    log(f"[{name}] {total_tokens / 1e9:.3f}B tok, "
                        f"{train_shard_count} shards, {rate:.1f}M tok/s, "
                        f"ETA: {eta_min:.0f}min")
                    last_progress_tokens = total_tokens
                if total_tokens >= target_tokens:
                    return True
        return False

    def doc_batches(dataset, batch_size: int):
        batch: list[str] = []
        for doc in dataset:
            text = doc.get("text", "")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def load_with_retry(hf_id: str, subset: str | None, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                kwargs = {"path": hf_id, "split": "train", "streaming": True}
                if subset is not None:
                    kwargs["name"] = subset
                return load_dataset(**kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    log(f"[{name}] Retry {attempt + 1}/{max_retries} "
                        f"{'(' + subset + ') ' if subset else ''}: {e}. "
                        f"Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    try:
        subsets_to_load = subsets if subsets else [None]
        for subset in subsets_to_load:
            if total_tokens >= target_tokens:
                break
            if subset:
                log(f"[{name}] Loading subset: {subset}")
            ds = load_with_retry(hf_id, subset)

            pool_workers = min(num_workers, 4)
            with Pool(pool_workers) as pool:
                for token_chunk in pool.imap(
                    tokenize_batch,
                    doc_batches(ds, batch_docs),
                    chunksize=4,
                ):
                    if ingest(token_chunk):
                        break

        # Flush remaining buffer
        if buf_pos > 0:
            flush_shard()

        elapsed = time.time() - t0
        log(f"[{name}] DONE: {total_tokens / 1e9:.2f}B tokens, "
            f"{train_shard_count} train + {val_shard_count} val shards, "
            f"{elapsed:.0f}s")
        return {
            "name": name, "dir": str(out),
            "train_shards": train_shard_count, "val_shards": val_shard_count,
            "tokens": total_tokens, "status": "complete",
        }

    except Exception as e:
        log(f"[{name}] ERROR: {e}\n{traceback.format_exc()}")
        if buf_pos > 0:
            flush_shard()
        return {
            "name": name, "dir": str(out),
            "train_shards": train_shard_count, "val_shards": val_shard_count,
            "tokens": total_tokens, "status": f"error: {e}",
        }


# ---------------------------------------------------------------------------
# SFT dataset download (JSONL)
# ---------------------------------------------------------------------------

def download_sft(name: str, hf_id: str, output_path: str) -> dict:
    """Download an SFT dataset and save each row as JSONL."""
    from datasets import load_dataset

    out = Path(output_path)
    if out.exists() and out.stat().st_size > 0:
        lines = sum(1 for _ in open(out))
        log(f"[{name}] SKIP: {output_path} already exists ({lines} examples)")
        return {"name": name, "path": str(out), "examples": lines, "status": "skipped"}

    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{name}] Downloading SFT: {hf_id}")

    for attempt in range(3):
        try:
            ds = load_dataset(hf_id, split="train", streaming=True)
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                log(f"[{name}] Retry {attempt + 1}/3: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log(f"[{name}] FAILED after 3 attempts: {e}")
                return {"name": name, "path": str(out), "examples": 0, "status": f"error: {e}"}

    count = 0
    t0 = time.time()
    with open(out, "w") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100_000 == 0:
                elapsed = time.time() - t0
                log(f"[{name}] {count} examples saved ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    log(f"[{name}] DONE: {count} examples in {elapsed:.0f}s -> {out}")
    return {"name": name, "path": str(out), "examples": count, "status": "complete"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mixed training data for Airyn")
    parser.add_argument("--workers", type=int, default=0, help="Tokenizer workers (0=auto)")
    parser.add_argument("--batch-docs", type=int, default=1000, help="Docs per tokenizer batch")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Tokens per shard")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else max(4, min(8, cpu_count() - 2))

    # Ensure dependencies
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        os.system(f"{sys.executable} -m pip install tiktoken --break-system-packages")
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets --break-system-packages")
        from datasets import load_dataset  # noqa: F401

    log(f"Data preparation starting with {num_workers} tokenizer workers")

    pretraining_configs = [
        {
            "name": "smollm_corpus",
            "hf_id": "HuggingFaceTB/smollm-corpus",
            "subsets": ["cosmopedia-v2", "python-edu", "fineweb-edu-dedup"],
            "output_dir": "data/smollm_corpus",
            "target_tokens": 100_000_000_000,
        },
        {
            "name": "proof_pile2",
            "hf_id": "EleutherAI/proof-pile-2",
            "subsets": ["arxiv", "open-web-math", "algebraic-stack"],
            "output_dir": "data/proof_pile2",
            "target_tokens": 55_000_000_000,
        },
        {
            "name": "dolmino_mix",
            "hf_id": "allenai/dolmino-mix-1124",
            "subsets": None,
            "output_dir": "data/dolmino_mix",
            "target_tokens": 100_000_000_000,
        },
    ]

    sft_configs = [
        {"name": "tulu3_sft", "hf_id": "allenai/tulu-3-sft-mixture", "output_path": "data/sft/tulu3_sft.jsonl"},
        {"name": "openr1_math", "hf_id": "open-r1/OpenR1-Math-220k", "output_path": "data/sft/openr1_math.jsonl"},
        {"name": "dolphin_r1", "hf_id": "cognitivecomputations/dolphin-r1", "output_path": "data/sft/dolphin_r1.jsonl"},
    ]

    manifest: dict = {"datasets": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Launch pretraining downloads in parallel threads (one thread per dataset).
    # Each thread uses a multiprocessing Pool for tokenization -- CPU-only.
    log("=" * 60)
    log("PRETRAINING DATA (parallel threads, CPU-only)")
    log("=" * 60)

    futures: dict = {}
    with ThreadPoolExecutor(max_workers=len(pretraining_configs)) as executor:
        for cfg in pretraining_configs:
            f = executor.submit(
                download_and_tokenize_pretraining,
                cfg["name"], cfg["hf_id"], cfg.get("subsets"),
                cfg["output_dir"], cfg["target_tokens"],
                args.shard_size, num_workers, args.batch_docs,
            )
            futures[f] = cfg["name"]

        # While pretraining downloads are running, do SFT datasets sequentially
        log("=" * 60)
        log("SFT DATA (sequential, small)")
        log("=" * 60)

        for cfg in sft_configs:
            try:
                result = download_sft(cfg["name"], cfg["hf_id"], cfg["output_path"])
                manifest["datasets"].append(result)
            except Exception as e:
                log(f"[{cfg['name']}] FAILED: {e}")
                manifest["datasets"].append({"name": cfg["name"], "status": f"error: {e}"})

        # Wait for pretraining threads to finish
        log("=" * 60)
        log("Waiting for pretraining downloads...")
        log("=" * 60)

        for f in as_completed(futures):
            dataset_name = futures[f]
            try:
                result = f.result()
                manifest["datasets"].append(result)
            except Exception as e:
                log(f"[{dataset_name}] THREAD FAILED: {e}")
                manifest["datasets"].append({"name": dataset_name, "status": f"error: {e}"})

    manifest["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Write manifest
    manifest_path = Path("data/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Manifest written: {manifest_path}")

    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for d in manifest["datasets"]:
        if "tokens" in d:
            log(f"  {d['name']}: {d['tokens'] / 1e9:.2f}B tokens [{d['status']}]")
        elif "examples" in d:
            log(f"  {d['name']}: {d['examples']} examples [{d['status']}]")
        else:
            log(f"  {d['name']}: [{d['status']}]")


if __name__ == "__main__":
    main()
