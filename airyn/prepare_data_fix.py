"""
Re-run the 3 failed datasets from prepare_data_mix.py:
  1. proof_pile2  (fix: trust_remote_code=True)
  2. dolmino_mix  (fix: zstandard installed)
  3. dolphin_r1   (fix: config name specified)

SmolLM corpus is skipped (already running in another tmux session).
"""
from __future__ import annotations

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

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
SHARD_SIZE = 100_000_000

_print_lock = Lock()

def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    all_tokens: list[int] = []
    for text in texts:
        all_tokens.append(eot)
        all_tokens.extend(enc.encode_ordinary(text))
    return np.array(all_tokens, dtype=np.uint16)


def count_existing_tokens(output_dir: Path) -> tuple[int, int]:
    shards = sorted(output_dir.glob("*_train_*.bin"))
    total = 0
    for f in shards:
        h = np.fromfile(f, dtype="<i4", count=3)
        if h.size >= 3:
            total += int(h[2])
    return len(shards), total


def download_and_tokenize(
    name: str, hf_id: str, subsets: list[str] | None,
    output_dir: str, target_tokens: int, num_workers: int, batch_docs: int,
) -> dict:
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_shards, existing_tokens = count_existing_tokens(out)
    if existing_shards > 0:
        log(f"[{name}] SKIP: {existing_shards} shards already exist ({existing_tokens/1e9:.2f}B tok)")
        return {"name": name, "dir": str(out), "train_shards": existing_shards,
                "tokens": existing_tokens, "status": "skipped"}

    log(f"[{name}] Starting: {hf_id} (target: {target_tokens/1e9:.0f}B tokens)")

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    train_shard_count = 0
    val_shard_count = 0
    t0 = time.time()
    last_progress = 0
    prefix = name

    def flush_shard():
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
        nonlocal buf_pos, last_progress
        pos = 0
        while pos < len(chunk):
            space = SHARD_SIZE - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos:buf_pos+n] = chunk[pos:pos+n]
            buf_pos += n
            pos += n
            if buf_pos >= SHARD_SIZE:
                flush_shard()
                if total_tokens - last_progress >= 1_000_000:
                    elapsed = time.time() - t0
                    rate = total_tokens / elapsed / 1e6
                    remaining = (target_tokens - total_tokens) / (rate * 1e6) / 60 if rate > 0 else float("inf")
                    log(f"[{name}] {total_tokens/1e9:.3f}B tok, {train_shard_count} shards, "
                        f"{rate:.1f}M tok/s, ETA: {remaining:.0f}min")
                    last_progress = total_tokens
                if total_tokens >= target_tokens:
                    return True
        return False

    def doc_batches(dataset, batch_size):
        batch = []
        for doc in dataset:
            text = doc.get("text", "")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def load_with_retry(hf_id, subset, max_retries=3):
        for attempt in range(max_retries):
            try:
                kwargs = {"path": hf_id, "split": "train", "streaming": True,
                          "trust_remote_code": True}
                if subset is not None:
                    kwargs["name"] = subset
                return load_dataset(**kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    log(f"[{name}] Retry {attempt+1}/{max_retries} "
                        f"{'('+subset+') ' if subset else ''}: {e}. Waiting {wait}s...")
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
                for token_chunk in pool.imap(tokenize_batch, doc_batches(ds, batch_docs), chunksize=4):
                    if ingest(token_chunk):
                        break

        if buf_pos > 0:
            flush_shard()

        elapsed = time.time() - t0
        log(f"[{name}] DONE: {total_tokens/1e9:.2f}B tokens, "
            f"{train_shard_count} train + {val_shard_count} val shards, {elapsed:.0f}s")
        return {"name": name, "dir": str(out), "train_shards": train_shard_count,
                "val_shards": val_shard_count, "tokens": total_tokens, "status": "complete"}
    except Exception as e:
        log(f"[{name}] ERROR: {e}\n{traceback.format_exc()}")
        if buf_pos > 0:
            flush_shard()
        return {"name": name, "dir": str(out), "train_shards": train_shard_count,
                "val_shards": val_shard_count, "tokens": total_tokens, "status": f"error: {e}"}


def download_sft(name, hf_id, output_path, config=None):
    from datasets import load_dataset

    out = Path(output_path)
    if out.exists() and out.stat().st_size > 0:
        lines = sum(1 for _ in open(out))
        log(f"[{name}] SKIP: already exists ({lines} examples)")
        return {"name": name, "path": str(out), "examples": lines, "status": "skipped"}

    out.parent.mkdir(parents=True, exist_ok=True)
    label = f"{hf_id} ({config})" if config else hf_id
    log(f"[{name}] Downloading SFT: {label}")

    for attempt in range(3):
        try:
            kwargs = {"path": hf_id, "split": "train", "streaming": True,
                      "trust_remote_code": True}
            if config:
                kwargs["name"] = config
            ds = load_dataset(**kwargs)
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                log(f"[{name}] Retry {attempt+1}/3: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log(f"[{name}] FAILED: {e}")
                return {"name": name, "path": str(out), "examples": 0, "status": f"error: {e}"}

    count = 0
    t0 = time.time()
    with open(out, "w") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100_000 == 0:
                log(f"[{name}] {count} examples saved ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    log(f"[{name}] DONE: {count} examples in {elapsed:.0f}s -> {out}")
    return {"name": name, "path": str(out), "examples": count, "status": "complete"}


def main():
    num_workers = max(4, min(8, cpu_count() - 2))

    # Install zstandard for dolmino
    try:
        import zstandard  # noqa: F401
    except ImportError:
        log("Installing zstandard...")
        os.system(f"{sys.executable} -m pip install zstandard --break-system-packages -q")

    log(f"Fixing 3 failed datasets with {num_workers} workers")

    results = []

    # Launch pretraining datasets in parallel
    pretraining = [
        {"name": "proof_pile2", "hf_id": "EleutherAI/proof-pile-2",
         "subsets": ["arxiv", "open-web-math", "algebraic-stack"],
         "output_dir": "data/proof_pile2", "target_tokens": 55_000_000_000},
        {"name": "dolmino_mix", "hf_id": "allenai/dolmino-mix-1124",
         "subsets": None,
         "output_dir": "data/dolmino_mix", "target_tokens": 100_000_000_000},
    ]

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for cfg in pretraining:
            f = executor.submit(
                download_and_tokenize,
                cfg["name"], cfg["hf_id"], cfg.get("subsets"),
                cfg["output_dir"], cfg["target_tokens"],
                num_workers, 1000,
            )
            futures[f] = cfg["name"]

        # While those run, download dolphin_r1 SFT
        log("=" * 60)
        log("SFT: Dolphin-R1 (reasoning + nonreasoning)")
        log("=" * 60)
        for config_name in ["reasoning", "nonreasoning"]:
            r = download_sft(
                f"dolphin_r1_{config_name}",
                "cognitivecomputations/dolphin-r1",
                f"data/sft/dolphin_r1_{config_name}.jsonl",
                config=config_name,
            )
            results.append(r)

        log("=" * 60)
        log("Waiting for pretraining downloads...")
        log("=" * 60)
        for f in as_completed(futures):
            name = futures[f]
            try:
                results.append(f.result())
            except Exception as e:
                log(f"[{name}] THREAD FAILED: {e}")
                results.append({"name": name, "status": f"error: {e}"})

    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for d in results:
        if "tokens" in d:
            log(f"  {d['name']}: {d['tokens']/1e9:.2f}B tokens [{d['status']}]")
        elif "examples" in d:
            log(f"  {d['name']}: {d['examples']} examples [{d['status']}]")
        else:
            log(f"  {d['name']}: [{d['status']}]")


if __name__ == "__main__":
    main()
