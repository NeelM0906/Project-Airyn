"""
Fast parallel data pipeline — separates download from tokenization.

Strategy:
  1. Use huggingface_hub to download parquet files with parallel threads
  2. Tokenize locally from parquet at full CPU speed (no network wait)

This saturates both network (parallel downloads) and CPU (parallel tokenization).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing import Pool, Process, cpu_count
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
SHARD_SIZE = 100_000_000

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def log(name, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{name}] {msg}", flush=True)


def write_shard(path, tokens):
    assert tokens.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def tokenize_batch(texts):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    all_tokens = []
    for text in texts:
        all_tokens.append(eot)
        all_tokens.extend(enc.encode_ordinary(text))
    return np.array(all_tokens, dtype=np.uint16)


def count_existing_tokens(output_dir):
    shards = sorted(output_dir.glob("*_train_*.bin"))
    total = 0
    for f in shards:
        h = np.fromfile(f, dtype="<i4", count=3)
        if h.size >= 3:
            total += int(h[2])
    return len(shards), total


def tokenize_from_parquet_files(name, parquet_dir, output_dir, target_tokens, pool_workers):
    """Tokenize pre-downloaded parquet files into .bin shards."""
    import pyarrow.parquet as pq

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_shards, existing_tokens = count_existing_tokens(out)
    if existing_tokens >= target_tokens:
        log(name, f"COMPLETE: {existing_shards} shards, {existing_tokens / 1e9:.2f}B tok")
        return

    parquet_files = sorted(Path(parquet_dir).rglob("*.parquet"))
    if not parquet_files:
        log(name, f"No parquet files found in {parquet_dir}")
        return

    log(name, f"Tokenizing {len(parquet_files)} parquet files with {pool_workers} workers "
        f"(have {existing_tokens / 1e9:.2f}B, need {target_tokens / 1e9:.0f}B)")

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = existing_tokens
    train_shard_count = len(list(out.glob(f"{name}_train_*.bin")))
    val_shard_count = len(list(out.glob(f"{name}_val_*.bin")))
    tokens_to_skip = existing_tokens
    t0 = time.time()
    last_progress = total_tokens

    def flush_shard():
        nonlocal buf_pos, total_tokens, train_shard_count, val_shard_count
        if val_shard_count < 1:
            path = out / f"{name}_val_{val_shard_count:06d}.bin"
            val_shard_count += 1
        else:
            path = out / f"{name}_train_{train_shard_count:06d}.bin"
            train_shard_count += 1
        write_shard(path, token_buf[:buf_pos])
        total_tokens += buf_pos
        buf_pos = 0

    def ingest(chunk):
        nonlocal buf_pos, last_progress
        pos = 0
        while pos < len(chunk):
            space = SHARD_SIZE - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos:buf_pos + n] = chunk[pos:pos + n]
            buf_pos += n
            pos += n
            if buf_pos >= SHARD_SIZE:
                flush_shard()
                if total_tokens - last_progress >= 1_000_000:
                    elapsed = time.time() - t0
                    new_tokens = total_tokens - existing_tokens
                    rate = new_tokens / elapsed / 1e6 if elapsed > 0 else 0
                    remaining = (target_tokens - total_tokens)
                    eta = remaining / (rate * 1e6) / 60 if rate > 0 else 0
                    log(name, f"{total_tokens / 1e9:.3f}B tok, {train_shard_count} shards, "
                        f"{rate:.1f}M tok/s, ETA: {eta:.0f}min")
                    last_progress = total_tokens
                if total_tokens >= target_tokens:
                    return True
        return False

    done = False
    with Pool(pool_workers) as pool:
        for pf in parquet_files:
            if done:
                break
            try:
                table = pq.read_table(pf, columns=["text"])
                texts = table.column("text").to_pylist()
                # Process in batches
                batch_size = 2000
                batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
                for chunk in pool.imap(tokenize_batch, batches, chunksize=4):
                    if tokens_to_skip > 0:
                        tokens_to_skip -= len(chunk)
                        continue
                    if ingest(chunk):
                        done = True
                        break
            except Exception as e:
                log(name, f"Error reading {pf.name}: {e}")
                continue

    if buf_pos > 0:
        flush_shard()

    elapsed = time.time() - t0
    new_tokens = total_tokens - existing_tokens
    log(name, f"DONE: {total_tokens / 1e9:.2f}B total tok, +{new_tokens / 1e9:.2f}B new, "
        f"{train_shard_count} shards, {elapsed:.0f}s")


def download_hf_dataset(name, repo_id, subset, local_dir, n_download_workers=16):
    """Download dataset parquet files using huggingface_hub with parallel threads."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN or None)
    log(name, f"Listing files in {repo_id}" + (f" ({subset})" if subset else ""))

    try:
        all_files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        log(name, f"Failed to list files: {e}")
        return False

    # Filter for parquet/data files in the right subset
    if subset:
        parquet_files = [f for f in all_files
                         if f.endswith(".parquet") and subset in f]
        if not parquet_files:
            # Try with data/ prefix
            parquet_files = [f for f in all_files
                             if f.endswith(".parquet") and f"data/{subset}" in f]
        if not parquet_files:
            parquet_files = [f for f in all_files if f.endswith(".parquet")]
    else:
        parquet_files = [f for f in all_files if f.endswith(".parquet")]

    if not parquet_files:
        log(name, f"No parquet files found in {repo_id}, falling back to streaming")
        return False

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    # Check which files already exist
    existing = set()
    for f in local.rglob("*.parquet"):
        existing.add(f.name)

    to_download = [f for f in parquet_files if Path(f).name not in existing]
    log(name, f"Found {len(parquet_files)} parquet files, {len(to_download)} to download")

    if not to_download:
        return True

    def download_one(file_path):
        try:
            hf_hub_download(
                repo_id, file_path, repo_type="dataset",
                local_dir=str(local), token=HF_TOKEN or None,
            )
            return file_path, True
        except Exception as e:
            return file_path, f"error: {e}"

    downloaded = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_download_workers) as executor:
        futures = {executor.submit(download_one, f): f for f in to_download}
        for future in as_completed(futures):
            path, result = future.result()
            downloaded += 1
            if downloaded % 10 == 0 or downloaded == len(to_download):
                elapsed = time.time() - t0
                rate = downloaded / elapsed if elapsed > 0 else 0
                log(name, f"Downloaded {downloaded}/{len(to_download)} files ({rate:.1f} files/s)")
            if result is not True:
                log(name, f"  Failed: {Path(path).name}: {result}")

    return True


def run_dataset(name, repo_id, subset, data_dir, output_dir, target_tokens, pool_workers, dl_workers=16):
    """Full pipeline: download parquet files, then tokenize."""
    parquet_dir = os.path.join(data_dir, "parquet", name)

    # Step 1: Download
    log(name, "=== PHASE 1: DOWNLOAD ===")
    ok = download_hf_dataset(name, repo_id, subset, parquet_dir, dl_workers)
    if not ok:
        # Fallback to streaming
        log(name, "Parquet download failed, falling back to streaming mode")
        run_streaming_fallback(name, repo_id, subset, output_dir, target_tokens, pool_workers)
        return

    # Step 2: Tokenize
    log(name, "=== PHASE 2: TOKENIZE ===")
    tokenize_from_parquet_files(name, parquet_dir, output_dir, target_tokens, pool_workers)


def run_streaming_fallback(name, hf_id, subset, output_dir, target_tokens, pool_workers):
    """Fallback: stream + tokenize (slower, network-bound)."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_shards, existing_tokens = count_existing_tokens(out)
    if existing_tokens >= target_tokens:
        log(name, f"COMPLETE: {existing_tokens / 1e9:.2f}B tok")
        return

    log(name, f"Streaming fallback: {hf_id} subset={subset} with {pool_workers} workers")

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = existing_tokens
    train_shard_count = len(list(out.glob(f"{name}_train_*.bin")))
    val_shard_count = len(list(out.glob(f"{name}_val_*.bin")))
    tokens_to_skip = existing_tokens
    t0 = time.time()
    last_progress = total_tokens

    def flush_shard():
        nonlocal buf_pos, total_tokens, train_shard_count, val_shard_count
        if val_shard_count < 1:
            path = out / f"{name}_val_{val_shard_count:06d}.bin"
            val_shard_count += 1
        else:
            path = out / f"{name}_train_{train_shard_count:06d}.bin"
            train_shard_count += 1
        write_shard(path, token_buf[:buf_pos])
        total_tokens += buf_pos
        buf_pos = 0

    def ingest(chunk):
        nonlocal buf_pos, last_progress
        pos = 0
        while pos < len(chunk):
            space = SHARD_SIZE - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos:buf_pos + n] = chunk[pos:pos + n]
            buf_pos += n
            pos += n
            if buf_pos >= SHARD_SIZE:
                flush_shard()
                if total_tokens - last_progress >= 1_000_000:
                    elapsed = time.time() - t0
                    new_tok = total_tokens - existing_tokens
                    rate = new_tok / elapsed / 1e6 if elapsed > 0 else 0
                    remaining = (target_tokens - total_tokens)
                    eta = remaining / (rate * 1e6) / 60 if rate > 0 else 0
                    log(name, f"{total_tokens / 1e9:.3f}B tok, {train_shard_count} shards, "
                        f"{rate:.1f}M tok/s, ETA: {eta:.0f}min")
                    last_progress = total_tokens
                if total_tokens >= target_tokens:
                    return True
        return False

    def doc_batches(dataset, batch_size=2000):
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

    try:
        kwargs = {"path": hf_id, "split": "train", "streaming": True, "token": HF_TOKEN or None}
        if subset:
            kwargs["name"] = subset
        ds = load_dataset(**kwargs)

        with Pool(pool_workers) as pool:
            for chunk in pool.imap(tokenize_batch, doc_batches(ds), chunksize=8):
                if tokens_to_skip > 0:
                    tokens_to_skip -= len(chunk)
                    continue
                if ingest(chunk):
                    break

        if buf_pos > 0:
            flush_shard()

        elapsed = time.time() - t0
        log(name, f"DONE: {total_tokens / 1e9:.2f}B tok, {train_shard_count} shards, {elapsed:.0f}s")
    except Exception as e:
        log(name, f"ERROR: {e}\n{traceback.format_exc()}")
        if buf_pos > 0:
            flush_shard()


def run_sft(name, hf_id, output_path, config=None):
    from datasets import load_dataset

    out = Path(output_path)
    if out.exists() and out.stat().st_size > 0:
        lines = sum(1 for _ in open(out))
        log(name, f"SKIP: already exists ({lines} examples)")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    log(name, f"Downloading: {hf_id} config={config}")

    for attempt in range(3):
        try:
            kwargs = {"path": hf_id, "split": "train", "streaming": True,
                      "token": HF_TOKEN or None}
            if config:
                kwargs["name"] = config
            ds = load_dataset(**kwargs)
            break
        except Exception as e:
            if attempt < 2:
                log(name, f"Retry {attempt + 1}/3: {e}")
                time.sleep(2 ** (attempt + 1))
            else:
                log(name, f"FAILED: {e}")
                return

    count = 0
    t0 = time.time()
    with open(out, "w") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100_000 == 0:
                log(name, f"{count} examples ({time.time() - t0:.0f}s)")

    log(name, f"DONE: {count} examples in {time.time() - t0:.0f}s")


def main():
    total_cores = cpu_count()

    # Ensure deps
    for pkg in ["zstandard", "pyarrow", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            os.system(f"{sys.executable} -m pip install {pkg} --break-system-packages -q")

    # Worker allocation
    pretrain_budget = max(total_cores - 16, 64)
    smollm_workers = pretrain_budget // 2
    dolmino_workers = pretrain_budget // 4
    owm_workers = pretrain_budget // 4

    log("main", f"Cores: {total_cores}, workers: smollm={smollm_workers} dolmino={dolmino_workers} owm={owm_workers}")
    log("main", f"HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET'}")

    # Each dataset runs as a separate process
    processes = []

    # SmolLM Corpus — 3 subsets, download parquet then tokenize
    for subset in ["cosmopedia-v2", "python-edu", "fineweb-edu-dedup"]:
        safe = subset.replace("-", "_")
        p = Process(target=run_dataset, args=(
            f"smollm_{safe}", "HuggingFaceTB/smollm-corpus", subset,
            "data", "data/smollm_corpus", 100_000_000_000,
            smollm_workers // 3, 16,
        ))
        processes.append((f"smollm_{safe}", p))

    # Dolmino — single dataset
    p = Process(target=run_dataset, args=(
        "dolmino_mix", "allenai/dolmino-mix-1124", None,
        "data", "data/dolmino_mix", 100_000_000_000,
        dolmino_workers, 16,
    ))
    processes.append(("dolmino_mix", p))

    # Open-Web-Math
    p = Process(target=run_dataset, args=(
        "open_web_math", "open-web-math/open-web-math", None,
        "data", "data/open_web_math", 15_000_000_000,
        owm_workers, 16,
    ))
    processes.append(("open_web_math", p))

    log("main", f"Launching {len(processes)} processes...")
    for name, p in processes:
        p.start()
        log("main", f"  Started: {name} (pid={p.pid})")

    for name, p in processes:
        p.join()
        log("main", f"  Finished: {name} (exit={p.exitcode})")

    log("main", "All done!")


if __name__ == "__main__":
    main()
