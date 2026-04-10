"""
Simple, fast data pipeline v2.
- Downloads parquet files with parallel threads (16 per dataset)
- Tokenizes from local parquet at full CPU speed
- Appends new shards (no skip logic — starts from next shard number)
- Each dataset is a separate process with its own worker pool
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


def count_shards(output_dir, prefix):
    """Count existing train/val shards and total tokens."""
    out = Path(output_dir)
    train = sorted(out.glob(f"{prefix}_train_*.bin"))
    val = sorted(out.glob(f"{prefix}_val_*.bin"))
    total_tok = 0
    for f in train + val:
        h = np.fromfile(f, dtype="<i4", count=3)
        if h.size >= 3:
            total_tok += int(h[2])
    return len(train), len(val), total_tok


def download_parquet(name, repo_id, subset, local_dir, dl_workers=16):
    """Download parquet files from HF Hub. Returns list of local paths."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN or None)
    try:
        all_files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        log(name, f"Cannot list repo files: {e}")
        return []

    if subset:
        pq = [f for f in all_files if f.endswith(".parquet") and subset in f]
    else:
        pq = [f for f in all_files if f.endswith(".parquet")]

    if not pq:
        log(name, f"No parquet files found")
        return []

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = {f.name for f in local.rglob("*.parquet")}
    to_dl = [f for f in pq if Path(f).name not in existing]

    log(name, f"{len(pq)} parquet files, {len(to_dl)} to download ({len(existing)} cached)")

    if to_dl:
        def dl_one(fp):
            try:
                hf_hub_download(repo_id, fp, repo_type="dataset",
                                local_dir=str(local), token=HF_TOKEN or None)
                return True
            except Exception as e:
                log(name, f"  dl error {Path(fp).name}: {e}")
                return False

        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=dl_workers) as ex:
            for ok in ex.map(dl_one, to_dl):
                done += 1
                if done % 20 == 0 or done == len(to_dl):
                    rate = done / (time.time() - t0)
                    log(name, f"  downloaded {done}/{len(to_dl)} ({rate:.1f} files/s)")

    return sorted(local.rglob("*.parquet"))


def tokenize_parquet(name, parquet_files, output_dir, prefix, target_new_tokens, pool_workers, text_col="text"):
    """Tokenize parquet files into .bin shards. Appends to existing shards."""
    import pyarrow.parquet as pq

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_train, existing_val, existing_tok = count_shards(output_dir, prefix)
    train_idx = existing_train
    val_idx = existing_val
    need_val = (val_idx == 0)

    log(name, f"Existing: {existing_train} train + {existing_val} val shards ({existing_tok/1e9:.2f}B tok)")
    log(name, f"Tokenizing {len(parquet_files)} parquet files with {pool_workers} workers, "
        f"target: +{target_new_tokens/1e9:.1f}B new tokens")

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    new_tokens = 0
    t0 = time.time()
    last_progress = 0

    def flush():
        nonlocal buf_pos, new_tokens, train_idx, val_idx, need_val
        if need_val:
            path = out / f"{prefix}_val_{val_idx:06d}.bin"
            val_idx += 1
            need_val = False
        else:
            path = out / f"{prefix}_train_{train_idx:06d}.bin"
            train_idx += 1
        write_shard(path, token_buf[:buf_pos])
        new_tokens += buf_pos
        buf_pos = 0

    def ingest(chunk):
        nonlocal buf_pos, last_progress
        pos = 0
        while pos < len(chunk):
            space = SHARD_SIZE - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos:buf_pos+n] = chunk[pos:pos+n]
            buf_pos += n
            pos += n
            if buf_pos >= SHARD_SIZE:
                flush()
                if new_tokens - last_progress >= 1_000_000:
                    elapsed = time.time() - t0
                    rate = new_tokens / elapsed / 1e6
                    total = existing_tok + new_tokens
                    eta = (target_new_tokens - new_tokens) / (rate * 1e6) / 60 if rate > 0 else 0
                    log(name, f"{total/1e9:.2f}B tok (+{new_tokens/1e9:.2f}B new), "
                        f"{train_idx} shards, {rate:.1f}M tok/s, ETA: {eta:.0f}min")
                    last_progress = new_tokens
                if new_tokens >= target_new_tokens:
                    return True
        return False

    done = False
    files_processed = 0
    with Pool(pool_workers) as pool:
        for pf in parquet_files:
            if done:
                break
            try:
                table = pq.read_table(pf, columns=[text_col])
                texts = table.column(text_col).to_pylist()
                batches = [texts[i:i+2000] for i in range(0, len(texts), 2000)]
                for chunk in pool.imap(tokenize_batch, batches, chunksize=4):
                    if ingest(chunk):
                        done = True
                        break
                files_processed += 1
            except Exception as e:
                log(name, f"Error in {pf.name}: {e}")
                continue

    if buf_pos > 0:
        flush()

    elapsed = time.time() - t0
    total = existing_tok + new_tokens
    log(name, f"DONE: {total/1e9:.2f}B tok total (+{new_tokens/1e9:.2f}B new), "
        f"{train_idx} train shards, {files_processed} parquet files, {elapsed:.0f}s")


def run_dataset(name, repo_id, subset, parquet_dir, output_dir, prefix,
                target_new_tokens, pool_workers, dl_workers=16, text_col="text"):
    """Full pipeline for one dataset."""
    log(name, f"=== DOWNLOAD: {repo_id} ===")
    files = download_parquet(name, repo_id, subset, parquet_dir, dl_workers)
    if not files:
        log(name, "No parquet files, falling back to streaming")
        run_streaming(name, repo_id, subset, output_dir, prefix, target_new_tokens, pool_workers, text_col)
        return
    log(name, f"=== TOKENIZE: {len(files)} parquet files ===")
    tokenize_parquet(name, files, output_dir, prefix, target_new_tokens, pool_workers, text_col)


def run_streaming(name, hf_id, subset, output_dir, prefix, target_new_tokens, pool_workers, text_col="text"):
    """Stream + tokenize fallback."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing_train, existing_val, existing_tok = count_shards(output_dir, prefix)
    train_idx = existing_train
    val_idx = existing_val
    need_val = (val_idx == 0)

    log(name, f"Streaming: {hf_id} subset={subset}, {pool_workers} workers, "
        f"existing: {existing_tok/1e9:.2f}B, target: +{target_new_tokens/1e9:.1f}B")

    kwargs = {"path": hf_id, "split": "train", "streaming": True, "token": HF_TOKEN or None}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(**kwargs)

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    new_tokens = 0
    t0 = time.time()
    last_progress = 0

    def flush():
        nonlocal buf_pos, new_tokens, train_idx, val_idx, need_val
        if need_val:
            path = out / f"{prefix}_val_{val_idx:06d}.bin"
            val_idx += 1
            need_val = False
        else:
            path = out / f"{prefix}_train_{train_idx:06d}.bin"
            train_idx += 1
        write_shard(path, token_buf[:buf_pos])
        new_tokens += buf_pos
        buf_pos = 0

    def ingest(chunk):
        nonlocal buf_pos, last_progress
        pos = 0
        while pos < len(chunk):
            space = SHARD_SIZE - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos:buf_pos+n] = chunk[pos:pos+n]
            buf_pos += n
            pos += n
            if buf_pos >= SHARD_SIZE:
                flush()
                if new_tokens - last_progress >= 1_000_000:
                    elapsed = time.time() - t0
                    rate = new_tokens / elapsed / 1e6
                    total = existing_tok + new_tokens
                    eta = (target_new_tokens - new_tokens) / (rate * 1e6) / 60 if rate > 0 else 0
                    log(name, f"{total/1e9:.2f}B tok (+{new_tokens/1e9:.2f}B new), "
                        f"{train_idx} shards, {rate:.1f}M tok/s, ETA: {eta:.0f}min")
                    last_progress = new_tokens
                if new_tokens >= target_new_tokens:
                    return True
        return False

    def doc_batches(dataset, batch_size=2000):
        batch = []
        for doc in dataset:
            text = doc.get(text_col, "")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    try:
        with Pool(pool_workers) as pool:
            for chunk in pool.imap(tokenize_batch, doc_batches(ds), chunksize=8):
                if ingest(chunk):
                    break
        if buf_pos > 0:
            flush()
        elapsed = time.time() - t0
        total = existing_tok + new_tokens
        log(name, f"DONE: {total/1e9:.2f}B tok (+{new_tokens/1e9:.2f}B new), {elapsed:.0f}s")
    except Exception as e:
        log(name, f"ERROR: {e}\n{traceback.format_exc()}")
        if buf_pos > 0:
            flush()


def main():
    cores = cpu_count()

    for pkg in ["zstandard", "pyarrow", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            os.system(f"{sys.executable} -m pip install {pkg} --break-system-packages -q")

    budget = max(cores - 16, 64)
    log("main", f"Cores: {cores}, budget: {budget}, HF_TOKEN: {'yes' if HF_TOKEN else 'NO'}")

    # Calculate how much more each dataset needs
    # smollm_corpus: has 45.7B, want 100B -> need 54.3B more
    # dolmino_mix: has ~4.7B, want 100B -> need ~95B more
    # open_web_math: has ~13.5B, want 15B -> need ~1.5B more

    processes = []

    # SmolLM subsets — each as separate process, share output dir
    # cosmopedia-v2 text col = "text", python-edu has "content", fineweb-edu has "text"
    smollm_w = budget // 3
    for subset, text_col in [("cosmopedia-v2", "text"), ("fineweb-edu-dedup", "text")]:
        safe = subset.replace("-", "_")
        p = Process(target=run_dataset, args=(
            f"smollm_{safe}", "HuggingFaceTB/smollm-corpus", subset,
            f"data/parquet/smollm_{safe}", "data/smollm_corpus", "smollm_corpus",
            30_000_000_000, smollm_w, 16, text_col,
        ))
        processes.append((f"smollm_{safe}", p))

    # Dolmino — streaming (parquet has zstd, might need special handling)
    p = Process(target=run_streaming, args=(
        "dolmino_mix", "allenai/dolmino-mix-1124", None,
        "data/dolmino_mix", "dolmino_mix", 95_000_000_000, budget // 3, "text",
    ))
    processes.append(("dolmino_mix", p))

    # Open-web-math — just needs 1.5B more (parquet already downloaded)
    p = Process(target=run_dataset, args=(
        "open_web_math", "open-web-math/open-web-math", None,
        "data/parquet/open_web_math", "data/open_web_math", "open_web_math",
        2_000_000_000, budget // 6, 16, "text",
    ))
    processes.append(("open_web_math", p))

    log("main", f"Launching {len(processes)} processes...")
    for name, p in processes:
        p.start()
        log("main", f"  {name} (pid={p.pid})")

    for name, p in processes:
        p.join()
        log("main", f"  Done: {name} (exit={p.exitcode})")

    log("main", "All complete!")


if __name__ == "__main__":
    main()
