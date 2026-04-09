"""
Fix round 2: datasets that still failed.
  1. open-web-math (replaces proof-pile-2 which uses deprecated loading script)
  2. dolphin-r1 reasoning-deepseek + reasoning-flash configs
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from threading import Lock

import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
SHARD_SIZE = 100_000_000

_print_lock = Lock()

def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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


def download_and_tokenize(name, hf_id, output_dir, target_tokens, num_workers):
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_shards, existing_tokens = count_existing_tokens(out)
    if existing_shards > 0:
        log(f"[{name}] SKIP: {existing_shards} shards ({existing_tokens/1e9:.2f}B tok)")
        return {"name": name, "tokens": existing_tokens, "status": "skipped"}

    log(f"[{name}] Starting: {hf_id} (target: {target_tokens/1e9:.0f}B tokens)")

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    train_shard_count = 0
    val_shard_count = 0
    t0 = time.time()
    last_progress = 0

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
            token_buf[buf_pos:buf_pos+n] = chunk[pos:pos+n]
            buf_pos += n
            pos += n
            if buf_pos >= SHARD_SIZE:
                flush_shard()
                if total_tokens - last_progress >= 1_000_000:
                    elapsed = time.time() - t0
                    rate = total_tokens / elapsed / 1e6
                    eta = (target_tokens - total_tokens) / (rate * 1e6) / 60 if rate > 0 else float("inf")
                    log(f"[{name}] {total_tokens/1e9:.3f}B tok, {train_shard_count} shards, "
                        f"{rate:.1f}M tok/s, ETA: {eta:.0f}min")
                    last_progress = total_tokens
                if total_tokens >= target_tokens:
                    return True
        return False

    def doc_batches(dataset, batch_size=1000):
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
        for attempt in range(3):
            try:
                ds = load_dataset(hf_id, split="train", streaming=True)
                break
            except Exception as e:
                if attempt < 2:
                    log(f"[{name}] Retry {attempt+1}/3: {e}")
                    time.sleep(2 ** (attempt+1))
                else:
                    raise

        with Pool(min(num_workers, 4)) as pool:
            for chunk in pool.imap(tokenize_batch, doc_batches(ds), chunksize=4):
                if ingest(chunk):
                    break

        if buf_pos > 0:
            flush_shard()

        elapsed = time.time() - t0
        log(f"[{name}] DONE: {total_tokens/1e9:.2f}B tok, {train_shard_count} train shards, {elapsed:.0f}s")
        return {"name": name, "tokens": total_tokens, "status": "complete"}
    except Exception as e:
        log(f"[{name}] ERROR: {e}\n{traceback.format_exc()}")
        if buf_pos > 0:
            flush_shard()
        return {"name": name, "tokens": total_tokens, "status": f"error: {e}"}


def download_sft(name, hf_id, output_path, config=None):
    from datasets import load_dataset

    out = Path(output_path)
    if out.exists() and out.stat().st_size > 0:
        lines = sum(1 for _ in open(out))
        log(f"[{name}] SKIP: already exists ({lines} examples)")
        return {"name": name, "examples": lines, "status": "skipped"}

    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{name}] Downloading: {hf_id} config={config}")

    for attempt in range(3):
        try:
            kwargs = {"path": hf_id, "split": "train", "streaming": True}
            if config:
                kwargs["name"] = config
            ds = load_dataset(**kwargs)
            break
        except Exception as e:
            if attempt < 2:
                log(f"[{name}] Retry {attempt+1}/3: {e}")
                time.sleep(2 ** (attempt+1))
            else:
                log(f"[{name}] FAILED: {e}")
                return {"name": name, "examples": 0, "status": f"error: {e}"}

    count = 0
    t0 = time.time()
    with open(out, "w") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100_000 == 0:
                log(f"[{name}] {count} examples ({time.time()-t0:.0f}s)")

    log(f"[{name}] DONE: {count} examples in {time.time()-t0:.0f}s")
    return {"name": name, "examples": count, "status": "complete"}


def main():
    num_workers = max(4, min(8, cpu_count() - 2))
    log(f"Fix round 2: open-web-math + dolphin-r1 configs ({num_workers} workers)")

    results = []

    # 1. Download open-web-math as replacement for proof-pile-2
    log("=" * 60)
    log("PRETRAINING: open-web-math (replaces proof-pile-2)")
    log("=" * 60)

    # Start open-web-math in background thread
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=1) as executor:
        owm_future = executor.submit(
            download_and_tokenize,
            "open_web_math", "open-web-math/open-web-math",
            "data/open_web_math", 15_000_000_000, num_workers,
        )

        # 2. While that runs, download dolphin-r1 reasoning configs
        log("=" * 60)
        log("SFT: dolphin-r1 reasoning-deepseek + reasoning-flash")
        log("=" * 60)

        for cfg in ["reasoning-deepseek", "reasoning-flash"]:
            safe_name = cfg.replace("-", "_")
            r = download_sft(
                f"dolphin_r1_{safe_name}",
                "cognitivecomputations/dolphin-r1",
                f"data/sft/dolphin_r1_{safe_name}.jsonl",
                config=cfg,
            )
            results.append(r)

        # Wait for open-web-math
        try:
            results.append(owm_future.result())
        except Exception as e:
            log(f"[open_web_math] FAILED: {e}")
            results.append({"name": "open_web_math", "status": f"error: {e}"})

    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for d in results:
        if "tokens" in d:
            log(f"  {d['name']}: {d['tokens']/1e9:.2f}B tok [{d['status']}]")
        elif "examples" in d:
            log(f"  {d['name']}: {d['examples']} examples [{d['status']}]")
        else:
            log(f"  {d['name']}: [{d['status']}]")


if __name__ == "__main__":
    main()
