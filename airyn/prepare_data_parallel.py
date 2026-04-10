"""
Parallel data preparation — uses all available CPU cores.
Each dataset gets its own process with its own tokenizer pool.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing import Pool, Process, cpu_count
from pathlib import Path

import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
SHARD_SIZE = 100_000_000


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


def run_pretraining(name, hf_id, subsets, output_dir, target_tokens, pool_workers):
    """Run in its own process — downloads + tokenizes one pretraining dataset."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_shards, existing_tokens = count_existing_tokens(out)
    if existing_tokens >= target_tokens:
        log(name, f"SKIP: already complete ({existing_shards} shards, {existing_tokens / 1e9:.2f}B tok)")
        return

    if existing_shards > 0:
        log(name, f"RESUMING: {existing_shards} shards ({existing_tokens / 1e9:.2f}B tok), "
            f"need {(target_tokens - existing_tokens) / 1e9:.1f}B more")

    log(name, f"Starting: {hf_id} with {pool_workers} tokenizer workers (target: {target_tokens / 1e9:.0f}B)")

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = existing_tokens
    train_shard_count = len(list(out.glob(f"{name}_train_*.bin")))
    val_shard_count = len(list(out.glob(f"{name}_val_*.bin")))
    tokens_to_skip = existing_tokens  # skip this many tokens from the stream
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
                    rate = total_tokens / elapsed / 1e6
                    eta = (target_tokens - total_tokens) / (rate * 1e6) / 60 if rate > 0 else 0
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
        subsets_to_load = subsets if subsets else [None]
        for subset in subsets_to_load:
            if total_tokens >= target_tokens:
                break
            if subset:
                log(name, f"Loading subset: {subset}")
            for attempt in range(3):
                try:
                    kwargs = {"path": hf_id, "split": "train", "streaming": True}
                    if subset is not None:
                        kwargs["name"] = subset
                    ds = load_dataset(**kwargs)
                    break
                except Exception as e:
                    if attempt < 2:
                        log(name, f"Retry {attempt + 1}/3: {e}")
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise

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
        log(name, f"DONE: {total_tokens / 1e9:.2f}B tok, {train_shard_count} train shards, {elapsed:.0f}s")

    except Exception as e:
        log(name, f"ERROR: {e}\n{traceback.format_exc()}")
        if buf_pos > 0:
            flush_shard()


def run_sft(name, hf_id, output_path, config=None):
    """Run in its own process — downloads one SFT dataset as JSONL."""
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
            kwargs = {"path": hf_id, "split": "train", "streaming": True}
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
    log("main", f"Total CPU cores: {total_cores}")

    # Ensure deps
    try:
        import zstandard  # noqa: F401
    except ImportError:
        os.system(f"{sys.executable} -m pip install zstandard --break-system-packages -q")

    # Allocate cores: 3 pretraining datasets share the pool budget
    # Reserve some for SFT downloaders and OS
    pretrain_budget = max(total_cores - 16, 64)
    # smollm gets the most (it's the largest), dolmino and open-web-math split the rest
    smollm_workers = pretrain_budget // 2
    dolmino_workers = pretrain_budget // 4
    owm_workers = pretrain_budget // 4

    log("main", f"Worker allocation: smollm={smollm_workers}, dolmino={dolmino_workers}, owm={owm_workers}")

    # Launch ALL datasets as separate processes
    processes = []

    # Pretraining
    p1 = Process(target=run_pretraining, args=(
        "smollm_corpus", "HuggingFaceTB/smollm-corpus",
        ["cosmopedia-v2", "python-edu", "fineweb-edu-dedup"],
        "data/smollm_corpus", 100_000_000_000, smollm_workers,
    ))
    processes.append(("smollm_corpus", p1))

    p2 = Process(target=run_pretraining, args=(
        "dolmino_mix", "allenai/dolmino-mix-1124",
        None,
        "data/dolmino_mix", 100_000_000_000, dolmino_workers,
    ))
    processes.append(("dolmino_mix", p2))

    p3 = Process(target=run_pretraining, args=(
        "open_web_math", "open-web-math/open-web-math",
        None,
        "data/open_web_math", 15_000_000_000, owm_workers,
    ))
    processes.append(("open_web_math", p3))

    # SFT (lightweight, IO-bound — just separate processes)
    for config in ["reasoning-deepseek", "reasoning-flash"]:
        safe = config.replace("-", "_")
        p = Process(target=run_sft, args=(
            f"dolphin_{safe}", "cognitivecomputations/dolphin-r1",
            f"data/sft/dolphin_r1_{safe}.jsonl", config,
        ))
        processes.append((f"dolphin_{safe}", p))

    # Start all
    log("main", f"Launching {len(processes)} parallel processes...")
    for name, p in processes:
        p.start()
        log("main", f"  Started: {name} (pid={p.pid})")

    # Wait for all
    for name, p in processes:
        p.join()
        log("main", f"  Finished: {name} (exit={p.exitcode})")

    log("main", "All done!")


if __name__ == "__main__":
    main()
