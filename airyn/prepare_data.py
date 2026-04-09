"""
Download and tokenize FineWeb-Edu (10B tokens sample) into .bin shards
compatible with the modded-nanogpt data format.

Output: data/fineweb10B/fineweb_train_*.bin, data/fineweb10B/fineweb_val_*.bin

Shard format:
  - 256 int32 header: [magic=20240520, version=1, num_tokens, 0, 0, ...]
  - uint16 token array

Usage:
  python airyn/prepare_data.py                     # default: 10B tokens, 100M per shard
  python airyn/prepare_data.py --workers 32         # control parallelism
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


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


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu data for Airyn training")
    parser.add_argument("--output-dir", type=str, default="data/fineweb10B")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per shard")
    parser.add_argument("--val-shards", type=int, default=1, help="Number of val shards")
    parser.add_argument("--total-tokens", type=int, default=10_000_000_000, help="Total tokens to download")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="HF dataset name")
    parser.add_argument("--dataset-name", type=str, default="sample-10BT", help="Dataset config/name")
    parser.add_argument("--workers", type=int, default=0, help="Num tokenizer workers (0=auto)")
    parser.add_argument("--batch-docs", type=int, default=1000, help="Docs per tokenizer batch")
    args = parser.parse_args()

    try:
        import tiktoken  # noqa: F401
    except ImportError:
        os.system(f"{sys.executable} -m pip install tiktoken")

    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    num_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset} ({args.dataset_name})...")
    dataset = load_dataset(args.dataset, name=args.dataset_name, split="train", streaming=True)

    # Collect document batches from the stream, feed to worker pool
    token_buf = np.empty(args.shard_size, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    val_shard_count = 0
    train_shard_count = 0
    t0 = time.time()

    def flush_shard():
        nonlocal buf_pos, total_tokens, val_shard_count, train_shard_count
        if val_shard_count < args.val_shards:
            path = output_dir / f"fineweb_val_{val_shard_count:06d}.bin"
            val_shard_count += 1
        else:
            path = output_dir / f"fineweb_train_{train_shard_count:06d}.bin"
            train_shard_count += 1
        write_shard(path, token_buf[:buf_pos])
        total_tokens += buf_pos
        elapsed = time.time() - t0
        rate = total_tokens / elapsed / 1e6
        print(f"  {path.name}: {total_tokens / 1e9:.2f}B tokens, "
              f"{rate:.1f}M tok/s, {elapsed:.0f}s elapsed", flush=True)
        buf_pos = 0

    def ingest(chunk: np.ndarray):
        nonlocal buf_pos
        pos = 0
        while pos < len(chunk):
            space = args.shard_size - buf_pos
            n = min(space, len(chunk) - pos)
            token_buf[buf_pos:buf_pos + n] = chunk[pos:pos + n]
            buf_pos += n
            pos += n
            if buf_pos >= args.shard_size:
                flush_shard()
                if total_tokens >= args.total_tokens:
                    return

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

    print(f"Tokenizing with {num_workers} workers, {args.batch_docs} docs/batch...", flush=True)

    with Pool(num_workers) as pool:
        for token_chunk in pool.imap(tokenize_batch, doc_batches(dataset, args.batch_docs), chunksize=4):
            ingest(token_chunk)
            if total_tokens >= args.total_tokens:
                break

    # Write remaining
    if buf_pos > 0:
        flush_shard()

    elapsed = time.time() - t0
    print(f"\nDone! {total_tokens / 1e9:.2f}B tokens in {elapsed:.0f}s "
          f"({total_tokens / elapsed / 1e6:.1f}M tok/s)")
    print(f"  Val shards: {val_shard_count}")
    print(f"  Train shards: {train_shard_count}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
