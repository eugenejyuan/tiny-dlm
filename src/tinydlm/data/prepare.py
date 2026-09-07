"""Offline BPE training and uint16 bin preparation.

A prepared bundle is `tokenizer.json`, one `<split>.bin` per split, and `prepare.json`
recording what produced them. Both splits are encoded with the vocabulary learned from the
training text, so a bundle's bins and tokenizer only mean anything together.

uv run python -m tinydlm.data.prepare --train raw/train.txt --val raw/val.txt --output-dir prepared/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import regex
from tqdm import tqdm

from tinydlm.tokenizer import SPECIAL_TOKENS, TOKENIZER_FILE, Tokenizer, train_bpe_file

PREPARE_FILE = "prepare.json"
BYTES_PER_MB = 1_000_000
SEPARATOR = "<|endoftext|>"


def write_bin(source: Path, n_bytes: int, separator: str, codec: Tokenizer, path: Path) -> dict:
    """Encode the first `n_bytes` of `source` into one flat little-endian uint16 stream.
    Every nonempty document gets one leading separator token: in the bin the separator is the
    only document boundary left, which is what lets the loader window at any offset.
    Returns this split's bundle metadata.
    """
    assert codec.vocab_size <= np.iinfo(np.uint16).max + 1, "vocabulary outgrew uint16"
    with source.open("rb") as handle:
        text = handle.read(n_bytes).decode("utf-8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("wb") as sink:
        documents = regex.splititer(regex.escape(separator), text)
        for document in tqdm(documents, desc=f"encoding {path.name}"):
            if document.strip():
                ids = np.array([codec.sep_id, *codec.encode(document)], dtype="<u2")
                sink.write(ids.tobytes())
                total += len(ids)
    assert total, f"{source} held no document in its first {n_bytes} bytes"
    assert path.stat().st_size == 2 * total, f"short write: {path} is not {total} uint16 tokens"
    print(f"{path.stem}: {total / 1e6:.1f}M tokens -> {path}  (source: {source})")
    return {"source": str(source), "source_bytes": n_bytes, "tokens": total}


def prepare(
    train: Path,
    val: Path,
    output_dir: Path,
    separator: str = SEPARATOR,
    vocab_size: int = 8192,
    max_mb: int | None = None,
    workers: int = os.cpu_count() or 1,
) -> Path:
    """Learn BPE from the training text, then encode both splits with that one vocabulary."""
    assert max_mb is None or max_mb > 0, f"max_mb must be positive, got {max_mb}"
    train, val = train.expanduser().resolve(), val.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    cap = max_mb * BYTES_PER_MB if max_mb else sys.maxsize
    train_bytes = min(train.stat().st_size, cap)
    val_bytes = min(val.stat().st_size, cap)

    split_on = (separator, *SPECIAL_TOKENS)
    codec = Tokenizer(train_bpe_file(train, vocab_size, split_on, workers, train_bytes))
    tokenizer_path = output / TOKENIZER_FILE
    codec.save(tokenizer_path)
    print(f"{codec.vocab_size} tokens -> {tokenizer_path}")

    bins = {
        "train": write_bin(train, train_bytes, separator, codec, output / "train.bin"),
        "val": write_bin(val, val_bytes, separator, codec, output / "val.bin"),
    }
    metadata = {"separator": separator, "max_mb": max_mb, "bins": bins}
    (output / PREPARE_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one local corpus into token bins.")
    parser.add_argument("--train", type=Path, required=True, help="training text")
    parser.add_argument("--val", type=Path, required=True, help="validation text")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="prepared bundle")
    parser.add_argument("--separator", default=SEPARATOR, help="document boundary in the text")
    parser.add_argument("--vocab-size", type=int, default=8192, help="bytes, merges, and specials")
    parser.add_argument(
        "--max-mb", type=int, help="prepare only the first MB of each source (default: all of it)"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1, help="BPE counting processes"
    )
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
