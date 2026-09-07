"""Fetch the TinyStories splits for offline preparation.

Documents are separated by <|endoftext|>, which is prepare.py's default separator.
"""

import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm

SPLITS = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{file}"
CHUNK = 1 << 20


def download(split: str, raw_dir: Path) -> Path:
    """Reuse existing files; new downloads use a temporary path until Content-Length matches.
    Interrupted downloads restart. Existing files are trusted without upstream checksum validation.
    """
    path = raw_dir / SPLITS[split]
    if path.exists():
        return path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(URL.format(file=path.name)) as response:
        size = int(response.headers["Content-Length"])
        with (
            partial.open("wb") as handle,
            tqdm(desc=path.name, total=size, unit="B", unit_scale=True) as bar,
        ):
            while chunk := response.read(CHUNK):
                bar.update(handle.write(chunk))
    written = partial.stat().st_size
    assert written == size, f"{partial} stopped at {written} of {size} bytes -- rerun to refetch"
    partial.replace(path)
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the TinyStories splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path.cwd(), help="download directory")
    args = parser.parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    for split in SPLITS:
        print(f"{split}: {download(split, raw_dir)}")


if __name__ == "__main__":
    main()
