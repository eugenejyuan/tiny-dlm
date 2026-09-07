"""Read prepared bins: one flat little-endian uint16 token stream per split."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tinydlm.config import Config


def memmap(prepared_dir: Path, split: str) -> np.memmap:
    """A flat stream makes a random offset the entire loader and shuffle policy."""
    path = prepared_dir / f"{split}.bin"
    assert path.exists(), f"{path} is missing -- run `python -m tinydlm.data.prepare` first"
    return np.memmap(path, dtype="<u2", mode="r")


def sample_batch(tokens: np.memmap, cfg: Config, device: str, generator=None) -> Tensor:
    """Random windows may cross boundaries because generation sees separators mid-canvas."""
    assert len(tokens) >= cfg.seq_len, f"need at least {cfg.seq_len} prepared tokens"
    offsets = torch.randint(len(tokens) - cfg.seq_len + 1, (cfg.batch_size,), generator=generator)
    window = np.stack([tokens[i : i + cfg.seq_len] for i in offsets.tolist()])  # (B, T) uint16
    return torch.from_numpy(window.astype(np.int64)).to(device)  # (B, T): ids
