"""The current model scales with depth: D = 64 * depth, H = depth, and Dh = 64."""

from dataclasses import dataclass

import torch

HEAD_DIM = 64


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Model and optimizer configuration for the current masked DLM reference.

    The tokenizer owns the vocabulary and its special-token IDs; they are copied in here so
    the model and objective can read them without it, and a checkpoint stores both together.
    """

    vocab_size: int
    sep_id: int
    mask_id: int
    depth: int = 6
    seq_len: int = 256  # T, and therefore also S
    batch_size: int = 32
    lr: float = 3e-4
    steps: int = 5000

    def __post_init__(self) -> None:
        assert min(self.depth, self.seq_len, self.batch_size, self.steps) > 0, vars(self)
        assert 0 < self.lr < float("inf"), f"invalid lr: {self.lr}"
        assert max(self.sep_id, self.mask_id) < self.vocab_size, vars(self)

    @property
    def D(self) -> int:
        return HEAD_DIM * self.depth

    @property
    def H(self) -> int:
        return self.depth

    @property
    def Dh(self) -> int:
        return HEAD_DIM
