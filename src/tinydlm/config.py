"""Keep one simple model-scale knob: `depth`.

Width and head count are derived from it while the head dimension stays fixed:

    D  = 64 * depth     model width
    H  = depth          heads
    Dh = 64             head dim, so that D == H * Dh holds by construction

Fixing `Dh` at 64 while width and head count grow together follows the nanoGPT/GPT-3 family.
"""

from dataclasses import dataclass, field

import torch

from tinydlm.tokenizer import MASK_TOKEN, SEP_TOKEN, special_id

HEAD_DIM = 64


@dataclass
class Config:
    """Model and optimizer configuration.
    """

    depth: int = 6

    vocab_size: int = 8192  # the prepared tokenizer is the artifact-level source of truth
    seq_len: int = 256  # T, and therefore also S
    batch_size: int = 32
    lr: float = 3e-4
    steps: int = 5000
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    D: int = field(init=False)
    H: int = field(init=False)
    Dh: int = field(init=False)
    sep_id: int = field(init=False)
    mask_id: int = field(init=False)

    def __post_init__(self) -> None:
        assert min(self.depth, self.seq_len, self.batch_size, self.steps) > 0, vars(self)
        assert 0 < self.lr < float("inf"), f"invalid lr: {self.lr}"
        self.D, self.H, self.Dh = HEAD_DIM * self.depth, self.depth, HEAD_DIM
        # The specials move with vocab_size (tokenizer.py) and cannot be constants. 
        self.sep_id = special_id(SEP_TOKEN, self.vocab_size)
        self.mask_id = special_id(MASK_TOKEN, self.vocab_size)
