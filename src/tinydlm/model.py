"""Axis legend, the same letters as everywhere else in this package:

    B    batch                 T    query position    S    key position (always == T)
    V    vocabulary            D    model width       H    heads
    Dh   head dim, D = H * Dh  Dff  MLP hidden width  Dh2  half a head dim, Dh / 2

The architecture:

    no causal mask     masks appear anywhere in the sequence, so every position has to 
                       attend to every position
    no timestep input  the mask pattern estimates t; explicit time conditioning had minimal 
                       effect in its OpenWebText ablation (Sahoo et al. 2024 Appendix E.5)
    RMSNorm, no gain   one scale-free normalization with no extra learned vector
    RoPE               position as a rotation, so attention reads the distance t - s
    QK-norm            bounds attention scores before the softmax
    ReLU^2 MLP         a two-matrix gated-power activation (So et al. 2021)
    tied embedding     input and output index the same V x D space, and untying it would
                       add a fifth of the parameters at depth 6

"""

import math

import torch
from torch import Tensor, nn
from einops import einsum, rearrange

from tinydlm.config import Config

NORM_EPS = 1e-6  # the floor under RMSNorm, so an all-zero row cannot divide by zero
ROPE_BASE = 10_000.0
MLP_RATIO = 4  # Dff = 4 * D, the project's fixed expansion ratio


def rms_norm(x: Tensor) -> Tensor:
    """x / rms(x), the norm of Zhang & Sennrich (2019), with no learnable gain: a learned
    projection always follows and can absorb any per-channel scale. Always over the last
    axis -- D for a residual stream, Dh for QK-norm -- which is why there is no argument.
    """
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + NORM_EPS)


class RoPE(nn.Module):
    """Rotary position embedding (Su et al. 2021): each adjacent pair of head dimensions is
    turned by an angle proportional to the position, so the dot product of two rotated
    vectors depends on the offset t - s and not on where the pair sits in the sequence.
    """

    def __init__(self, head_dim: int, seq_len: int) -> None:
        super().__init__()
        freq = ROPE_BASE ** (-torch.arange(0, head_dim, 2) / head_dim)  # (Dh2,): one per pair
        angle = einsum(torch.arange(seq_len).float(), freq, "T, Dh2 -> T Dh2")
        self.register_buffer("cos", angle.cos(), persistent=False)  # (T, Dh2)
        self.register_buffer("sin", angle.sin(), persistent=False)  # (T, Dh2)

    def forward(self, x: Tensor) -> Tensor:
        """One planar rotation per pair: (even, odd) turned by the angle for that position."""
        assert x.shape[1] <= self.cos.shape[0], "sequence exceeds the RoPE table"
        even, odd = rearrange(x, "B T H (Dh2 two) -> two B T H Dh2", two=2)
        cos = rearrange(self.cos[: x.shape[1]], "T Dh2 -> 1 T 1 Dh2")  # broadcast over B and H
        sin = rearrange(self.sin[: x.shape[1]], "T Dh2 -> 1 T 1 Dh2")
        rotated = [even * cos - odd * sin, even * sin + odd * cos]
        return rearrange(rotated, "two B T H Dh2 -> B T H (Dh2 two)")


class Attention(nn.Module):
    """Bidirectional attention. The entire difference from GPT is the missing causal mask.

    The score matrix is materialized instead of disappearing into F.scaled_dot_product_attention, 
    so the reader can see that every position attends to every position. 
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads = cfg.H
        self.q_proj = nn.Linear(cfg.D, cfg.H * cfg.Dh)
        self.k_proj = nn.Linear(cfg.D, cfg.H * cfg.Dh)
        self.v_proj = nn.Linear(cfg.D, cfg.H * cfg.Dh)
        self.out_proj = nn.Linear(cfg.H * cfg.Dh, cfg.D)

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        q = rearrange(self.q_proj(x), "B T (H Dh) -> B T H Dh", H=self.heads)
        k = rearrange(self.k_proj(x), "B S (H Dh) -> B S H Dh", H=self.heads)
        v = rearrange(self.v_proj(x), "B S (H Dh) -> B S H Dh", H=self.heads)
        q, k = rope(rms_norm(q)), rope(rms_norm(k))  # QK-norm then RoPE, both shape-preserving

        score = einsum(q, k, "B T H Dh, B S H Dh -> B H T S") / math.sqrt(q.shape[-1])
        attn = score.softmax(dim=-1)  # no mask: this is the whole architectural claim
        o = einsum(attn, v, "B H T S, B S H Dh -> B T H Dh")
        return self.out_proj(rearrange(o, "B T H Dh -> B T (H Dh)"))


class MLP(nn.Module):
    """Squared ReLU (So et al. 2021), the position-wise half of the block."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.in_proj = nn.Linear(cfg.D, MLP_RATIO * cfg.D)
        self.out_proj = nn.Linear(MLP_RATIO * cfg.D, cfg.D)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.in_proj(x).relu().square())


class Block(nn.Module):
    """Pre-norm residual (Xiong et al. 2020): each sublayer reads a normalized copy and
    writes back an increment, so the residual stream runs unbroken from embedding to
    readout.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        x = x + self.attn(rms_norm(x), rope)
        return x + self.mlp(rms_norm(x))


class TinyDLM(nn.Module):
    """The denoiser: x_t in, a distribution at every position out.

    It predicts all T positions, including the ones that were never masked -- objective.py
    simply does not score those. There is no separate "which positions are masked" input,
    because [MASK] is a token id like any other and the corruption is already in x. Its output
    logit is always -inf: [MASK] is an input state, never a clean-token prediction.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.D)
        self.rope = RoPE(cfg.Dh, cfg.seq_len)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.lm_head = nn.Linear(cfg.D, cfg.vocab_size)
        self.lm_head.weight = self.embed.weight

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        h = self.embed(x)  # (B, T, D)
        for block in self.blocks:
            h = block(h, self.rope)
        logits = self.lm_head(rms_norm(h))  # its weight is tied to the input embedding table
        logits[..., self.cfg.mask_id] = -torch.inf  # MDLM's zero-masking-probability constraint
        assert logits.shape == (B, T, self.cfg.vocab_size), logits.shape
        return logits
