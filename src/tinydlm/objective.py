"""Linear-schedule absorbing-state objective.

Confidence-ordered decoding in sample.py is a separate heuristic, not required by this loss.
"""

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


def nelbo_loss(model: nn.Module, x: Tensor, t: Tensor | None = None) -> Tensor:
    """Absorbing-state negative ELBO (Sahoo et al. 2024, Eq. 11):

        L = E_{t~U(0,1), x_t~q(.|x,t)} [(1/t) sum_{i in masked} -log p(x_i | x_t)] / T

    Each sequence draws one t; each token is independently masked with probability t.
    The linear survival schedule alpha(t) = 1 - t gives weight -alpha'(t)/(1-alpha(t)) = 1/t.
    This estimates a likelihood bound, not exact NLL or the heuristic sampler's likelihood.

    Normalize by B*T, not the random mask count. Separators are prediction targets too:
    windows may cross document boundaries, and generation must learn where stories end.
    An exactly zero t masks nothing and contributes zero; clamp the denominator to avoid 0/0.
    Keep nonzero draws unchanged and clip gradients in train.py.
    """
    B, T = x.shape
    t = t if t is not None else torch.rand(B, 1, device=x.device)
    assert t.shape == (B, 1), t.shape
    mask = torch.rand(B, T, device=x.device) < t  # (B, T): Bernoulli(t), drawn per token
    # The absorbing-state id moves with this model's vocabulary; tokenizer.py defines its slot.
    xt = torch.where(mask, model.cfg.mask_id, x)  # (B, T): absorbing and irreversible

    nll = F.cross_entropy(
        rearrange(model(xt), "B T V -> (B T) V"),
        rearrange(x, "B T -> (B T)"),
        reduction="none",
    )
    nll = rearrange(nll, "(B T) -> B T", B=B)  # every position scored, masked or not
    return (nll * mask / t.clamp_min(torch.finfo(t.dtype).tiny)).sum() / (B * T)
