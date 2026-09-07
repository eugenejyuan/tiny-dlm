"""Confidence-ordered unmasking of a fixed-length canvas.

Only masked holes can change: prefix, suffix, and earlier reveals remain fixed.
The loop fills every hole; story() separately truncates display at the next separator.
"""

import argparse
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch import Tensor

from tinydlm.config import Config, default_device
from tinydlm.model import TinyDLM
from tinydlm.train import load

GUMBEL_MIN = 1e-20  # the floor under u in -log(-log(u)): see propose


def blank(prefix: list[int], suffix: list[int], length: int, samples: int, cfg: Config) -> Tensor:
    """Place masked holes between the given prefix and suffix.
    The CLI seeds a leading separator, but training still corrupts separators as prediction targets.
    """
    holes = length - len(prefix) - len(suffix)
    assert holes > 0 and length <= cfg.seq_len and samples > 0, (holes, length, samples)
    row = torch.tensor(prefix + [cfg.mask_id] * holes + suffix)  # (T,)
    return repeat(row, "T -> B T", B=samples)


def reveal_counts(holes: int, steps: int) -> list[int]:
    """Distribute reveals across a rounded ramp; counts sum to the initial hole count.
    Skip empty steps. One pass proposes every hole independently given the initial canvas;
    more passes condition later proposals on earlier reveals.
    """
    assert steps >= 1, f"a canvas needs at least one denoising pass, got {steps}"
    edges = [round(step * holes / steps) for step in range(steps + 1)]
    return [count for count in (edges[i + 1] - edges[i] for i in range(steps)) if count > 0]


def propose(model: TinyDLM, canvas: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    """Propose one token per position and its confidence.

    Gumbel-max samples softmax(logits / temperature); temperature zero selects argmax.
    Rank positions by the candidate's log-probability under the untempered distribution.
    The denoiser excludes [MASK]; reserved IDs remain possible until learned otherwise.
    """
    assert 0 <= temperature < float("inf"), f"invalid temperature: {temperature}"
    logits = model(canvas)  # (B, T, V)

    gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(min=GUMBEL_MIN)))
    token = (logits + temperature * gumbel).argmax(dim=-1)  # (B, T); temperature 0 is argmax

    candidate = rearrange(token, "B T -> B T 1")
    logprob = logits.log_softmax(dim=-1).gather(-1, candidate)  # (B, T, 1)
    return token, rearrange(logprob, "B T 1 -> B T")


@torch.no_grad()
def generate(model: TinyDLM, canvas: Tensor, steps: int, temperature: float) -> Tensor:
    """Reveal the most confident holes, then condition the next pass on the updated canvas."""
    mask_id = model.cfg.mask_id  # the checkpoint's own layout, not a module constant
    holes = (canvas == mask_id).sum(dim=-1)  # (B,)
    assert holes.min() == holes.max(), f"the batch shares one canvas shape, got {holes.tolist()}"

    for reveal in reveal_counts(int(holes[0]), steps):
        hole = canvas == mask_id  # (B, T): the only positions this step is permitted to write
        # If K exceeds the hole count, topk can select a frozen position despite its -inf
        # confidence. Bound K before selection so the conditioning remains unchanged.
        assert reveal <= int(hole.sum(dim=-1).min()), f"revealing {reveal} of too few holes"

        token, confidence = propose(model, canvas, temperature)
        chosen = confidence.masked_fill(~hole, -torch.inf).topk(reveal, dim=-1).indices  # (B, K)
        canvas = canvas.scatter(1, chosen, token.gather(1, chosen))

    assert not (canvas == mask_id).any(), "the schedule must consume every hole"
    return canvas


def story(ids: list[int], sep_id: int) -> list[int]:
    """Strip the seeded separator and stop at the next document boundary.
    No subsequent separator means the story did not finish inside the fixed-length canvas.
    """
    body = ids[1:] if ids[:1] == [sep_id] else ids
    return body[: body.index(sep_id)] if sep_id in body else body


def main() -> None:
    cwd = Path.cwd()
    parser = argparse.ArgumentParser(description="Sample stories from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=cwd / "checkpoint.pt", help="weights")
    parser.add_argument("--prompt", default="", help="frozen prefix; empty means unconditional")
    parser.add_argument("--suffix", default="", help="frozen suffix, for infilling both ends")
    parser.add_argument("--length", type=int, default=None, help="canvas size; seq_len by default")
    parser.add_argument("--steps", type=int, default=None, help="denoising passes; one per hole")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 is greedy")
    parser.add_argument("--samples", type=int, default=2, help="canvases denoised side by side")
    parser.add_argument("--seed", type=int, default=0, help="sets the sampling random draws")
    parser.add_argument("--device", default=default_device(), help="cuda when one is present")
    args = parser.parse_args()

    model, cfg, step, tokenizer = load(args.checkpoint, args.device)
    torch.manual_seed(args.seed)

    length = cfg.seq_len if args.length is None else args.length
    steps = length if args.steps is None else args.steps
    # The separator leads every prompt because prepared streams put one before each document.
    prefix = [cfg.sep_id, *tokenizer.encode(args.prompt)]
    canvas = blank(prefix, tokenizer.encode(args.suffix), length, args.samples, cfg)
    canvas = canvas.to(args.device)
    filled = generate(model, canvas, steps, args.temperature)

    print(f"depth {cfg.depth} @ step {step}, T={length}, temperature={args.temperature}")
    for row in filled.tolist():
        print("-" * 78)
        print(tokenizer.decode(story(row, cfg.sep_id)))


if __name__ == "__main__":
    main()
