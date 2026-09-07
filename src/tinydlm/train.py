"""Single-device training: random token windows, masked loss, and periodic checkpoints.

The 1/t loss estimator can have high variance: clip gradients to limit rare large updates.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from itertools import count
from pathlib import Path

import torch
from torch import Tensor

from tinydlm.config import Config, default_device
from tinydlm.data.dataset import memmap, sample_batch
from tinydlm.data.prepare import PREPARE_FILE
from tinydlm.model import TinyDLM
from tinydlm.objective import nelbo_loss
from tinydlm.tokenizer import TOKENIZER_FILE, Tokenizer

BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0  # see the module docstring: the 1/t tail, not superstition
WARMUP_FRACTION = 0.02
FINAL_LR_FRACTION = 0.1  # cosine floor; the final update need not reach it
LOG_EVERY = 50
CHECKPOINT_FILE = "checkpoint.pt"  # the final one; intermediates are step-NNNNNN.pt
EVAL_BATCHES = 16
EVAL_SEED = 10_000


def learning_rate(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay toward FINAL_LR_FRACTION of the peak.
    Step indices stop before the cosine endpoint; a two-step run has no decay.
    """
    warmup = max(1, round(WARMUP_FRACTION * cfg.steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine)


def train_step(model: TinyDLM, optimizer: torch.optim.Optimizer, x: Tensor) -> float:
    # bf16 autocast on CUDA and fp32 on CPU; stored weights stay fp32. bf16's exponent range
    # reduces overflow risk relative to fp16 but does not guarantee finiteness.
    device_type = x.device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        loss = nelbo_loss(model, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


@torch.no_grad()
def evaluate(model: TinyDLM, tokens, cfg: Config) -> float:
    """Held-out NELBO with randomized noise strata to reduce integration variance.
    Evaluation reseeds the process RNG. The current model has no dropout or batch normalization.
    """
    torch.manual_seed(EVAL_SEED)
    device = next(model.parameters()).device
    data_rng = torch.Generator().manual_seed(EVAL_SEED)
    draws = []
    for index in range(EVAL_BATCHES):
        t = (index + torch.rand(cfg.batch_size, 1, device=device)) / EVAL_BATCHES
        draws.append(nelbo_loss(model, sample_batch(tokens, cfg, device, data_rng), t).item())
    return sum(draws) / len(draws)


def run_dir(runs_dir: Path, name: str) -> Path:
    """Claim the first free runs_dir/name, name-2, name-3, ...
    """
    runs_dir = runs_dir.expanduser().resolve()
    for suffix in count(1):
        path = runs_dir / (name if suffix == 1 else f"{name}-{suffix}")
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def save(
    model: TinyDLM, cfg: Config, step: int, tokenizer: Tokenizer, path: Path, provenance: dict
) -> Path:
    """Save weights, tokenizer, and run metadata for inference, not exact training resume."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "config": asdict(cfg),
            "step": step,
            "provenance": provenance,
            "model": model.state_dict(),
            "merges": tokenizer.merges,
        },
        path,
    )
    return path


def load(path: Path, device: str) -> tuple[TinyDLM, Config, int, Tokenizer]:
    """Reload the checkpoint onto the given device; the config says nothing about where it ran."""
    blob = torch.load(path.expanduser().resolve(), map_location=device, weights_only=True)
    assert blob["schema"] == 1, f"unsupported checkpoint schema: {blob['schema']}"
    cfg = Config(**blob["config"])
    model = TinyDLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    return model, cfg, blob["step"], Tokenizer(blob["merges"])


def train(
    cfg: Config,
    tokenizer: Tokenizer,
    device: str,
    seed: int,
    prepared_dir: Path,
    out_dir: Path,
    save_every: int = 0,
    save_after: int = 0,
) -> Path:
    """One seed controls initialization, window offsets, and corruption in this process.

    Intermediate checkpoints land every save_every steps once save_after is reached; the final
    one is always written, and only it carries the held-out NELBO.
    """
    torch.manual_seed(seed)
    streams = {split: memmap(prepared_dir, split) for split in ("train", "val")}
    model = TinyDLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {device}, "
        f"{cfg.steps} steps x {cfg.batch_size} x {cfg.seq_len} tokens, seed {seed}"
    )

    provenance = {
        "seed": seed,
        "device": device,
        "prepared": json.loads((prepared_dir / PREPARE_FILE).read_text(encoding="utf-8")),
    }
    start = time.perf_counter()
    for step in range(cfg.steps):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = train_step(model, optimizer, sample_batch(streams["train"], cfg, device))
        if step == 0 or (step + 1) % LOG_EVERY == 0 or step + 1 == cfg.steps:
            elapsed = time.perf_counter() - start
            rate = cfg.batch_size * cfg.seq_len * (step + 1) / elapsed / 1e3
            print(
                f"step {step + 1:>5}/{cfg.steps}  nelbo {loss:6.3f}  "
                f"lr {lr:.2e}  {rate:6.1f}k tok/s  {elapsed:5.1f}s"
            )
        done = step + 1
        if save_every and done >= save_after and done % save_every == 0 and done < cfg.steps:
            path = save(model, cfg, done, tokenizer, out_dir / f"step-{done:06d}.pt", 
                        provenance | {"elapsed_seconds": time.perf_counter() - start},)
            print(f"checkpoint -> {path}")

    val_nelbo = evaluate(model, streams["val"], cfg)
    provenance |= {
        "elapsed_seconds": time.perf_counter() - start,
        "val_nelbo_nats_per_token": val_nelbo,
    }
    path = save(model, cfg, cfg.steps, tokenizer, out_dir / CHECKPOINT_FILE, provenance)
    print(f"val NELBO {val_nelbo:.3f} nats/token -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the masked diffusion model on one device.")
    parser.add_argument("--prepared-dir", type=Path, default=Path.cwd(), help="tokenizer and bins")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="holds run folders")
    parser.add_argument("--name", default="run", help="run folder name; -2, -3 ... on conflict")
    parser.add_argument("--save-every", type=int, default=10_000, help="0 keeps only the final one")
    parser.add_argument("--save-after", type=int, default=0, help="first step eligible to save")
    parser.add_argument("--depth", type=int, default=Config.depth, help="the only size knob")
    parser.add_argument("--steps", type=int, default=Config.steps, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len, help="tokens per sequence")
    parser.add_argument("--lr", type=float, default=Config.lr, help="peak learning rate")
    parser.add_argument("--device", default=default_device(), help="cuda when one is present")
    parser.add_argument("--seed", type=int, default=0, help="sets all random draws")
    args = parser.parse_args()

    prepared_dir = args.prepared_dir.expanduser().resolve()
    tokenizer = Tokenizer.load(prepared_dir / TOKENIZER_FILE)
    cfg = Config(
        vocab_size=tokenizer.vocab_size,
        sep_id=tokenizer.sep_id,
        mask_id=tokenizer.mask_id,
        depth=args.depth,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
    )
    out_dir = run_dir(args.runs_dir, args.name)
    print(f"run directory {out_dir}")
    train(cfg, tokenizer, args.device, args.seed, prepared_dir, out_dir, 
          args.save_every, args.save_after)


if __name__ == "__main__":
    main()
