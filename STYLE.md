# Style

This is the quick reference for writing core code. Broader design principles live in
[the project mission](specs/mission.md).

## Tensor axes

Core modules share this vocabulary:

```text
B    batch
T    query position
S    key position, always equal to T
V    vocabulary
D    model width
H    heads
Dh   head dimension, D = H * Dh
Dff  MLP hidden width
Dh2  half a head dimension, Dh / 2
K    positions revealed in one denoising step, K <= T
```

Attention keeps `T` and `S` separate because query and key positions have different roles even
when their lengths are equal. Add any new axis here before using it in core code.

## Shape annotations

Tensor names describe meaning, not shape. Annotate a shape only where axes are introduced or
rearranged, broadcasting is easy to miss, or the shape explains the algorithm.

Use an `einops` pattern when axes define the operation:

```python
score = einsum(q, k, "B T H Dh, B S H Dh -> B H T S")
```

Use a concise trailing comment when no pattern string carries the shape:

```python
t = torch.rand(B, 1, device=x.device)  # (B, 1): one noise level per sequence
```

Do not annotate routine shape-preserving operations or obvious plumbing.

## Repository rules

- All tensor rearrangement in `src/tinydlm/` goes through `einops.rearrange` or
  `einops.einsum`; do not use `.view`, `.transpose`, or `.permute`.
- Keep feature flags out of the core path. Experimental variants belong in `lab/`.
- Comments and docstrings explain rationale or math, not a second version of the code.
- Everything under `src/`, `lab/`, and `tests/` uses ASCII English.

Run `just lint` to check the enforceable rules.
