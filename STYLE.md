# Style

Use clear ASCII English for source and comments; data fixtures may contain Unicode.
Experiment notes may use English or Chinese. Scope and file placement live in
[the mission](specs/mission.md).

## Tensor notation

```text
B    batch                  T    query position       S    key position
V    vocabulary             D    model width          H    heads
Dh   head dimension         Dff  MLP hidden width     Dh2  half a head dimension
K    positions revealed per denoising step
```

In the current model, S = T and D = H * Dh. Define any additional axes where introduced.
Tensor names describe meaning. Annotate shapes at rearrangements, broadcasts, or other
non-obvious transitions; skip routine shape-preserving operations. Prefer einops:

```python
score = einsum(q, k, "B T H Dh, B S H Dh -> B H T S")
```

Use a concise trailing comment when no pattern string carries the shape:

```python
t = torch.rand(B, 1, device=x.device)  # (B, 1): one noise level per sequence
```

Native operations are fine when they make the code clearer or serve a measured optimization.

## Comments and checks

Explain equations, assumptions, and failure modes; do not narrate obvious code or repeat
project policy in modules. Keep each explanation in one place and avoid speculative extensions.

After setup, inspect style with `uv run ruff check src` and `uv run ruff format --check src`.
These do not check mathematical correctness.
