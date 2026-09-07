# Mission

Learn diffusion models by deriving, implementing, and testing them. The current masked DLM is
the first implementation; the project can grow with new research and concrete learning questions.

## Principles

- Keep each method understandable from input to loss to sampling. No repository-wide line cap.
- Before adding code, check whether existing code can be reused or simplified. Add only what the
  current task needs; avoid speculative abstractions, dependencies, compatibility, and structure.
- Handwrite the algorithm being studied; use focused libraries for ordinary plumbing.
- Keep useful math beside the code. State assumptions and distinguish measured results from plans.
- Add checks for meaningful failure modes. A shorter implementation is not better if it hides
  errors or makes the algorithm harder to follow.

## Placement

- `src/tinydlm/`: maintained code. Keep today's flat model/objective/train/sample layout.
- `specs/`: purpose and near-term plan; code and README own implementation details and usage.
- `lab/<question>/`: create only when an experiment exists. Keep its code and notes together.
- `tests/`: create with the first tests; use small local fixtures.
- `data/` and `runs/`: ignored datasets and generated artifacts.

Split methods or extract shared utilities when actual implementations need it. New folders and
per-folder README files are not required. Record an experiment's question, source, command,
revision, data/configuration, seed, and conclusion in one note; keep failed results too.
