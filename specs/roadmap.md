# Tech Stack

> tinyDLM keeps the diffusion core on PyTorch and einops, while using small, conventional
> tools for data preparation and packaging.

The stack follows one boundary: code that explains masked diffusion stays explicit; ordinary
plumbing uses focused libraries. The goal is one clear path through the project, not a
configurable framework.

## Core

The algorithmic code in the model, objective, sampler, and training loop is expressed through
only two third-party libraries:

| Tool | Role |
|---|---|
| **PyTorch** | Tensors, automatic differentiation, mixed precision, and optimization |
| **einops** | Named-axis projections and rearrangements at important data-flow transitions |

The core does not use `transformers`, `diffusers`, `accelerate`, Lightning, or a Trainer
abstraction. These libraries are useful for larger systems, but here they would hide the code the
reader came to understand.

## Data and project tooling

| Tool | Role |
|---|---|
| `regex` | Unicode-aware pretokenization |
| NumPy | `uint16` token streams and memory mapping |
| `tqdm` | Progress for downloading, BPE training, and encoding |
| Python standard library | TinyStories download and local corpus manifests |
| `uv` | Python installation, environment management, and lockfile |
| Hatchling | Build backend for the Python package |
| Ruff | Formatting and linting |
| pytest | Tests and executable invariants |
| `just` | Optional shortcuts for common commands |

Byte-level BPE remains handwritten because it is useful to see. Dataset download machinery does
not: fetching only writes local files and a `corpus.json`, after which preparation is fully
offline and corpus-independent.

Every `just` recipe wraps a direct `uv run python -m ...` command. `just` is convenient, but it is
never required to use the repository.

## Package boundary

The public project name is **tinyDLM**, the Python distribution is `tiny-dlm`, and the import
package is `tinydlm`.

```text
tiny-dlm/
├── src/tinydlm/   # the only installable package
│   ├── data/      # corpus manifests and preparation
│   ├── config.py, tokenizer.py
│   └── model.py, objective.py, train.py, sample.py
├── lab/           # baselines, sweeps, and experiments
├── tests/         # executable contracts
└── specs/         # project decisions
```

The `src/` layout makes the shipping boundary explicit and ensures tests resolve the installed
package. `lab/` stays repository-local so experiments can grow without enlarging the core or
becoming part of the public Python API.

## Runtime

- Python 3.12
- Single-device training
- `bfloat16` autocast on CUDA with FP32 master weights
- FP32 on CPU, including the smoke test
- Process-level parallelism for future sweeps: one complete run per GPU under `lab/`

Run artifacts are stored by corpus under `data/<corpus>/`; checkpoints keep the configuration
that created their weights. Both locations stay outside version control.

## Deliberate limits

The core has no feature flags, distributed-training branches, or production orchestration. DDP,
FSDP, `torch.compile`, alternative samplers, and architectural experiments stay out of the main
path unless they make the central idea simpler to understand.
