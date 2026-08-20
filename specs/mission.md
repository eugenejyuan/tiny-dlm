# Mission

> **A minimal masked diffusion language model, built for clarity.**

tinyDLM is a compact toy implementation of a masked diffusion language model. It is written
for someone who already understands a Transformer but is new to language diffusion.

The goal is simple: after reading the core code for an hour, a reader should be able to explain

- how a clean sequence becomes a randomly masked sequence at noise level `t`;
- why a bidirectional Transformer can denoise it without a causal mask; and
- how iterative unmasking turns masked-token prediction into generation and infilling.

The reader should not need to treat a training framework, model library, or sampler abstraction
as magic.

## Clarity through minimal design

tinyDLM stays understandable by keeping its structure small and direct. Each core module has one
job, and the path from text to loss to sample should be visible without tracing a framework or a
deep abstraction tree.

Minimal does not mean compressed. It means keeping only what clarifies the model:

1. **One obvious path.** Data preparation, corruption, denoising, training, and sampling each
   have a clear place.
2. **One responsibility per module.** The file structure mirrors the algorithm rather than a
   framework architecture.
3. **Math beside the code.** Equations appear where they make a core operation easier to
   understand.
4. **Shapes at meaningful transitions.** Core tensors are labeled where dimensions enter,
   change, broadcast, or become important to the data flow—not on every line.
5. **Variants outside the core.** Optional experiments belong in `lab/`, not behind feature
   flags in the main path.

## Design signature

tinyDLM does not claim to invent the algorithms it cites. Its contribution is the way those
ideas are selected, composed, and exposed for learning:

- **A Transformer with one line missing.** The attention scores are materialized, and the
  absent causal mask is visible exactly where a GPT would apply it.
- **One loss that explains the model.** Random mask density, masked-token cross-entropy, and
  the `1/t` weight fit in one short function; the architecture and sampler follow from it.
- **The corrupted sequence is its own clock.** The model receives no timestep embedding: the
  density of `[MASK]` tokens already exposes the noise level.
- **The canvas is the sampler state.** A position is undecided if and only if it still contains
  `[MASK]`; prompts, suffixes, and revealed tokens are therefore frozen by construction.
- **Document boundaries remain learnable.** No token is exempt from corruption. In particular,
  `<|sep|>` must be predicted because it is how a generated story learns to end.
- **One integer describes model scale.** `depth` derives width and head count while keeping the
  head dimension fixed, so runs remain easy to compare.
- **A manifest separates data from datasets.** Fetching is corpus-specific; tokenization and
  binary preparation are offline and corpus-independent. Private data needs JSON, not a code
  change.
- **The structure mirrors the algorithm.** Model, objective, training, and sampling each have one
  canonical home and can be read in data-flow order.
- **Shapes mark important transitions.** Core modules share a small axis vocabulary, and
  `einops` patterns expose the projections and rearrangements that matter. Routine
  shape-preserving lines stay quiet.

These choices are small individually. Together they make the repository's central argument:
masked diffusion is not an opaque new stack, but a short sequence of inspectable decisions.

## Project constitution

The following rules protect that argument as the project grows:

1. Executable code under `src/tinydlm/` stays within an approximate 800-line guardrail. The
   budget protects a small design; reaching a particular number is not the goal.
2. Important tensor transitions in the core use an `einops` pattern or a concise axis comment.
   Shape annotations are omitted when they add no information. Raw `.view`, `.transpose`, and
   `.permute` calls are excluded from the package.
3. Core docstrings include math and rationale where they clarify the algorithm; comments explain
   *why*, not a second version of *what*.
4. Domain-specific failure modes become assertions or tests.
5. Each core concept has one canonical implementation path and one obvious home.
6. Source code, comments, logs, and tests use clear ASCII English so the core remains readable
   to an international audience.

The full notation and style rules live in [STYLE.md](../STYLE.md).

## Scope

tinyDLM includes the complete educational path: byte-level BPE, corpus preparation, a
bidirectional denoiser, the absorbing-state ELBO, a single-device training loop, and
confidence-ordered generation with prefix and suffix infilling.

It deliberately does not aim to be a serving framework, a general model zoo, or a chat system.
SFT, RL, distributed training, and large-scale infrastructure belong in other projects. Baselines,
sweeps, and experimental variants stay under `lab/` so exploration never enlarges the core a
reader must understand.

## Release bar

A public release is complete when all of the following are true:

- `just smoke` runs the end-to-end path on CPU;
- a default training run produces readable TinyStories samples;
- `just test`, `just lint`, and `just lines` pass;
- the README presents one short path from setup to sampling; and
- the core training objective and sampler can be understood without leaving the repository.
