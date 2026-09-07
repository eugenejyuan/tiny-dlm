# tinyDLM

A masked diffusion language model written from scratch, small enough to read in one sitting.
Personal playground: I use it to understand diffusion LMs by building one, not to ship a library.

Trained end to end on TinyStories; it generates readable text. Results below.

## Run

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/), and [just](https://github.com/casey/just)
(`uv tool install rust-just`). Recipes live in [justfile](justfile); `uv run` handles the venv.

```sh
just fetch      # downloads the TinyStories splits
just prepare    # trains BPE (vocab 4096) and encodes both splits to uint16 bins
just train      # writes runs/run/checkpoint.pt
just sample --prompt "Once upon a time"
```

Training claims its own folder under `runs/` from `--name`. 

## Results

One real run so far: 66k steps (~ 1 epoch) on the full TinyStories training split, defaults otherwise
(depth 6, seq-len 256, batch 32, lr 3e-4, single GPU).

- Final training loss **2.3**, validation **2.176**.
- Samples read fine at the default step count. Quality falls off fast when you cut denoising steps.

Not tuned, not compared against anything. A working baseline, not a number to cite.

## TODO

- training monitoring
- scale up and evaluation
