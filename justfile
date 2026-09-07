set positional-arguments

# List the available commands.
default:
    @just --list

# Download the full TinyStories source splits.
fetch:
    uv run python -m tinydlm.data.tinystories --raw-dir data/tinystories/raw

# Train BPE on the full training split and encode both splits; use --max-mb for a subset.
prepare *args:
    uv run python -m tinydlm.data.prepare \
        --train data/tinystories/raw/TinyStoriesV2-GPT4-train.txt \
        --val data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt \
        --output-dir data/tinystories/prepared \
        --vocab-size 4096 --workers 1 "$@"

# Train with the Python CLI's model/optimizer defaults unless overridden; --name picks the run folder.
train *args:
    uv run python -m tinydlm.train \
        --prepared-dir data/tinystories/prepared \
        --runs-dir runs "$@"

# Sample using the tokenizer and configuration stored in the checkpoint.
sample *args:
    uv run python -m tinydlm.sample --checkpoint runs/run/checkpoint.pt "$@"
