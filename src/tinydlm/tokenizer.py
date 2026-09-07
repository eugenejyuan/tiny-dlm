"""Byte-level BPE: learn merges, then apply them in rank order.

Vocabulary layout for V tokens:
    0 .. 255      raw bytes, id == byte value
    256 .. V-9    learned merges
    V-8           <|sep|>, inserted by the data layer
    V-7           <|mask|>, inserted by the objective
    V-6 .. V-1    reserved special tokens
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import pairwise
from mmap import ACCESS_READ, mmap
from pathlib import Path

import regex
from tqdm import tqdm

N_BYTES = 256
SEP_TOKEN, MASK_TOKEN = "<|sep|>", "<|mask|>"
SPECIAL_TOKENS = (SEP_TOKEN, MASK_TOKEN, *(f"<|reserved_{i}|>" for i in range(6)))
TOKENIZER_FILE = "tokenizer.json"

# GPT-4o's pretokenizer, verbatim from tiktoken_ext/openai_public.py in openai/tiktoken.
# Merges never cross a match boundary, so this pattern is part of the tokenizer contract.
PRETOKEN_PATTERN = regex.compile(
    "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )
)


def _delimiters(split_on: Sequence[str]) -> str:
    """Match literal delimiters, longest first so a prefix cannot shadow a longer one."""
    assert split_on and all(split_on), f"delimiters must be nonempty: {split_on}"
    ordered = sorted(split_on, key=len, reverse=True)
    return "|".join(regex.escape(delimiter) for delimiter in ordered)


def _chunk_bounds(
    path: Path, delimiters: str, limit: int | None, *, chunk_bytes: int = 16 * 1024 * 1024
) -> list[int]:
    """Cut after complete delimiters once a chunk reaches `chunk_bytes`.
    """
    end = path.stat().st_size
    if limit is not None:
        end = min(limit, end)
    if end <= chunk_bytes:  # also handles empty files, which cannot be memory-mapped
        return [0, end]

    bounds = [0]
    with path.open("rb") as handle, mmap(handle.fileno(), end, access=ACCESS_READ) as corpus:
        for match in regex.finditer(delimiters.encode(), corpus):
            boundary = match.end()
            if bounds[-1] + chunk_bytes <= boundary < end:
                bounds.append(boundary)
    return [*bounds, end]


def _count_range(path: Path, delimiters: str, span: tuple[int, int]) -> Counter[str]:
    """Read one byte range and count pretokens without crossing delimiters."""
    start, end = span
    with path.open("rb") as handle:
        handle.seek(start)
        text = handle.read(end - start).decode("utf-8", errors="replace")
    counts: Counter[str] = Counter()
    for chunk in regex.splititer(delimiters, text):
        counts.update(match[0] for match in PRETOKEN_PATTERN.finditer(chunk))
    return counts


def _count_corpus(
    path: Path, split_on: Sequence[str], workers: int, limit: int | None
) -> dict[tuple[int, ...], int]:
    """Combine chunk counts, then encode each distinct pretoken once.
    """
    assert workers > 0, f"workers must be positive, got {workers}"
    delimiters = _delimiters(split_on)
    ranges = list(pairwise(_chunk_bounds(path, delimiters, limit)))
    count = partial(_count_range, path, delimiters)

    counts: Counter[str] = Counter()
    if workers == 1 or len(ranges) == 1:
        for span in tqdm(ranges, desc="counting chunks"):
            counts.update(count(span))
    else:
        with ProcessPoolExecutor(min(workers, len(ranges))) as pool:
            for done in tqdm(pool.map(count, ranges), total=len(ranges), desc="counting chunks"):
                counts.update(done)
    return {tuple(word.encode()): n for word, n in counts.items()}


def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Merge greedily left to right: (a, a, a) with pair (a, a) becomes (new, a).
    Training and encoding share this operation.
    """
    merged, i = [], 0
    while i < len(word):
        matched = word[i : i + 2] == pair
        merged.append(new_id if matched else word[i])
        i += 2 if matched else 1
    return tuple(merged)


def train_bpe_file(
    path: Path,
    vocab_size: int,
    split_on: Sequence[str] = SPECIAL_TOKENS,
    workers: int = 1,
    limit: int | None = None,
) -> list[tuple[int, int]]:
    """Count the corpus, then merge its most frequent pair until the vocabulary is full.
    `pair_words` indexes the words containing each pair, so each merge only updates those
    words and their counts instead of recounting the corpus.
    """
    base_size = N_BYTES + len(SPECIAL_TOKENS)
    n_merges = vocab_size - base_size
    if n_merges <= 0:
        raise ValueError(f"vocab_size must exceed {base_size}, got {vocab_size}")

    counts = _count_corpus(path, split_on, workers, limit)
    words, frequencies = list(counts), list(counts.values())
    vocab = [bytes([byte]) for byte in range(N_BYTES)]
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    def add_word(i: int) -> None:
        for pair in pairwise(words[i]):
            pair_counts[pair] += frequencies[i]
            pair_words[pair].add(i)

    def remove_word(i: int) -> None:
        for pair in pairwise(words[i]):
            pair_counts[pair] -= frequencies[i]
            pair_words[pair].discard(i)
            if not pair_counts[pair]:
                del pair_words[pair]
                del pair_counts[pair]
 
    for i in range(len(words)):
        add_word(i)

    merges: list[tuple[int, int]] = []
    for new_id in tqdm(range(N_BYTES, N_BYTES + n_merges), desc="bpe merges"):
        if not pair_counts:
            break  # corpus exhausted: every pretoken is already a single token
        # Break frequency ties by token bytes, independent of worker completion order.
        pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        for i in sorted(pair_words[pair]):
            remove_word(i)
            words[i] = _merge(words[i], pair, new_id)
            add_word(i)
        assert pair not in pair_words  # greedy merging removes every occurrence
        vocab.append(vocab[pair[0]] + vocab[pair[1]])
        merges.append(pair)

    if len(merges) != n_merges:
        raise ValueError(
            f"corpus ran out of pairs after {len(merges)} merges; "
            f"use vocab_size={base_size + len(merges)}"
        )
    return merges


class Tokenizer:
    """A vocabulary rebuilt from its ranked merges, plus the special-token block above it."""

    def __init__(self, merges: Sequence[Sequence[int]]):
        self.merges = [(left, right) for left, right in merges]
        self.ranks = {pair: N_BYTES + rank for rank, pair in enumerate(self.merges)}
        self.vocab = [bytes([byte]) for byte in range(N_BYTES)]
        for left, right in self.merges:
            self.vocab.append(self.vocab[left] + self.vocab[right])
        self.sep_id = len(self.vocab) + SPECIAL_TOKENS.index(SEP_TOKEN)
        self.mask_id = len(self.vocab) + SPECIAL_TOKENS.index(MASK_TOKEN)
        self.vocab.extend(token.encode() for token in SPECIAL_TOKENS)
        self.vocab_size = len(self.vocab)
        self._cache: dict[bytes, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        """Apply merges in rank order. Literal special-token text encodes as ordinary bytes;
        only the data layer and objective insert special IDs.
        """
        ids: list[int] = []
        for match in PRETOKEN_PATTERN.finditer(text):
            ids.extend(self._encode_word(match[0].encode()))
        return ids

    def _encode_word(self, word: bytes) -> list[int]:
        """Merge a pretoken lowest rank first; reuse the result for repeated pretokens."""
        if word in self._cache:
            return self._cache[word]

        ids = tuple(word)
        while len(ids) > 1:
            pair = min(pairwise(ids), key=lambda p: self.ranks.get(p, self.vocab_size))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.ranks[pair])
        self._cache[word] = list(ids)
        return self._cache[word]

    def decode(self, ids: Iterable[int]) -> str:
        """Join token bytes before UTF-8 decoding, since a character can span tokens.
        Special IDs decode to their visible spelling.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.merges), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))
