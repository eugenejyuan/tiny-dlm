"""Named pretokenization patterns from OpenAI's public tiktoken definitions."""

import regex

# Exact strings from tiktoken_ext/openai_public.py in openai/tiktoken: GPT2 is the
# r50k/p50k pattern, CL100K serves GPT-3.5/GPT-4, and O200K serves GPT-4o.
GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
CL100K = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
O200K = "|".join(
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
PATTERNS = {"gpt2": GPT2, "cl100k": CL100K, "o200k": O200K}
DEFAULT = "o200k"


def pattern(name: str = DEFAULT) -> regex.Pattern:
    assert name in PATTERNS, f"unknown pattern {name!r}: {sorted(PATTERNS)}"
    return regex.compile(PATTERNS[name])
