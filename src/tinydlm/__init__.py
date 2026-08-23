"""Shared axis vocabulary for the core modules; each module repeats only what its main data flow
uses:

    B    batch                 T    query position    S    key position (always == T)
    V    vocabulary            D    model width       H    heads
    Dh   head dim, D = H * Dh  Dff  MLP hidden width  Dh2  half a head dim, Dh / 2
    K    positions revealed in one denoising step, K <= T
    1    a broadcast axis kept on purpose: t is (B, 1) so it broadcasts across T

Names say what a tensor is, not what shape it has. Important transitions in the core expose
their axes through an einops pattern or a concise trailing comment; routine shape-preserving
lines stay quiet. Rearrangement goes through einops; `.view` / `.transpose` / `.permute` are
banned in this package. See STYLE.md.
"""
