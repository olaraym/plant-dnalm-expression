"""k-mer frequency features: the simple baseline the language models must beat."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_LUT = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i
    _LUT[ord(chr(_b).lower())] = _i


def kmer_frequencies(seq: str, k: int) -> np.ndarray:
    """Normalized counts of all 4**k k-mers in one sequence. Windows containing N are skipped."""
    codes = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    out = np.zeros(4**k, dtype=np.float32)
    if len(codes) < k:
        return out
    win = sliding_window_view(codes, k)
    valid = (win >= 0).all(axis=1)
    if not valid.any():
        return out
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    idx = win[valid] @ powers
    counts = np.bincount(idx, minlength=4**k).astype(np.float32)
    return counts / counts.sum()


def kmer_matrix(seqs: list[str], ks: tuple[int, ...] = (6,)) -> np.ndarray:
    """Feature matrix (n_sequences, sum of 4**k) for one or more k values."""
    blocks = [np.vstack([kmer_frequencies(s, k) for s in seqs]) for k in ks]
    return np.hstack(blocks).astype(np.float32)


def kmer_names(k: int) -> list[str]:
    """k-mer strings in the same order as the feature columns (useful for interpreting weights)."""
    names = [""]
    for _ in range(k):
        names = [n + b for n in names for b in "ACGT"]
    return names
