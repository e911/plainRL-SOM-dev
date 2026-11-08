"""Load the repository's recorded CartPole transitions as an (N, 11) array [s, a, r, s', done]."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "generate_sample_rb" / "rb_samples" / "rb_samples_20261029_165302_CartPole-v1_60000_steps.txt"


def load(limit: int | None = None) -> np.ndarray:
    """Parse the tensor-repr text file.

    Args:
        limit: Keep only the first ``limit`` transitions (``None`` keeps all).

    Returns:
        Array of shape (N, 11).
    """
    rows = []
    for line in open(DATA):
        rows.append(np.hstack([np.ravel(x) for x in ast.literal_eval(line.strip().replace("tensor(", " ("))]))
        if limit is not None and len(rows) >= limit:
            break
    return np.array(rows, dtype=float)
