"""Action purity of the committed WeightedSOMBuffer trained on the recorded CartPole data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import load  # noqa: E402
from SOMBuffer import WeightedSOMBuffer  # noqa: E402


def main() -> None:
    """Train on the first 10k and 60k transitions (episode index as step) and print unit action purity.

    Returns:
        None.
    """
    X = load()
    for N in (10000, 60000):
        np.random.seed(0)
        som = WeightedSOMBuffer()
        ep = 0
        for x in X[:N]:
            som.add(x[:4], x[4], x[5], x[6:10], bool(x[10]), ep)
            ep += int(x[10])
        b = np.array([i * 10 + j for i, j in (som._get_bmu(x) for x in X[:N])])
        a = X[:N, 4]
        pur = mixed = 0.0
        for u in np.unique(b):
            m = b == u
            p = a[m].mean()
            pur += max(p, 1 - p) * m.sum()
            mixed += m.sum() * (min(p, 1 - p) > 0.2)
        print(f"N={N} episodes={ep}: units used {len(np.unique(b))}/100, action purity {pur / N:.3f}, share in units with >20% minority action {mixed / N:.3f}")


if __name__ == "__main__":
    main()
