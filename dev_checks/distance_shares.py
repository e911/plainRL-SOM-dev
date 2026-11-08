"""Share of each block of WeightedSOMBuffer's weighted squared distance on the recorded CartPole data."""

from __future__ import annotations

import numpy as np

from load_data import load


def main() -> None:
    """Print each block's share over random pairs and over the closest 5% of pairs, and the raw SDs.

    Returns:
        None.
    """
    X = load()
    w = np.array([0.5, 0.8, 2.5, 1.2, 0.4, 0.15, 0.4, 0.64, 2.0, 0.96, 1.0])
    blocks = {"state": slice(0, 4), "action": slice(4, 5), "reward": slice(5, 6), "next state": slice(6, 10), "done": slice(10, 11)}
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(X), size=200000), rng.integers(len(X), size=200000)
    D = (w * (X[i] - X[j])) ** 2
    tot = D.sum(1).mean()
    for k, s in blocks.items():
        print(f"random pairs {k:10s} weight^2 sum {np.sum(w[s] ** 2):5.2f} share {D[:, s].sum(1).mean() / tot * 100:5.1f}%")
    d = D.sum(1)
    m = d < np.quantile(d, 0.05)
    for k, s in blocks.items():
        print(f"closest 5% {k:10s} share {D[m][:, s].sum(1).mean() / D[m].sum(1).mean() * 100:5.1f}%")
    print("raw SD:", np.round(X.std(0), 3), "done rate", X[:, 10].mean().round(4))


if __name__ == "__main__":
    main()
