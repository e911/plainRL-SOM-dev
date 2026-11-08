"""Pre-run validation of StructuredSOMBuffer on the first 10k recorded CartPole transitions."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import load  # noqa: E402
from structured_som import StructuredSOMBuffer  # noqa: E402


def main() -> None:
    """Check the corrections after one hit, then fit four variants and report samples, physics error, purity, ending units.

    Returns:
        None.
    """
    X = load(10000)
    x = X[0]
    raw = np.r_[x[:4], x[4], x[5], x[6:10] - x[:4], x[10]]
    for c in ("fixed", "buggy", "none"):
        b = StructuredSOMBuffer(correction=c, seed=0)
        b.add(x[:4], int(x[4]), x[5], x[6:10], bool(x[10]), 0)
        hit = np.argwhere(b.count > 0)[0]
        m = b.unit_stats(hit[:1], hit[1:2])[0][0]
        print(f"{c}: 1 hit, mean == x? {np.allclose(m, raw)}, reward {m[5]:.3f}")
    env = gym.make("CartPole-v1").unwrapped
    env.reset(seed=0)

    def phys(s: np.ndarray, a: int) -> np.ndarray:
        """True next state from state s and action a.

        Args:
            s: State.
            a: Action.

        Returns:
            Next state.
        """
        env.state = np.array(s, dtype=np.float64)
        return env.step(int(a))[0]

    variants = [("md-fixed-diag", dict(correction="fixed")), ("md-exact-diag", dict(averaging="exact")),
                ("md-fixed-full", dict(correction="fixed", covariance="full")), ("md-exact-full", dict(averaging="exact", covariance="full"))]
    for name, kw in variants:
        b = StructuredSOMBuffer(seed=1, **kw)
        ep = 0
        for x in X:
            b.add(x[:4], int(x[4]), x[5], x[6:10], bool(x[10]), ep)
            ep += int(x[10])
        s, a, r, s2, d = b.sample(3000)
        err = np.array([np.abs(phys(si, ai) - s2i).mean() for si, ai, s2i in zip(s[:500], a[:500], s2[:500])])
        V = X[:3000]
        ids = np.array([i * 10 + j for i, j in (np.unravel_index(np.argmin(b._distance(b._standardise(np.r_[v[:4], v[4], v[5], v[6:10] - v[:4], v[10]]))),
                                                                b.grid_size) for v in V)])
        pur = sum(max(V[ids == u, 4].mean(), 1 - V[ids == u, 4].mean()) * (ids == u).sum() for u in np.unique(ids)) / len(V)
        endu = {u for u in np.unique(ids) if V[ids == u, 10].mean() > 0.5}
        ends = np.where(V[:, 10] == 1)[0]
        print(f"{name}: units hit {(b.count > 0).sum()}, memory {b.memory_floats()}, sample done {d.mean():.3f} (data {X[:, 10].mean():.3f}), "
              f"reward [{r.min():.2f},{r.max():.2f}], physics err {np.median(err):.4f}, action purity {pur:.3f}, "
              f"endings in done-units {np.mean([ids[k] in endu for k in ends]):.2f}")


if __name__ == "__main__":
    main()
