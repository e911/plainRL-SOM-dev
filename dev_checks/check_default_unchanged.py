"""Check that StructuredSOMBuffer's defaults are bit-identical to the version used for the main run (Entry 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _structured_som_v1 as v1  # noqa: E402
import structured_som as v2  # noqa: E402
from load_data import load  # noqa: E402


def main() -> None:
    """Feed 10k transitions to both versions under several options and compare state and samples.

    Returns:
        None. Prints one line per option set and raises AssertionError on any difference.
    """
    X = load(10000)
    for kw in (dict(), dict(averaging="exact"), dict(covariance="full"), dict(correction="buggy"), dict(unit_choice="count")):
        a, b = v1.StructuredSOMBuffer(seed=3, **kw), v2.StructuredSOMBuffer(seed=3, **kw)
        ep = 0
        for x in X:
            for buf in (a, b):
                buf.add(x[:4], int(x[4]), x[5], x[6:10], bool(x[10]), ep)
            ep += int(x[10])
        assert np.array_equal(a.weights, b.weights) and np.array_equal(a.mean, b.mean) and np.array_equal(a.spread, b.spread)
        for _ in range(5):
            for u, v in zip(a.sample(64), b.sample(64)):
                assert np.array_equal(u, v)
        print(kw or "defaults", "identical")


if __name__ == "__main__":
    main()
