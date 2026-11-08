"""Fifth study: which averaging rate is best once endings are drawn as Bernoulli?

Studies 2-4 found that exact per-unit averages with Bernoulli endings beat
the alpha-0.9 EMA memory and a same-memory FIFO buffer, but every EMA rate
other than 0.9 was tested only with rounded endings, and those EMAs started
each unit at zero (a spurious first variance that lasts longer for small
rates). This study compares averaging rates with Bernoulli endings and a
clean start: ``averaging="hybrid"`` with ``kappa_min = a`` is exact until the
rate 1/n falls to ``a`` and an exponential average with weight ``a`` on the
new transition afterwards (no zero start, no correction).

It also tests "fast early, exact later": rate ``max(1/n, 0.9)`` while the
memory has received at most 10 000 transitions, exact (1/n) afterwards
(``kappa_until=10000``). The exact memory trails FIFO at 5k and 10k steps and
leads from 25k on (Studies 3-4); the fast EMA scored higher at 5k.

Arms (all diagonal, Bernoulli endings, uniform choice of units), NEW seeds
160-179 (20 seeds); agent, task, budget and evaluation as in ``experiments.py``:

* ``rate-0.9``, ``rate-0.5``, ``rate-0.1``, ``rate-0.03`` -- clean-start EMA
* ``hybrid-bern`` -- rate 0.01 (the hybrid arm of Study 4)
* ``exact-bern`` -- exact averaging
* ``fast-early-exact`` -- rate 0.9 for the first 10 000 transitions, then exact
* ``fifo-310`` -- real FIFO buffer of the same memory

Scores (as in ``experiments_burnin.py``): early = mean at 5k, 10k, 25k;
whole-run = mean over the 8 checkpoints; late = mean from 100k; 5k = the first
checkpoint. Paired by seed, 95% percentile bootstrap over seeds.

Claims, stated before the run (all reported whichever way they fall):

1. ``rate-0.9`` trails ``exact-bern`` on the late score.
2. ``rate-0.03`` and ``hybrid-bern`` do not differ from ``exact-bern`` on the
   whole-run score (intervals include 0).
3. ``fast-early-exact`` beats ``exact-bern`` at the 5k checkpoint and does not
   lose to it on the whole-run score (interval not entirely below 0).
4. No direction predicted, reported: every rate against ``exact-bern`` and
   against ``fifo-310`` on all scores; generated ending rates.

Run:     sh run_rate.sh 26
Summary: python experiments_rate.py --summarize
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import experiments as E
import experiments_burnin as B  # registers exact-bern and hybrid-bern; provides early score
from structured_som import StructuredSOMBuffer

SEEDS = tuple(range(160, 180))
OUT = Path(__file__).resolve().parent / "out" / "rate_200k"


def _rate(kappa_min: float, kappa_until: int | None = None) -> Callable[[int], Any]:
    """Factory of a clean-start averaging arm with Bernoulli endings.

    Args:
        kappa_min: Floor of the averaging rate (EMA weight on the new transition).
        kappa_until: Total transitions after which the floor is dropped (exact
            averaging from then on); ``None`` keeps it throughout.

    Returns:
        A function from seed to a new buffer.
    """

    def make(seed: int) -> StructuredSOMBuffer:
        """Build the buffer.

        Args:
            seed: Seed of the buffer.

        Returns:
            The buffer.
        """
        return StructuredSOMBuffer(averaging="hybrid", kappa_min=kappa_min, kappa_until=kappa_until, done_decode="bernoulli", seed=seed)

    return make


E.ARMS.update({"rate-0.9": _rate(0.9), "rate-0.5": _rate(0.5), "rate-0.1": _rate(0.1), "rate-0.03": _rate(0.03),
               "fast-early-exact": _rate(0.9, 10_000)})
STUDY = ["rate-0.9", "rate-0.5", "rate-0.1", "rate-0.03", "hybrid-bern", "exact-bern", "fast-early-exact", "fifo-310"]


def summarize(out: Path = OUT) -> None:
    """Print mean scores, ending rates and the claims' paired differences.

    Args:
        out: Directory of the result files.

    Returns:
        None. Prints the summary.
    """
    data = E.load(out)
    rng = np.random.default_rng(0)
    cols = [str(c) for c in E.CHECKPOINTS]
    print("== mean greedy return per checkpoint (seeds 160-179); early = 5k-25k, whole = all, late = from 100k")
    print(f"{'arm':18s} n  " + " ".join(f"{int(c) // 1000:>5}k" for c in cols) + "  early  whole   late  seedSD  end-rate")
    for arm in STUDY:
        if arm not in data:
            continue
        runs = list(data[arm].values())
        whole = [B.score(r, "curve") for r in runs]
        end = np.mean([r["diag"]["done_rate"] for r in runs]) if "done_rate" in runs[0]["diag"] else float("nan")
        print(f"{arm:18s} {len(runs):2d} " + " ".join(f"{np.mean([B.score(r, c) for r in runs]):6.1f}" for c in cols)
              + f" {np.mean([B.score(r, 'early') for r in runs]):6.1f} {np.mean(whole):6.1f} {np.mean([B.score(r, 'late') for r in runs]):6.1f}"
              f" {np.std(whole, ddof=1):6.1f}  {end:.4f}")
    pairs = [("rate-0.9", "exact-bern", "claim 1"), ("rate-0.03", "exact-bern", "claim 2"), ("hybrid-bern", "exact-bern", "claim 2"),
             ("fast-early-exact", "exact-bern", "claim 3"), ("rate-0.5", "exact-bern", "4"), ("rate-0.1", "exact-bern", "4")]
    pairs += [(a, "fifo-310", "4") for a in STUDY if a != "fifo-310"]
    for kind in ("early", "curve", "late", *cols):
        print(f"\n== paired differences, score = {kind}")
        for a, b, tag in pairs:
            s = B.paired(data, a, b, kind, rng)
            if s:
                print(f"[{tag:7s}] {a} - {b}: {s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}], won {s['won']}/{s['n']}")


def main() -> None:
    """Command line: run arms and seeds, or summarise.

    Returns:
        None.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs="+", default=STUDY, choices=STUDY)
    p.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--steps", type=int, default=E.STEPS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize(a.out)
        return
    cps = tuple(c for c in E.CHECKPOINTS if c <= a.steps) or (a.steps,)
    for seed in a.seed:
        for arm in a.arm:
            if (a.out / f"{arm}_{seed}.json").exists():
                continue
            E.run(arm, seed, a.steps, cps, a.out)


if __name__ == "__main__":
    main()
