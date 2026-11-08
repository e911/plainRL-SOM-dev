"""Fourth study: can a burn-in rule fix the slow start of the exact-mean + Bernoulli SOM buffer?

The exact-mean + Bernoulli-endings buffer (best SOM buffer; studies 2-3) was
far behind FIFO at 5k steps (91 vs 182 on seeds 120-139). A diagnostic on
development seeds 997-999 (``dev_checks/early_diagnostic.py``,
``dev_checks/out_early_diagnostic.txt``) found that units with fewer than 5
hits produce 23-38% of generated transitions at 1k steps but only 4-10% at 5k,
the same for every buffer; while the physics error of the exact-mean buffer's
generated transitions did not fall during the first 25k steps (about
0.009-0.018) although the alpha-0.9 buffer's fell to about 0.005. Exact means
keep averaging transitions from where a unit used to be while the map still
moves. Two remedies are tested:

* burn-in: only units with at least ``K`` hits generate (``min_hits=K``; if
  none qualifies yet, every hit unit generates);
* hybrid averaging: exact mean and variance for a unit's first 100 hits, then
  an exponential average with rate 0.01 that slowly forgets
  (``averaging="hybrid"``, ``kappa_min=0.01``; the rule of the main RLSOM
  repository's SOM).

Arms (all diagonal, Bernoulli endings, uniform choice among eligible units),
NEW seeds 140-159 (20 seeds); agent, task, budget and evaluation as in
``experiments.py``:

* ``exact-bern`` (reference, = ``md-exact-diag-bern``)
* ``exact-bern-k5``, ``exact-bern-k20`` (burn-in)
* ``hybrid-bern``, ``hybrid-bern-k20``
* ``fifo-310``

Scores: early = mean greedy return at 5k, 10k and 25k; whole-run = mean over
the 8 checkpoints; late = mean from 100k. Paired by seed, 95% percentile
bootstrap over seeds.

Claims, stated before the run (all reported whichever way they fall):

1. ``exact-bern-k20`` beats ``exact-bern`` on the early score.
2. ``hybrid-bern`` beats ``exact-bern`` on the early score.
3. Neither ``exact-bern-k20`` nor ``hybrid-bern`` loses to ``exact-bern`` on
   the whole-run score (interval not entirely below 0).
4. No direction predicted, reported: ``exact-bern-k5`` and ``hybrid-bern-k20``
   against ``exact-bern``; every arm against ``fifo-310`` on all three scores;
   generated ending rates.

Run:     sh run_burnin.sh 26
Summary: python experiments_burnin.py --summarize
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import experiments as E
from structured_som import StructuredSOMBuffer

SEEDS = tuple(range(140, 160))
OUT = Path(__file__).resolve().parent / "out" / "burnin_200k"
EARLY = (5_000, 10_000, 25_000)


def _som(averaging: str, min_hits: int = 1) -> Callable[[int], Any]:
    """Factory of a diagonal, Bernoulli-ending SOM arm.

    Args:
        averaging: ``"exact"`` or ``"hybrid"``.
        min_hits: Burn-in threshold.

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
        return StructuredSOMBuffer(averaging=averaging, done_decode="bernoulli", min_hits=min_hits, kappa_min=0.01, seed=seed)

    return make


E.ARMS.update({"exact-bern": _som("exact"), "exact-bern-k5": _som("exact", 5), "exact-bern-k20": _som("exact", 20),
               "hybrid-bern": _som("hybrid"), "hybrid-bern-k20": _som("hybrid", 20)})
STUDY = ["exact-bern", "exact-bern-k5", "exact-bern-k20", "hybrid-bern", "hybrid-bern-k20", "fifo-310"]


def score(r: dict[str, Any], kind: str) -> float:
    """One run's score, adding ``"early"`` to :func:`experiments.score`.

    Args:
        r: A saved result.
        kind: ``"early"``, ``"curve"``, ``"late"`` or a checkpoint step as text.

    Returns:
        The score.
    """
    if kind == "early":
        return float(np.mean([r["eval"][str(c)] for c in EARLY]))
    return E.score(r, kind)


def paired(data: dict[str, dict[int, dict[str, Any]]], first: str, second: str, kind: str, rng: np.random.Generator,
           draws: int = 10_000) -> dict[str, float] | None:
    """Paired difference ``first - second`` over common seeds (as :func:`experiments.paired`, with the early score).

    Args:
        data: Results by arm and seed.
        first: First arm.
        second: Second arm.
        kind: Score kind (see :func:`score`).
        rng: Generator of the bootstrap.
        draws: Bootstrap draws.

    Returns:
        ``n``, ``mean``, ``lo``, ``hi`` and ``won``; ``None`` if no common seed.
    """
    seeds = sorted(set(data.get(first, {})) & set(data.get(second, {})))
    if not seeds:
        return None
    diff = np.array([score(data[first][s], kind) - score(data[second][s], kind) for s in seeds])
    boot = diff[rng.integers(len(diff), size=(draws, len(diff)))].mean(1)
    return {"n": len(diff), "mean": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "won": int((diff > 0).sum())}


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
    print("== mean greedy return per checkpoint (seeds 140-159); early = 5k-25k, whole = all, late = from 100k")
    print(f"{'arm':18s} n  " + " ".join(f"{int(c) // 1000:>5}k" for c in cols) + "  early  whole   late  seedSD  end-rate")
    for arm in STUDY:
        if arm not in data:
            continue
        runs = list(data[arm].values())
        whole = [score(r, "curve") for r in runs]
        end = np.mean([r["diag"]["done_rate"] for r in runs]) if "done_rate" in runs[0]["diag"] else float("nan")
        print(f"{arm:18s} {len(runs):2d} " + " ".join(f"{np.mean([score(r, c) for r in runs]):6.1f}" for c in cols)
              + f" {np.mean([score(r, 'early') for r in runs]):6.1f} {np.mean(whole):6.1f} {np.mean([score(r, 'late') for r in runs]):6.1f}"
              f" {np.std(whole, ddof=1):6.1f}  {end:.4f}")
    pairs = [("exact-bern-k20", "exact-bern", "claim 1/3"), ("hybrid-bern", "exact-bern", "claim 2/3"), ("exact-bern-k5", "exact-bern", "4"),
             ("hybrid-bern-k20", "exact-bern", "4"), ("hybrid-bern-k20", "hybrid-bern", "4")]
    pairs += [(a, "fifo-310", "4") for a in STUDY if a != "fifo-310"]
    for kind in ("early", "curve", "late", *cols):
        print(f"\n== paired differences, score = {kind}")
        for a, b, tag in pairs:
            s = paired(data, a, b, kind, rng)
            if s:
                print(f"[{tag:9s}] {a} - {b}: {s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}], won {s['won']}/{s['n']}")


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
