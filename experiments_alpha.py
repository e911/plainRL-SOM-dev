"""Second study: recency (averaging window) or ending rate -- what makes the fixed SOM buffer work?

The main run (``experiments.py``, seeds 100-109) found that the fixed
diagonal SOM (EMA with weight alpha = 0.9 on the new transition) beat the
same SOM with exact running means by 86 points, and that the exact-mean SOM
generated far fewer endings (1.0% vs 4.0%). Two explanations are confounded
there: (a) recency -- with alpha = 0.9 each unit's mean is ~90% its latest
transition, while the exact mean weights the unit's whole history equally;
(b) the ending rate -- generated endings are decoded by rounding a Gaussian
draw, which rarely gives 1 when a unit's mean ending value is small.

Design. Same agent, task, budget and evaluation as ``experiments.py``
(CartPole-v1, 200 000 steps, greedy evaluation at 5k ... 200k), NEW seeds
110-119 (so the reference arms also replicate the main run). All SOM arms:
structured distance, running standardisation, diagonal Gaussians, fixed
correction, units chosen uniformly.

* Averaging window: ``md-a0.99``, ``md-fixed-diag`` (alpha 0.9), ``md-a0.5``,
  ``md-a0.1``, ``md-a0.01``, ``md-exact-diag`` (all of history). The effective
  number of a unit's recent hits averaged grows from ~1 to everything.
* Ending decoding: ``md-fixed-diag-bern`` and ``md-exact-diag-bern`` draw the
  ending flag as Bernoulli(unit's mean ending value) instead of rounding a
  Gaussian draw, so both generate endings at their units' recorded rates.
* ``fifo-310``: real FIFO buffer of the same memory (3 410 numbers).

Claims, stated before the run (whole-run score = mean greedy return over the 8
checkpoints; paired by seed; 95% bootstrap over seeds; all reported whichever
way they fall):

1. Recency: ``md-fixed-diag`` (alpha 0.9) beats ``md-a0.01`` and
   ``md-exact-diag``.
2. Replication on new seeds: ``md-fixed-diag`` does not lose to ``fifo-310``
   on the whole-run score (interval not entirely below 0), and trails it on
   the late score (mean difference below 0).
3. Ending decoding, decision rule (no direction predicted): if
   ``md-exact-diag-bern - md-fixed-diag-bern`` has an interval including 0
   while ``md-exact-diag - md-fixed-diag`` is below 0, the ending rate
   explains the exact mean's deficit; if ``md-exact-diag-bern`` still loses,
   recency does. Generated ending rates are reported for every arm.
4. The alpha sweep (0.99, 0.5, 0.1) is reported without a predicted shape.

Run:     sh run_alpha.sh 26          (all arms x seeds; resumable)
Summary: python experiments_alpha.py --summarize
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import experiments as E
from structured_som import StructuredSOMBuffer

SEEDS = tuple(range(110, 120))
OUT = Path(__file__).resolve().parent / "out" / "alpha_200k"


def _som(alpha: float = 0.9, averaging: str = "ema", done_decode: str = "round") -> Callable[[int], Any]:
    """Factory of a fixed-correction diagonal SOM arm.

    Args:
        alpha: EMA weight on the new transition.
        averaging: ``"ema"`` or ``"exact"``.
        done_decode: ``"round"`` or ``"bernoulli"``.

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
        return StructuredSOMBuffer(alpha=alpha, averaging=averaging, correction="fixed", done_decode=done_decode, seed=seed)

    return make


NEW_ARMS: dict[str, Callable[[int], Any]] = {
    "md-a0.99": _som(0.99),
    "md-a0.5": _som(0.5),
    "md-a0.1": _som(0.1),
    "md-a0.01": _som(0.01),
    "md-fixed-diag-bern": _som(0.9, done_decode="bernoulli"),
    "md-exact-diag-bern": _som(averaging="exact", done_decode="bernoulli"),
}
E.ARMS.update(NEW_ARMS)
STUDY = ["md-a0.99", "md-fixed-diag", "md-a0.5", "md-a0.1", "md-a0.01", "md-exact-diag", "md-fixed-diag-bern", "md-exact-diag-bern", "fifo-310"]
LABEL = {"md-a0.99": "alpha 0.99", "md-fixed-diag": "alpha 0.9", "md-a0.5": "alpha 0.5", "md-a0.1": "alpha 0.1", "md-a0.01": "alpha 0.01",
         "md-exact-diag": "exact", "md-fixed-diag-bern": "alpha 0.9, Bernoulli end", "md-exact-diag-bern": "exact, Bernoulli end",
         "fifo-310": "FIFO 310"}


def summarize(out: Path = OUT) -> None:
    """Print mean scores, generated ending rates and the claims' paired differences.

    Args:
        out: Directory of the result files.

    Returns:
        None. Prints the summary.
    """
    data = E.load(out)
    rng = np.random.default_rng(0)
    cols = [str(c) for c in E.CHECKPOINTS]
    print("== mean greedy return per checkpoint (seeds 110-119), whole = mean over checkpoints, late = mean from 100k")
    print(f"{'arm':26s} n  " + " ".join(f"{int(c) // 1000:>5}k" for c in cols) + "  whole   late  seedSD  end-rate")
    for arm in STUDY:
        if arm not in data:
            continue
        runs = list(data[arm].values())
        whole = [E.score(r, "curve") for r in runs]
        end = np.mean([r["diag"]["done_rate"] for r in runs]) if "done_rate" in runs[0]["diag"] else float("nan")
        print(f"{LABEL[arm]:26s} {len(runs):2d} " + " ".join(f"{np.mean([E.score(r, c) for r in runs]):6.1f}" for c in cols)
              + f" {np.mean(whole):6.1f} {np.mean([E.score(r, 'late') for r in runs]):6.1f} {np.std(whole, ddof=1):6.1f}  {end:.4f}")
    pairs = [("md-fixed-diag", "md-a0.01", "claim 1"), ("md-fixed-diag", "md-exact-diag", "claim 1"), ("md-fixed-diag", "fifo-310", "claim 2"),
             ("md-exact-diag-bern", "md-fixed-diag-bern", "claim 3"), ("md-exact-diag", "md-fixed-diag", "claim 3"),
             ("md-fixed-diag-bern", "md-fixed-diag", "3"), ("md-exact-diag-bern", "md-exact-diag", "3"),
             ("md-a0.99", "md-fixed-diag", "4"), ("md-a0.5", "md-fixed-diag", "4"), ("md-a0.1", "md-fixed-diag", "4"),
             ("md-a0.01", "md-fixed-diag", "4"), ("md-exact-diag", "md-a0.01", "4")]
    pairs += [(a, "fifo-310", "4") for a in STUDY if a not in ("fifo-310", "md-fixed-diag")]
    for kind in ("curve", "late", *cols):
        print(f"\n== paired differences, score = {kind}")
        for a, b, tag in pairs:
            s = E.paired(data, a, b, kind, rng)
            if s:
                print(f"[{tag:7s}] {LABEL[a]} - {LABEL[b]}: {s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}], won {s['won']}/{s['n']}")


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
