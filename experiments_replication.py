"""Third study: pre-registered replication of the exact-mean + Bernoulli-endings SOM buffer.

In the second study (``experiments_alpha.py``, seeds 110-119) the SOM buffer
with exact running means and endings drawn as Bernoulli(unit's mean ending
value) was the best SOM buffer (whole-run 329, late 403), beat the same
buffer with rounded endings (+47.2 whole, +104.6 late) and did not trail the
same-memory FIFO late (-19.8 [-108.1, +70.2]). Those conclusions were not
pre-registered, so this study repeats the comparison on NEW seeds 120-139
(20 seeds), with the claims below written before launch.

Same agent, task, budget and evaluation as ``experiments.py`` (CartPole-v1,
200 000 steps, greedy evaluation of 10 episodes at 5k ... 200k). No code is
changed; the arms are those of ``experiments.py`` and ``experiments_alpha.py``:

* ``md-exact-diag-bern`` -- exact means, diagonal, Bernoulli endings (the candidate).
* ``md-exact-diag`` -- exact means, diagonal, rounded endings.
* ``md-fixed-diag`` -- EMA alpha 0.9 with the fixed correction, rounded endings
  (the buffer of the main run).
* ``fifo-310`` -- real FIFO buffer of the same memory.

Claims (whole-run score = mean greedy return over the 8 checkpoints; late =
mean from 100k; paired by seed; 95% percentile bootstrap over the 20 seeds):

1. ``md-exact-diag-bern`` beats ``md-fixed-diag`` on the whole-run score.
2. ``md-exact-diag-bern`` beats ``md-exact-diag`` on the whole-run AND the late score.
3. ``md-exact-diag-bern`` does not trail ``fifo-310`` late: the interval of
   the late-score difference is not entirely below 0.
4. Replication of the earlier runs: ``md-fixed-diag`` trails ``fifo-310`` on
   the late score (interval entirely below 0).
5. No direction predicted, reported: ``md-exact-diag-bern - fifo-310`` on the
   whole-run score and at every checkpoint; generated ending rates.

Run:     sh run_replication.sh 26
Summary: python experiments_replication.py --summarize
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import experiments as E
import experiments_alpha  # noqa: F401  (registers the Bernoulli-ending arms in E.ARMS)

SEEDS = tuple(range(120, 140))
OUT = Path(__file__).resolve().parent / "out" / "replication_200k"
STUDY = ["md-exact-diag-bern", "md-exact-diag", "md-fixed-diag", "fifo-310"]
LABEL = {"md-exact-diag-bern": "exact, Bernoulli end", "md-exact-diag": "exact, rounded end", "md-fixed-diag": "alpha 0.9, rounded end",
         "fifo-310": "FIFO 310"}


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
    print("== mean greedy return per checkpoint (seeds 120-139), whole = mean over checkpoints, late = mean from 100k")
    print(f"{'arm':24s} n  " + " ".join(f"{int(c) // 1000:>5}k" for c in cols) + "  whole   late  seedSD  end-rate")
    for arm in STUDY:
        if arm not in data:
            continue
        runs = list(data[arm].values())
        whole = [E.score(r, "curve") for r in runs]
        end = np.mean([r["diag"]["done_rate"] for r in runs]) if "done_rate" in runs[0]["diag"] else float("nan")
        print(f"{LABEL[arm]:24s} {len(runs):2d} " + " ".join(f"{np.mean([E.score(r, c) for r in runs]):6.1f}" for c in cols)
              + f" {np.mean(whole):6.1f} {np.mean([E.score(r, 'late') for r in runs]):6.1f} {np.std(whole, ddof=1):6.1f}  {end:.4f}")
    pairs = [("md-exact-diag-bern", "md-fixed-diag", "claim 1"), ("md-exact-diag-bern", "md-exact-diag", "claim 2"),
             ("md-exact-diag-bern", "fifo-310", "claim 3/5"), ("md-fixed-diag", "fifo-310", "claim 4"), ("md-exact-diag", "md-fixed-diag", "5"),
             ("md-exact-diag", "fifo-310", "5")]
    for kind in ("curve", "late", *cols):
        print(f"\n== paired differences, score = {kind}")
        for a, b, tag in pairs:
            s = E.paired(data, a, b, kind, rng)
            if s:
                print(f"[{tag:9s}] {LABEL[a]} - {LABEL[b]}: {s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}], won {s['won']}/{s['n']}")


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
