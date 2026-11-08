"""Online CartPole: plainRL-SOM's single-map SOM buffer, fixed, against real replay buffers.

Every run trains plainRL-SOM's DQN agent (networks.DQN 64-64, Adam 1e-3, MSE,
batch 64, gamma 0.99, target sync every 500 steps, learning from step 150,
epsilon 1 -> 0.01 decaying x0.995 per episode) online on CartPole-v1 for
200 000 environment steps with ONE replay memory, and evaluates the greedy
policy (10 episodes) after 5k, 10k, 25k, 50k, 75k, 100k, 150k and 200k steps.
In every arm only real endings are stored as terminal (the 500-step time
limit is not), see change.md.

Arms (SOM grid 10 x 10 in all SOM arms):

* ``orig`` -- plainRL-SOM's ``WeightedSOMBuffer`` as committed (3e23ea1):
  hand weights on raw [s, a, r, s', done], EMA alpha 0.9 with the broken
  bias correction, diagonal Gaussians, units chosen uniformly.
* ``md-<correction>-<cov>`` -- :class:`structured_som.StructuredSOMBuffer`:
  the distance of ``distancemetric_simple.md`` on [s, a, r, delta_s, done]
  with running standardisation; ``correction`` is ``buggy`` / ``fixed`` /
  ``none`` (EMA alpha 0.9) or ``exact`` (exact running mean and covariance);
  ``cov`` is ``diag`` or ``full``. ``md-fixed-diag-count`` chooses units in
  proportion to their hits instead of uniformly.
* ``fifo-k`` / ``reservoir-k`` / ``keep-endings-k`` -- real transitions,
  ``k`` of them. k = 310 matches the diagonal SOMs' memory (3 418 numbers,
  11 per transition; ``orig`` stores 3 400), k = 810 the full-covariance
  SOMs' (8 918). ``fifo-10000`` is the usual large buffer (reference).

Scores. Primary: mean greedy return over the 8 checkpoints (area under the
learning curve). Also reported: each checkpoint, and the mean over the
checkpoints from 100k on (late). Pairs are formed by seed (seeds 100-109;
every arm of a seed starts from the same network weights and environment
seed); intervals are 95% percentile bootstraps over seeds.

Claims, stated before the run (all reported whichever way they fall):

1. Bias fix: ``md-fixed-diag`` beats ``md-buggy-diag`` on the primary score.
2. No correction: ``md-none-diag`` and ``md-fixed-diag`` do not differ (the
   interval of the difference includes 0), since at alpha 0.9 the
   uncorrected bias is gone after two hits of a unit.
3. Against real data of the same memory: ``md-fixed-diag`` LOSES to
   ``fifo-310`` on the primary score (a development check with the committed
   metric found 111 vs 221 in the last 100 of 500 episodes).
4. No direction predicted, reported: exact vs EMA averaging (``md-exact-diag``
   vs ``md-fixed-diag``), full vs diagonal covariance (``md-fixed-full`` vs
   ``md-fixed-diag``), the md metric vs the committed one (``md-buggy-diag``
   vs ``orig``), units by count vs uniform (``md-fixed-diag-count`` vs
   ``md-fixed-diag``), the full-covariance SOM vs ``fifo-810``, and every SOM
   arm vs ``reservoir``, ``keep-endings`` and ``fifo-10000``.

Run:   sh run_experiments.sh 26            (all arms x seeds; resumable)
       python experiments.py --arm md-fixed-diag --seed 100
Summary: python experiments.py --summarize
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent
from args import Args
from replay_buffers import RealBuffer
from SOMBuffer import WeightedSOMBuffer
from structured_som import StructuredSOMBuffer

STEPS = 200_000
CHECKPOINTS = (5_000, 10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000)
SEEDS = tuple(range(100, 110))
OUT = Path(__file__).resolve().parent / "out" / "online_200k"


def _md(correction: str, covariance: str, unit_choice: str = "uniform") -> Callable[[int], Any]:
    """Factory of a structured SOM buffer arm.

    Args:
        correction: ``"buggy"``, ``"fixed"``, ``"none"`` or ``"exact"``.
        covariance: ``"diag"`` or ``"full"``.
        unit_choice: ``"uniform"`` or ``"count"``.

    Returns:
        A function from seed to a new buffer.
    """
    averaging = "exact" if correction == "exact" else "ema"
    corr = "fixed" if correction == "exact" else correction

    def make(seed: int) -> StructuredSOMBuffer:
        """Build the buffer.

        Args:
            seed: Seed of the buffer.

        Returns:
            The buffer.
        """
        return StructuredSOMBuffer(averaging=averaging, correction=corr, covariance=covariance, unit_choice=unit_choice, seed=seed)

    return make


ARMS: dict[str, Callable[[int], Any]] = {
    "orig": lambda seed: WeightedSOMBuffer(),
    "md-buggy-diag": _md("buggy", "diag"),
    "md-fixed-diag": _md("fixed", "diag"),
    "md-none-diag": _md("none", "diag"),
    "md-exact-diag": _md("exact", "diag"),
    "md-fixed-full": _md("fixed", "full"),
    "md-none-full": _md("none", "full"),
    "md-exact-full": _md("exact", "full"),
    "md-fixed-diag-count": _md("fixed", "diag", "count"),
}
for _k in (310, 810):
    for _kind in ("fifo", "reservoir", "keep-endings"):
        ARMS[f"{_kind}-{_k}"] = (lambda kind, k: (lambda seed: RealBuffer(kind, k, 4, seed)))(_kind, _k)
ARMS["fifo-10000"] = lambda seed: RealBuffer("fifo", 10_000, 4, seed)


def run(arm: str, seed: int, steps: int = STEPS, checkpoints: tuple[int, ...] = CHECKPOINTS, out: Path = OUT) -> dict[str, Any]:
    """Train one agent with one buffer and save its scores.

    Args:
        arm: Key of :data:`ARMS`.
        seed: Seed of python, numpy and torch, the environments and the buffer.
        steps: Environment steps.
        checkpoints: Steps after which to evaluate.
        out: Directory of the result files.

    Returns:
        The saved result (arm, seed, evaluations, training returns, memory, time).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    args = Args()
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    agent = DQNAgent(4, 2, args=args)
    rb = ARMS[arm](seed)
    t0 = time.time()
    res = agent.train_steps(env, eval_env, rb, steps, checkpoints, args.batch_size, args.update_target_freq, seed=seed)
    mem = rb.memory_floats() if hasattr(rb, "memory_floats") else 100 * (3 * 11 + 1)
    diag: dict[str, Any] = {"memory_floats": int(mem)}
    if hasattr(rb, "count") or hasattr(rb, "node_ct"):
        s, a, r, s2, d = rb.sample(5000)
        counts = rb.count if hasattr(rb, "count") else rb.node_ct
        diag.update(units_hit=int((counts > 0).sum()), done_rate=float(np.mean(d)), reward_mean=float(np.mean(r)), reward_min=float(np.min(r)),
                    reward_max=float(np.max(r)), action1_rate=float(np.mean(a)))
    result = {"arm": arm, "seed": seed, "eval": {str(k): v for k, v in res["eval"].items()}, "episodes": res["episodes"],
              "train_returns": res["train_returns"], "diag": diag, "seconds": time.time() - t0}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{arm}_{seed}.json").write_text(json.dumps(result))
    ev = res["eval"]
    print(f"{arm} seed {seed}: " + " ".join(f"{k // 1000}k={ev[k]:.0f}" for k in sorted(ev)) + f" | curve {np.mean(list(ev.values())):.1f} | "
          f"{time.time() - t0:.0f} s", flush=True)
    return result


def load(out: Path = OUT) -> dict[str, dict[int, dict[str, Any]]]:
    """Read every saved result.

    Args:
        out: Directory of the result files.

    Returns:
        ``{arm: {seed: result}}``.
    """
    data: dict[str, dict[int, dict[str, Any]]] = {}
    for p in sorted(out.glob("*.json")):
        r = json.loads(p.read_text())
        data.setdefault(r["arm"], {})[r["seed"]] = r
    return data


def score(r: dict[str, Any], kind: str) -> float:
    """One run's score.

    Args:
        r: A saved result.
        kind: ``"curve"`` (all checkpoints), ``"late"`` (from 100k on) or a checkpoint step as text.

    Returns:
        The score.
    """
    ev = {int(k): v for k, v in r["eval"].items()}
    if kind == "curve":
        return float(np.mean(list(ev.values())))
    if kind == "late":
        return float(np.mean([v for k, v in ev.items() if k >= 100_000]))
    return float(ev[int(kind)])


def paired(data: dict[str, dict[int, dict[str, Any]]], first: str, second: str, kind: str, rng: np.random.Generator,
           draws: int = 10_000) -> dict[str, float] | None:
    """Paired difference ``first - second`` over common seeds.

    Args:
        data: Results by arm and seed.
        first: First arm.
        second: Second arm.
        kind: Score kind (see :func:`score`).
        rng: Generator of the bootstrap.
        draws: Bootstrap draws.

    Returns:
        ``n``, ``mean``, ``lo``, ``hi`` (95% percentile bootstrap over seeds)
        and ``won``; ``None`` if the arms share no seed.
    """
    seeds = sorted(set(data.get(first, {})) & set(data.get(second, {})))
    if not seeds:
        return None
    diff = np.array([score(data[first][s], kind) - score(data[second][s], kind) for s in seeds])
    boot = diff[rng.integers(len(diff), size=(draws, len(diff)))].mean(1)
    return {"n": len(diff), "mean": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "won": int((diff > 0).sum())}


def summarize(out: Path = OUT) -> None:
    """Print mean scores per arm and checkpoint, and the claims' paired differences.

    Args:
        out: Directory of the result files.

    Returns:
        None. Prints the summary.
    """
    data = load(out)
    rng = np.random.default_rng(0)
    cols = [str(c) for c in CHECKPOINTS]
    print("== mean greedy return per checkpoint (seeds), curve = mean over checkpoints, late = mean from 100k")
    print(f"{'arm':22s} n  " + " ".join(f"{int(c) // 1000:>5}k" for c in cols) + "  curve   late  memory")
    for arm in ARMS:
        if arm not in data:
            continue
        runs = list(data[arm].values())
        row = " ".join(f"{np.mean([score(r, c) for r in runs]):6.1f}" for c in cols)
        print(f"{arm:22s} {len(runs):2d} {row} {np.mean([score(r, 'curve') for r in runs]):6.1f} {np.mean([score(r, 'late') for r in runs]):6.1f}  "
              f"{runs[0]['diag']['memory_floats']}")
    print("\n== SOM sample diagnostics at 200k (mean over seeds)")
    for arm in ARMS:
        runs = [r for r in data.get(arm, {}).values() if "done_rate" in r["diag"]]
        if runs:
            m = {k: np.mean([r["diag"][k] for r in runs]) for k in ("units_hit", "done_rate", "reward_mean", "reward_min", "reward_max", "action1_rate")}
            print(f"{arm:22s} units {m['units_hit']:.0f}, done rate {m['done_rate']:.3f}, reward mean {m['reward_mean']:.2f} "
                  f"[{m['reward_min']:.2f}, {m['reward_max']:.2f}], action-1 rate {m['action1_rate']:.2f}")
    pairs = [("md-fixed-diag", "md-buggy-diag", "claim 1"), ("md-none-diag", "md-fixed-diag", "claim 2"), ("md-fixed-diag", "fifo-310", "claim 3"),
             ("md-exact-diag", "md-fixed-diag", "4"), ("md-fixed-full", "md-fixed-diag", "4"), ("md-none-full", "md-fixed-full", "4"),
             ("md-exact-full", "md-fixed-full", "4"), ("md-buggy-diag", "orig", "4"), ("md-fixed-diag-count", "md-fixed-diag", "4"),
             ("md-fixed-full", "fifo-810", "4")]
    for som in ("orig", "md-fixed-diag", "md-exact-diag", "md-fixed-full", "md-exact-full"):
        k = 810 if som.endswith("full") else 310
        pairs += [(som, f"{kind}-{k}", "4") for kind in ("fifo", "reservoir", "keep-endings") if (som, f"{kind}-{k}", "claim 3") not in pairs]
        pairs.append((som, "fifo-10000", "4"))
    seen = set()
    for kind in ("curve", "late", *cols):
        print(f"\n== paired differences, score = {kind}")
        for a, b, tag in pairs:
            if (a, b, kind) in seen:
                continue
            seen.add((a, b, kind))
            s = paired(data, a, b, kind, rng)
            if s:
                print(f"[{tag:7s}] {a} - {b}: {s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}], won {s['won']}/{s['n']}")


def main() -> None:
    """Command line: run arms and seeds, or summarise.

    Returns:
        None.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize(a.out)
        return
    cps = tuple(c for c in CHECKPOINTS if c <= a.steps) or (a.steps,)
    for seed in a.seed:
        for arm in a.arm:
            if (a.out / f"{arm}_{seed}.json").exists():
                continue
            run(arm, seed, a.steps, cps, a.out)


if __name__ == "__main__":
    main()
