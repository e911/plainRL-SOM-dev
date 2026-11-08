"""Why does the exact-mean + Bernoulli SOM start slowly? Diagnostic on development seeds only.

Runs the agent loop of ``DQNAgent.train_steps`` (same order of operations)
for a short stream and, at a few steps, measures the generated data:

* physics error: mean absolute difference between the generated ``s'`` and
  CartPole's true next state from the generated ``(s, a)``;
* low-count share: fraction of generated transitions drawn from units with
  fewer than 5 / 20 hits;
* drift: mean distance (standardised space, state channels) between a unit's
  current weight and its stored mean, i.e. how far the unit has moved away
  from the transitions it averaged;
* ending rate of the generated data and of the recent real data;
* greedy return (10 episodes).

Usage (from the copy's root):
    python dev_checks/early_diagnostic.py > dev_checks/out_early_diagnostic.txt
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments as E  # noqa: E402
import experiments_alpha  # noqa: E402,F401  (registers Bernoulli arms)
from agent import DQNAgent  # noqa: E402
from args import Args  # noqa: E402

CHECK = (1_000, 2_000, 5_000, 10_000, 25_000)


def physics_error(env: Any, s: np.ndarray, a: np.ndarray, s2: np.ndarray, n: int = 300) -> float:
    """Mean absolute error of generated next states against the true dynamics.

    Args:
        env: Unwrapped CartPole environment used as a simulator.
        s: Generated states.
        a: Generated actions.
        s2: Generated next states.
        n: Number of transitions checked.

    Returns:
        Mean absolute error over components and transitions.
    """
    errs = []
    for si, ai, s2i in zip(s[:n], a[:n], s2[:n]):
        env.state = np.array(si, dtype=np.float64)
        errs.append(np.abs(env.step(int(ai))[0] - s2i).mean())
    return float(np.mean(errs))


def diagnose(rb: Any, sim: Any, recent_done: list[float]) -> dict[str, float]:
    """Measure one buffer's generated data.

    Args:
        rb: A StructuredSOMBuffer.
        sim: Unwrapped CartPole simulator.
        recent_done: Ending flags of the last real transitions.

    Returns:
        Dict of diagnostics.
    """
    s, a, r, s2, d = rb.sample(2000)
    hit = rb.count > 0
    counts = rb.count[hit]
    # share of generated samples (uniform over hit units) from low-count units
    low5, low20 = float(np.mean(counts < 5)), float(np.mean(counts < 20))
    ix, iy = np.nonzero(hit)
    mean, _ = rb.unit_stats(ix, iy)
    z = np.array([rb._standardise(m) for m in mean])
    drift = float(np.mean(np.linalg.norm(z[:, : rb.d] - rb.weights[ix, iy, : rb.d], axis=1)))
    return {"phys": physics_error(sim, s, a, s2), "low5": low5, "low20": low20, "drift": drift, "gen_end": float(np.mean(d)),
            "real_end": float(np.mean(recent_done)) if recent_done else float("nan"), "units": int(hit.sum())}


def run(arm: str, seed: int, steps: int = 25_000) -> None:
    """Train one development run and print diagnostics at the check steps.

    Args:
        arm: Arm name in ``experiments.ARMS``.
        seed: Development seed.
        steps: Environment steps.

    Returns:
        None. Prints one line per check step.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    args = Args()
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    sim = gym.make("CartPole-v1").unwrapped
    sim.reset(seed=0)
    agent = DQNAgent(4, 2, args=args)
    rb = E.ARMS[arm](seed)
    state, _ = env.reset(seed=seed)
    e, ep_r, is_train, recent = 0, 0.0, False, []
    for t in range(1, steps + 1):
        action = agent.select_action(state)
        nxt, reward, term, trunc, _ = env.step(action)
        ep_r += reward
        rb.add(state, action, reward, nxt, term, e)
        recent = (recent + [float(term)])[-2000:]
        state = nxt
        if is_train:
            agent.update(*rb.sample(args.batch_size))
        if t % args.update_target_freq == 0:
            agent.update_target_network()
        if term or trunc:
            agent.decay_epsilon()
            e += 1
            is_train = t > 150
            state, _ = env.reset()
            ep_r = 0.0
        if t in CHECK:
            dg = diagnose(rb, sim, recent)
            ret = agent.evaluate(eval_env, 10, seed=seed * 1000 + t)
            print(f"{arm:20s} seed {seed} step {t:6d} ep {e:4d} | return {ret:6.1f} | phys err {dg['phys']:.4f} | units {dg['units']:3d} "
                  f"| units <5 hits {dg['low5']:.2f} <20 hits {dg['low20']:.2f} | drift {dg['drift']:.3f} | end gen {dg['gen_end']:.3f} "
                  f"real {dg['real_end']:.3f}", flush=True)


if __name__ == "__main__":
    for seed in (997, 998, 999):
        for arm in ("md-exact-diag-bern", "md-fixed-diag", "md-exact-diag"):
            run(arm, seed)
