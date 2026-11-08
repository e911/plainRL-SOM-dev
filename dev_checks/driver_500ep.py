"""Scratchpad check of plainRL-SOM's WeightedSOMBuffer (copy; repo untouched).

Arms: orig (code as committed), fixbias (bias correction uses (1-alpha)^t),
fifo10k (real FIFO buffer, 10k), fifo309 (real FIFO at the SOM's float budget).
Runs the repo's DQNAgent.train loop logic with a seed and records returns and
diagnostics of the sampled minibatches.
"""

from __future__ import annotations

import json
import random
import sys
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, ".")
from agent import DQNAgent  # noqa: E402
from args import Args  # noqa: E402
from SOMBuffer import WeightedSOMBuffer  # noqa: E402


class FixedBiasSOM(WeightedSOMBuffer):
    """WeightedSOMBuffer whose sample() divides by 1-(1-alpha)^t instead of 1-alpha^t."""

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Draw a minibatch with the corrected bias correction.

        Args:
            batch_size: Number of transitions.

        Returns:
            states, actions, rewards, next_states, dones arrays.
        """
        valid = np.argwhere(self.node_ct > 0)
        idx = valid[np.random.choice(len(valid), size=batch_size)]
        ix, iy = idx[:, 0], idx[:, 1]
        c = self.node_ct[ix, iy][:, None]
        corr = 1.0 - np.power(1.0 - self.alpha, c)
        means = self.node_means[ix, iy] / corr
        var = self.node_vars[ix, iy] / corr
        s = np.random.normal(means, np.sqrt(np.maximum(var, 1e-5)) + 1e-5)
        sd = self.state_dim
        return (s[:, :sd], np.clip(np.round(s[:, sd]), 0, 1).astype(np.int32), s[:, sd + 1],
                s[:, sd + 2:2 * sd + 2], np.round(np.clip(s[:, -1], 0, 1)).astype(bool))


class Fifo:
    """Real FIFO replay buffer of fixed capacity."""

    def __init__(self, cap: int) -> None:
        """Create the buffer.

        Args:
            cap: Capacity in transitions.
        """
        self.d: deque[tuple[Any, ...]] = deque(maxlen=cap)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool, step: int = 1) -> None:
        """Append one transition.

        Args:
            s: State.
            a: Action.
            r: Reward.
            s2: Next state.
            done: Done flag.
            step: Unused (API match).
        """
        self.d.append((s, a, r, s2, done))

    def sample(self, n: int) -> tuple[np.ndarray, ...]:
        """Uniform minibatch with replacement.

        Args:
            n: Batch size.

        Returns:
            states, actions, rewards, next_states, dones arrays.
        """
        b = [self.d[i] for i in np.random.randint(len(self.d), size=n)]
        return tuple(np.array(x) for x in zip(*b))


def run(arm: str, seed: int, n_episodes: int) -> dict[str, Any]:
    """Run one online DQN training with the repo's loop.

    Args:
        arm: Buffer arm name.
        seed: Seed for python, numpy, torch and env.
        n_episodes: Episodes to train.

    Returns:
        Dict with returns and minibatch diagnostics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    args = Args(n_episodes=n_episodes)
    env = gym.make("CartPole-v1")
    ag = DQNAgent(4, 2, args=args)
    rb = {"orig": WeightedSOMBuffer, "fixbias": FixedBiasSOM}.get(arm)
    rb = rb() if rb else Fifo(10000 if arm == "fifo10k" else 309)
    rets, diag = [], []
    total = 0
    for e in range(n_episodes):
        is_train = total > 150
        s, _ = env.reset(seed=seed * 100000 + e)
        done, ep = False, 0.0
        while not done:
            total += 1
            a = ag.select_action(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep += r
            rb.add(s, a, r, s2, done, e)
            s = s2
            if is_train:
                b = rb.sample(args.batch_size)
                ag.update(*b)
                if total % 1000 == 0:
                    diag.append({"step": total, "r_mean": float(np.mean(b[2])), "r_max": float(np.max(b[2])),
                                 "r_min": float(np.min(b[2])), "done_frac": float(np.mean(b[4])),
                                 "act1_frac": float(np.mean(b[1])),
                                 "lowcount_units": float(np.mean(rb.node_ct[rb.node_ct > 0] < 5)) if hasattr(rb, "node_ct") else None})
            if total % args.update_target_freq == 0:
                ag.update_target_network()
        ag.decay_epsilon()
        rets.append(ep)
    return {"arm": arm, "seed": seed, "returns": rets, "steps": total, "diag": diag}


if __name__ == "__main__":
    arm, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = run(arm, seed, n)
    with open(f"out_{arm}_{seed}.json", "w") as f:
        json.dump(out, f)
    r = out["returns"]
    print(arm, seed, "mean last100", np.mean(r[-100:]), "max", max(r), "steps", out["steps"])
