"""Bounded buffers of real transitions, the baselines of the SOM buffers.

Ported from RLSOM's ``analysis/online_compare.py`` (``RealBuffer``), with the
``add``/``sample`` interface of plainRL-SOM's buffers.
"""

from __future__ import annotations

import numpy as np


class RealBuffer:
    """A bounded buffer of real transitions stored as rows ``[s, a, r, s', done]``."""

    def __init__(self, kind: str, capacity: int, state_dim: int, seed: int | None = None) -> None:
        """Allocate an empty buffer.

        Args:
            kind: ``"fifo"`` (most recent), ``"reservoir"`` (uniform sample of
                everything seen) or ``"keep-endings"`` (every ending, then a
                reservoir sample of the rest).
            capacity: Transitions it can hold.
            state_dim: Number of state components.
            seed: Seed of the reservoir draws and of the minibatches.

        Raises:
            ValueError: If ``kind`` is not recognised.
        """
        if kind not in ("fifo", "reservoir", "keep-endings"):
            raise ValueError(f"unknown buffer kind {kind!r}")
        self.kind, self.cap, self.d = kind, capacity, state_dim
        self.rows = np.zeros((capacity, 2 * state_dim + 3))
        self.size, self.next, self.seen = 0, 0, 0
        self.rng = np.random.default_rng(seed)
        self.ends: list[int] = []
        self.others: list[int] = []
        self.seen_ends = self.seen_others = 0

    def memory_floats(self) -> int:
        """Numbers the buffer can store.

        Returns:
            Capacity times numbers per transition.
        """
        return self.cap * (2 * self.d + 3)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, step: int = 1) -> None:
        """Offer one transition.

        Args:
            state: State.
            action: Action.
            reward: Reward.
            next_state: Next state.
            done: True only if the episode really ended.
            step: Unused; keeps the SOM buffers' signature.

        Returns:
            None. Stores, replaces or drops the transition by the buffer's rule.
        """
        row = np.concatenate([np.asarray(state, dtype=float), [float(action), float(reward)], np.asarray(next_state, dtype=float), [float(done)]])
        self.seen += 1
        if self.kind == "fifo":
            self.rows[self.next] = row
            self.next = (self.next + 1) % self.cap
            self.size = min(self.size + 1, self.cap)
        elif self.kind == "reservoir":
            if self.size < self.cap:
                self.rows[self.size] = row
                self.size += 1
            else:
                j = int(self.rng.integers(self.seen))
                if j < self.cap:
                    self.rows[j] = row
        else:
            self._add_keep_endings(row, bool(done))

    def _add_keep_endings(self, row: np.ndarray, done: bool) -> None:
        """Offer one transition to the keep-endings buffer.

        Args:
            row: The transition as one row.
            done: Its ending flag.

        Returns:
            None. Endings replace other transitions once the buffer is full;
            when only endings are left they are reservoir-sampled.
        """
        full = len(self.ends) + len(self.others) >= self.cap
        if done:
            self.seen_ends += 1
            if not full:
                slot = len(self.ends) + len(self.others)
                self.ends.append(slot)
            elif self.others:
                slot = self.others.pop(int(self.rng.integers(len(self.others))))
                self.ends.append(slot)
            elif self.rng.random() < self.cap / self.seen_ends:
                slot = self.ends[int(self.rng.integers(len(self.ends)))]
            else:
                return
        else:
            self.seen_others += 1
            if not full:
                slot = len(self.ends) + len(self.others)
                self.others.append(slot)
            else:
                j = int(self.rng.integers(self.seen_others))
                if j >= len(self.others):
                    return
                slot = self.others[j]
        self.rows[slot] = row
        self.size = len(self.ends) + len(self.others)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform minibatch, with replacement, of the stored transitions.

        Args:
            batch_size: Number of transitions.

        Returns:
            ``(states, actions, rewards, next_states, dones)`` arrays.
        """
        b = self.rows[self.rng.integers(0, self.size, size=batch_size)]
        d = self.d
        return b[:, :d], b[:, d].astype(np.int64), b[:, d + 1], b[:, d + 2 : 2 * d + 2], b[:, 2 * d + 2].astype(bool)
