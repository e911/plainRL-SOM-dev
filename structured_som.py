"""Single-map SOM replay buffer with the structured distance of ``distancemetric_simple.md``.

One SOM learns online from the agent's stream of transitions and is the
agent's only replay memory. Each transition is the 11-number vector (for
CartPole) ``x = [s, a, r, delta_s, done]`` with ``delta_s = s' - s``.

Best-matching unit (BMU). The distance works on a standardised copy of the
vector: ``s``, ``r`` and ``delta_s`` are z-scored with a running mean and
standard deviation that is updated at every step (the whole stream is never
available online); ``a`` and ``done`` stay 0/1 and are compared by mismatch:

    D = alpha_s * sum_i beta_i (s_i - w_s,i)^2
      + alpha_a * 1[a != round(w_a)]
      + alpha_r * (r - w_r)^2
      + alpha_delta * sum_i gamma_i (delta_s_i - w_delta,i)^2
      + alpha_done * 1[done != round(w_done)]

The defaults are those of ``distancemetric_simple.md`` (``train_simple.py``):
``alpha_s = 1``, ``alpha_a = 0.5``, ``alpha_r = 0``, ``alpha_delta = 1``,
``alpha_done = 1``; for CartPole ``beta = [0.5, 0.5, 2.0, 1.5]`` and
``gamma = [0, 0.5, 0, 1.5]`` (``delta_x`` and ``delta_theta`` are exact copies
of the velocities after z-scoring, so they get no weight); other tasks get
weights of 1.

Map updates follow plainRL-SOM's ``WeightedSOMBuffer``: the BMU and its grid
neighbours move towards the standardised vector with step size
``lr * exp(-episode / 180)`` and neighbourhood width
``max(sigma * exp(-episode / 180), 0.4)``.

Per-unit statistics. Every BMU keeps a mean and a spread of the RAW vectors it
receives (raw, so that the drifting standardisation does not make them stale):

* ``averaging="ema"`` -- plainRL-SOM's exponential average with weight
  ``alpha`` on the new vector: ``m <- (1 - alpha) m + alpha x`` and
  ``V <- (1 - alpha) V + alpha (m_old - x)(m_old - x)^T`` (diagonal or full).
  Read with one of three corrections for the start at zero:
  ``"buggy"`` divides by ``1 - alpha^t`` (plainRL-SOM's line, 9x too large
  after one hit at ``alpha = 0.9``), ``"fixed"`` divides by
  ``1 - (1 - alpha)^t`` (the correct factor for this update), ``"none"``
  does not correct.
* ``averaging="exact"`` -- the exact running mean and (population) covariance
  of everything the unit received (Welford); no correction is needed.

Generation draws independent transitions: pick a unit (uniformly among the
units hit so far, as plainRL-SOM, or in proportion to their hit counts, as
the md file), draw ``x ~ N(mean, cov)`` (diagonal or full covariance) and
decode ``a = clip(round(x_a))``, ``done = round(clip(x_done, 0, 1))``,
``s' = s + delta_s``. With ``done_decode="bernoulli"`` the ending flag is
instead drawn as ``Bernoulli(p)`` with ``p`` the unit's mean ending value
(added for the alpha / ending-rate study, ``experiments_alpha.py``; the
default ``"round"`` leaves every earlier result unchanged). For the burn-in
study (``experiments_burnin.py``): ``min_hits`` lets only units with enough
hits generate, and ``averaging="hybrid"`` is exact for a unit's first
``1 / kappa_min`` hits and then forgets slowly. Both defaults leave earlier
results unchanged. For the averaging-rate study (``experiments_rate.py``):
``kappa_until`` limits the hybrid floor rate to the first transitions of the
stream ("fast early, exact later"); its default ``None`` leaves earlier
results unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Floor added to every variance before sampling (plainRL-SOM's value).
VAR_FLOOR = 1e-5


class StructuredSOMBuffer:
    """Single-map SOM over ``[s, a, r, delta_s, done]`` used as a generative replay buffer."""

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 2,
        grid_size: tuple[int, int] = (10, 10),
        lr: float = 0.1,
        sigma: float = 3.0,
        alpha: float = 0.9,
        averaging: str = "ema",
        correction: str = "fixed",
        covariance: str = "diag",
        unit_choice: str = "uniform",
        alpha_s: float = 1.0,
        alpha_a: float = 0.5,
        alpha_r: float = 0.0,
        alpha_delta: float = 1.0,
        alpha_done: float = 1.0,
        state_weights: Sequence[float] | None = None,
        delta_weights: Sequence[float] | None = None,
        seed: int | None = None,
        done_decode: str = "round",
        min_hits: int = 1,
        kappa_min: float = 0.01,
        kappa_until: int | None = None,
    ) -> None:
        """Create an empty map.

        Args:
            state_dim: Number of state components.
            n_actions: Number of discrete actions.
            grid_size: Rows and columns of the map.
            lr: Initial step size of the map update.
            sigma: Initial neighbourhood width (in grid cells).
            alpha: Weight of the new vector in the exponential average.
            averaging: ``"ema"``, ``"exact"`` or ``"hybrid"`` per-unit
                statistics. ``"hybrid"`` averages with rate
                ``max(1 / n, kappa_min)``: the exact mean and variance for a
                unit's first ``1 / kappa_min`` hits, then an exponential
                average that slowly forgets (no zero start, no correction).
            correction: ``"buggy"``, ``"fixed"`` or ``"none"`` (only used with ``"ema"``).
            covariance: ``"diag"`` or ``"full"`` per-unit spread.
            unit_choice: ``"uniform"`` over hit units or ``"count"`` (in proportion to hits).
            alpha_s: Weight of the state term.
            alpha_a: Weight of the action-mismatch term.
            alpha_r: Weight of the reward term.
            alpha_delta: Weight of the state-change term.
            alpha_done: Weight of the done-mismatch term.
            state_weights: Per-component state weights; ``None`` uses the md
                file's CartPole weights for 4 components and ones otherwise.
            delta_weights: Per-component state-change weights; ``None`` as above.
            seed: Seed of the initial weights and of the sampling.
            done_decode: How a generated ending flag is decoded: ``"round"``
                rounds the Gaussian draw (the default, as before);
                ``"bernoulli"`` draws ``done ~ Bernoulli(p)`` with ``p`` the
                unit's (corrected) mean ending value, clipped to [0, 1].
            min_hits: Burn-in: only units with at least this many hits generate
                (default 1 = every unit hit so far). If no unit has enough hits
                yet, every hit unit generates.
            kappa_min: Smallest averaging rate of ``"hybrid"`` averaging.
            kappa_until: With ``"hybrid"`` averaging, apply ``kappa_min`` only
                while the memory has received fewer than this many transitions
                in total; afterwards the rate is ``1 / n`` (exact averaging
                from then on). ``None`` (default) applies it throughout.

        Raises:
            ValueError: If an option has an unknown value.
        """
        for name, value, allowed in (
            ("averaging", averaging, ("ema", "exact", "hybrid")),
            ("correction", correction, ("buggy", "fixed", "none")),
            ("covariance", covariance, ("diag", "full")),
            ("unit_choice", unit_choice, ("uniform", "count")),
            ("done_decode", done_decode, ("round", "bernoulli")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        self.d, self.n_actions = state_dim, n_actions
        self.dim = 2 * state_dim + 3
        self.grid_size = grid_size
        self.initial_lr, self.initial_sigma, self.alpha = lr, sigma, alpha
        self.averaging, self.correction, self.covariance, self.unit_choice = averaging, correction, covariance, unit_choice
        self.done_decode = done_decode
        self.min_hits, self.kappa_min = max(int(min_hits), 1), kappa_min
        self.kappa_until = kappa_until
        self.rng = np.random.default_rng(seed)
        if state_weights is None:
            state_weights = [0.5, 0.5, 2.0, 1.5] if state_dim == 4 else [1.0] * state_dim
        if delta_weights is None:
            delta_weights = [0.0, 0.5, 0.0, 1.5] if state_dim == 4 else [1.0] * state_dim
        d = state_dim
        # Squared-difference weights of the continuous channels, laid out like the vector.
        self.cont_w = np.zeros(self.dim)
        self.cont_w[:d] = alpha_s * np.asarray(state_weights, dtype=float)
        self.cont_w[d + 1] = alpha_r
        self.cont_w[d + 2 : 2 * d + 2] = alpha_delta * np.asarray(delta_weights, dtype=float)
        self.alpha_a, self.alpha_done = alpha_a, alpha_done
        self.ia, self.idone = d, 2 * d + 2
        # Channels that are z-scored: s, r, delta_s.
        self.cont = np.ones(self.dim, dtype=bool)
        self.cont[[self.ia, self.idone]] = False
        self.weights = self.rng.uniform(-0.1, 0.1, (grid_size[0], grid_size[1], self.dim))
        self.count = np.zeros(grid_size, dtype=np.int64)
        self.mean = np.zeros((grid_size[0], grid_size[1], self.dim))
        shape = (grid_size[0], grid_size[1], self.dim) + ((self.dim,) if covariance == "full" else ())
        self.spread = np.zeros(shape)
        self.norm_n = 0
        self.norm_mean = np.zeros(self.dim)
        self.norm_m2 = np.zeros(self.dim)
        x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), indexing="ij")
        self.grid_coords = np.stack([x, y], axis=-1)

    def memory_floats(self) -> int:
        """Numbers the buffer stores.

        Returns:
            Units times (weights + mean + spread + count), plus the running mean
            and second moment of the standardised channels.
        """
        units = self.grid_size[0] * self.grid_size[1]
        spread = self.dim * (self.dim + 1) // 2 if self.covariance == "full" else self.dim
        return units * (self.dim + self.dim + spread + 1) + 2 * int(self.cont.sum())

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        """Z-score the continuous channels with the current running statistics.

        Args:
            x: Raw vector ``[s, a, r, delta_s, done]``.

        Returns:
            The vector with ``s``, ``r`` and ``delta_s`` standardised; ``a`` and
            ``done`` unchanged. Channels with no spread yet are only centred.
        """
        sd = np.sqrt(self.norm_m2 / max(self.norm_n, 1))
        sd = np.where(sd > 1e-8, sd, 1.0)
        z = x.copy()
        z[self.cont] = (x[self.cont] - self.norm_mean[self.cont]) / sd[self.cont]
        return z

    def _update_norm(self, x: np.ndarray) -> None:
        """Add one raw vector to the running mean and second moment (Welford).

        Args:
            x: Raw vector.

        Returns:
            None. Updates ``norm_n``, ``norm_mean`` and ``norm_m2``.
        """
        self.norm_n += 1
        delta = x - self.norm_mean
        self.norm_mean += delta / self.norm_n
        self.norm_m2 += delta * (x - self.norm_mean)

    def _distance(self, z: np.ndarray) -> np.ndarray:
        """Structured distance from a standardised vector to every unit.

        Args:
            z: Standardised vector.

        Returns:
            A ``grid_size`` array of distances.
        """
        cont = np.sum(self.cont_w * (self.weights - z) ** 2, axis=-1)
        wa = np.clip(np.round(self.weights[..., self.ia]), 0, self.n_actions - 1)
        wd = np.clip(np.round(self.weights[..., self.idone]), 0, 1)
        return cont + self.alpha_a * (wa != z[self.ia]) + self.alpha_done * (wd != z[self.idone])

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, step: int = 1) -> None:
        """Learn one transition.

        Args:
            state: State ``s``.
            action: Action taken.
            reward: Reward received.
            next_state: Next state ``s'``.
            done: True only if the episode really ended (not at a time limit).
            step: Episode index, which sets the map's step size and neighbourhood.

        Returns:
            None. Updates the standardisation, moves the BMU and its neighbours,
            and adds the raw vector to the BMU's statistics.
        """
        s = np.asarray(state, dtype=float)
        x = np.concatenate([s, [float(action), float(reward)], np.asarray(next_state, dtype=float) - s, [float(done)]])
        self._update_norm(x)
        z = self._standardise(x)
        bmu = np.unravel_index(np.argmin(self._distance(z)), self.grid_size)
        decay = np.exp(-step / 180.0)
        lr = self.initial_lr * decay
        sigma = max(self.initial_sigma * decay, 0.4)
        dist_sq = np.sum((self.grid_coords - np.array(bmu)) ** 2, axis=-1)
        self.weights += lr * np.exp(-dist_sq / (2 * sigma**2))[..., None] * (z - self.weights)
        self.count[bmu] += 1
        n = self.count[bmu]
        m_old = self.mean[bmu].copy()
        if self.averaging == "exact":
            self.mean[bmu] = m_old + (x - m_old) / n
            if self.covariance == "full":
                self.spread[bmu] += np.outer(x - m_old, x - self.mean[bmu])
            else:
                self.spread[bmu] += (x - m_old) * (x - self.mean[bmu])
        elif self.averaging == "hybrid":
            # Exponentially weighted mean and variance with rate k = max(1/n, kappa_min);
            # for k = 1/n this is exactly the running mean and population variance.
            floor = self.kappa_min if self.kappa_until is None or self.norm_n <= self.kappa_until else 0.0
            k = max(1.0 / n, floor)
            delta = x - m_old
            self.mean[bmu] = m_old + k * delta
            outer = np.outer(delta, delta) if self.covariance == "full" else delta**2
            self.spread[bmu] = (1.0 - k) * (self.spread[bmu] + k * outer)
        else:
            diff = m_old - x
            a = self.alpha
            self.mean[bmu] = (1.0 - a) * m_old + a * x
            outer = np.outer(diff, diff) if self.covariance == "full" else diff**2
            self.spread[bmu] = (1.0 - a) * self.spread[bmu] + a * outer

    def unit_stats(self, ix: np.ndarray, iy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corrected mean and spread of the given units.

        Args:
            ix: Row indices.
            iy: Column indices.

        Returns:
            ``(mean, spread)``: means ``(n, dim)`` and variances ``(n, dim)`` or
            covariances ``(n, dim, dim)``, after the chosen correction.
        """
        counts = self.count[ix, iy].astype(float)
        mean, spread = self.mean[ix, iy], self.spread[ix, iy]
        if self.averaging == "exact":
            return mean, spread / np.maximum(counts, 1.0).reshape((-1,) + (1,) * (spread.ndim - 1))
        if self.averaging == "hybrid":
            return mean, spread
        if self.correction == "buggy":
            factor = 1.0 - np.power(self.alpha, counts)
        elif self.correction == "fixed":
            factor = 1.0 - np.power(1.0 - self.alpha, counts)
        else:
            factor = np.ones_like(counts)
        return mean / factor[:, None], spread / factor.reshape((-1,) + (1,) * (spread.ndim - 1))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate independent transitions.

        Args:
            batch_size: Number of transitions.

        Returns:
            ``(states, actions, rewards, next_states, dones)`` arrays.

        Raises:
            RuntimeError: If no transition has been added yet.
        """
        hit = np.argwhere(self.count > 0)
        if len(hit) == 0:
            raise RuntimeError("sample() called before any add()")
        if self.min_hits > 1:
            ready = hit[self.count[hit[:, 0], hit[:, 1]] >= self.min_hits]
            if len(ready):
                hit = ready
        if self.unit_choice == "count":
            p = self.count[hit[:, 0], hit[:, 1]].astype(float)
            pick = self.rng.choice(len(hit), size=batch_size, p=p / p.sum())
        else:
            pick = self.rng.integers(len(hit), size=batch_size)
        ix, iy = hit[pick, 0], hit[pick, 1]
        mean, spread = self.unit_stats(ix, iy)
        noise = self.rng.standard_normal((batch_size, self.dim))
        if self.covariance == "full":
            cov = 0.5 * (spread + np.swapaxes(spread, 1, 2)) + VAR_FLOOR * np.eye(self.dim)
            vals, vecs = np.linalg.eigh(cov)
            root = vecs * np.sqrt(np.clip(vals, VAR_FLOOR, None))[:, None, :]
            x = mean + np.einsum("bij,bj->bi", root, noise)
        else:
            x = mean + (np.sqrt(np.maximum(spread, VAR_FLOOR)) + 1e-5) * noise
        d = self.d
        states = x[:, :d]
        actions = np.clip(np.round(x[:, self.ia]), 0, self.n_actions - 1).astype(np.int64)
        rewards = x[:, d + 1]
        next_states = states + x[:, d + 2 : 2 * d + 2]
        if self.done_decode == "bernoulli":
            dones = self.rng.random(batch_size) < np.clip(mean[:, self.idone], 0.0, 1.0)
        else:
            dones = np.round(np.clip(x[:, self.idone], 0, 1)).astype(bool)
        return states, actions, rewards, next_states, dones
