# Changes in this copy of plainRL-SOM

This folder is a full copy of `plainRL-SOM` (origin `sheeraja/RL-SOM`, branch
`main`, commit `3e23ea1`), made on 2026-11-04 for experiments. **Nothing here
is committed, and pushing is disabled** (`git remote set-url --push origin
DISABLED-no-push`). The original `plainRL-SOM` folder is untouched.
`git diff` shows the edits to existing files; new files are untracked.

## 1. Bugs found in the original code

### 1.1 Bias correction of the unit statistics (`SOMBuffer.py`, `WeightedSOMBuffer.sample`, lines 165-166)

`add` updates each unit's mean and variance with an exponential average that
starts at zero and puts weight `alpha = 0.9` on the new vector:

    m <- (1 - alpha) m + alpha x

`sample` then divides by `1 - alpha^t` (Adam's form, which belongs to the update
`m <- alpha m + (1 - alpha) x`). For this update the correct factor is
`1 - (1 - alpha)^t`. With `alpha = 0.9`, a unit hit once returns **9x** its
transition (5.2x after two hits, 2.4x after five, 1.14x after twenty).

Seen in practice: CartPole's reward is always 1, but generated rewards reached
14.6. In a 500-episode development check (5 seeds), fixing only this factor
raised the mean return of the last 100 episodes from 59 to 111.

Remaining artefact, not changed: on a unit's first hit the variance update uses
`m_old = 0`, so the first variance is `alpha * x^2` (one hit with reward 1
gives a reward spread of about 1). The correct factor does not remove this; it
fades by a factor of 10 per further hit. The `exact` averaging (2.2) has no
such artefact.

### 1.2 Time limit stored as a terminal state (`agent.py`, `train`)

`done = terminated or truncated` was stored in the buffer. CartPole-v1 truncates
at 500 steps while the pole is still up, so the best states were taught to the
DQN as failures: target `r` instead of `r + gamma * max Q(s')`, about 1 instead
of about 100. The state has no time in it, so the error spreads to similar
upright states, more often the better the agent gets. In a SOM whose distance
includes `done`, these false endings are also grouped with real falls, and the
SOM then generates `done = 1` from healthy states.

### 1.3 Other issues noted, not changed in the original classes
- `WeightedSOMBuffer` hard-codes CartPole (`state_dim=4`, `input_dim=11`, hand
  weights on raw values); other environments crash. `args.grid_size` is unused.
- Actions are decoded by `clip(round(.), 0, 1)`, so only two actions are possible.
- The SOM's step size and neighbourhood decay with the **episode** index
  (`exp(-e/180)`), not with steps. Kept as designed.
- `log_utils.print_info` uses undefined names (`max_score`, `max_ep`); never called.
- `generate_sample_rb.py` writes to a hard-coded `/home/sr8685/...` path.
- `__pycache__/*.pyc` files are tracked in git.
- `train.py` needs `tyro`, which is not in the RLSOM virtualenv; the
  experiment runner uses `argparse` instead.

## 2. Changes

### 2.1 `agent.py` (edited)
- **Fix 1.2:** `train()` now stores `done = terminated`; the episode still ends
  on `terminated or truncated`.
- `train()` takes an optional `rb` argument (default `WeightedSOMBuffer()`, as before).
- New `train_steps()`: the same loop as `train()` (learning from step 150,
  target sync every `update_target_freq` steps, epsilon decay once per
  episode, one update per step) run for a fixed number of **steps**, with greedy
  evaluations at given checkpoints. Only real endings are stored as terminal.
- New `evaluate()`: mean return of the greedy policy over n episodes.
- Added type hints and docstrings to the methods touched; the rest of the class
  is unchanged.

### 2.2 `structured_som.py` (new): `StructuredSOMBuffer`
A single SOM used as the replay buffer, with the distance of
`RLSOM/distancemetric_simple.md` (`train_simple.make_structured_rl_distance`):

- Vector `[s, a, r, delta_s, done]` with `delta_s = s' - s` (not `s'`); `s'` is
  rebuilt as `s + delta_s` when sampling.
- `s`, `r`, `delta_s` are z-scored with a **running** mean and SD updated at
  every step (the md file standardises over the whole buffer, which an online
  agent does not have). `a` and `done` stay 0/1.
- `D = alpha_s sum beta_i (s_i - w_i)^2 + alpha_a 1[a != round(w_a)] + alpha_r (r - w_r)^2
  + alpha_delta sum gamma_i (delta_s_i - w_i)^2 + alpha_done 1[done != round(w_done)]`
  with the md defaults `alpha_s = 1, alpha_a = 0.5, alpha_r = 0, alpha_delta = 1,
  alpha_done = 1`, CartPole `beta = [0.5, 0.5, 2.0, 1.5]`,
  `gamma = [0, 0.5, 0, 1.5]` (ones for other state sizes).
- Map update as `WeightedSOMBuffer` (lr 0.1, sigma 3.0, `exp(-episode/180)`
  decay, sigma floor 0.4), in standardised space.
- Per-unit statistics of the **raw** vectors (so they do not go stale as the
  standardisation drifts), selectable:
  - `averaging="ema"` (alpha 0.9, plainRL-SOM's update) with
    `correction="buggy"` (original factor), `"fixed"` (correct factor) or `"none"`;
  - `averaging="exact"`: exact running mean and covariance (Welford).
  - `covariance="diag"` (variance per number) or `"full"` (full covariance, so
    sampled `delta_s`, reward and done move with `s` and `a`).
- Generation: independent transitions; a unit is chosen `"uniform"`ly among
  units hit so far (as plainRL-SOM) or by `"count"` (as the md file).
- `memory_floats()`: diagonal 100 x (11 + 11 + 11 + 1) + 18 = 3 418 numbers;
  full 100 x (11 + 11 + 66 + 1) + 18 = 8 918.

### 2.3 `replay_buffers.py` (new): `RealBuffer`
FIFO, reservoir and keep-endings buffers of real transitions, ported from
`RLSOM/analysis/online_compare.py`, with the `add`/`sample` interface above.

### 2.4 `experiments.py`, `run_experiments.sh` (new)
The online comparison (claims stated in the docstring before the run): 16 arms
x 10 seeds (100-109), CartPole-v1, 200 000 steps, greedy evaluation (10
episodes) at 5k, 10k, 25k, 50k, 75k, 100k, 150k, 200k. Results in
`out/online_200k/<arm>_<seed>.json`; summary with
`python experiments.py --summarize`. Script hashes at launch are in
`out/prerun_manifest.txt`.

## 3. Checks before the run
- All 16 arms ran for 6 000 steps (seed 997) without errors.
- `fixed` correction returns exactly the transition after one hit; `buggy`
  returns reward 9.0, `none` 0.9.
- On 10 000 transitions of the repo's random-policy CartPole data: every map
  unit holds one action (purity 1.000, the action-mismatch term works);
  generated done rate 0.03-0.05 (data 0.046); one-step physics error of
  generated transitions 0.0026-0.0053 (full covariance lowest).
- **The done term never fires online:** no unit's done weight rounds to 1, so
  0% of ending transitions land in an "ending unit" (the md file reports about
  46% offline, with whole-buffer standardisation and multi-epoch training).
  Online, a mismatch on done is the same constant for every unit, so done does
  not affect the BMU choice. Endings share units with non-endings and are
  generated through each unit's mean `done` value. Not changed; reported.
- With `ema` averaging the generated CartPole rewards spread from about -1 to
  3.4 (first-hit variance artefact, 1.1); with `exact` averaging they stay in
  [0.99, 1.01].

## 4. Results of the 200k-step run (2026-11-04)

All 160 runs finished without errors. Full summary:
`out/online_200k_summary.txt` (`python experiments.py --summarize`).

Mean greedy return per checkpoint (10 seeds):

| Arm | 5k | 25k | 50k | 100k | 150k | 200k | curve | late (>=100k) |
|---|---|---|---|---|---|---|---|---|
| orig (as committed) | 43 | 179 | 159 | 143 | 165 | 203 | 166 | 170 |
| md-buggy-diag | 47 | 80 | 197 | 255 | 265 | 229 | 182 | 250 |
| **md-fixed-diag** | **254** | **356** | 341 | 339 | 356 | 355 | **333** | 350 |
| md-none-diag | 190 | 250 | 367 | 360 | 353 | 257 | 303 | 323 |
| md-exact-diag | 70 | 179 | 269 | 335 | 283 | 298 | 246 | 305 |
| md-fixed-full | 130 | 240 | 275 | 260 | 323 | 398 | 272 | 327 |
| md-fixed-diag-count | 177 | 264 | 269 | 279 | 351 | 307 | 270 | 312 |
| fifo-310 | 214 | 268 | 277 | 313 | **412** | **479** | 317 | **402** |
| reservoir-310 | 213 | 153 | 169 | 204 | 106 | 227 | 193 | 179 |
| keep-endings-310 | 254 | 209 | 181 | 70 | 30 | 27 | 143 | 42 |
| fifo-810 | 188 | 156 | 324 | 348 | 418 | 421 | 302 | 396 |
| fifo-10000 | 207 | 166 | 279 | 423 | 322 | 374 | 297 | 373 |

Claims (curve score, paired by seed, 95% bootstrap over seeds):
1. Bias fix: md-fixed-diag - md-buggy-diag = +150.1 [+77.9, +226.5], 9/10. **Holds.**
2. No correction vs fixed: -29.2 [-70.2, +12.4], 2/10. **Holds** (interval includes 0; the trend favours the fix).
3. md-fixed-diag loses to fifo-310: **Failed.** +15.8 [-28.8, +56.1], 7/10 (a tie). It leads early
   (5k-25k) and trails late: late score -51.6 [-111.1, +0.7].
4. Not predicted: exact averaging is worse than EMA (-86.2 [-133.9, -38.7]); full covariance is worse
   than diagonal (-60.3 [-97.6, -22.1]); units by count are worse than uniform (-62.4 [-106.3, -19.8]);
   md metric vs committed metric, both with the broken correction: +16.9 [-72.2, +104.2] (tie);
   md-fixed-diag beats reservoir-310 (+139.9, 10/10) and keep-endings-310 (+189.9, 10/10) and ties
   fifo-10000 (+35.7 [-11.1, +87.9]); orig loses to fifo-310 (-151.2, 0/10).

## 5. Changes for the alpha / ending-decoding study (2026-11-04)

- `structured_som.py`: new option `done_decode` (`"round"`, the default and the
  only behaviour used before; `"bernoulli"` draws the ending flag as
  Bernoulli(unit's mean ending value)). The Bernoulli draw comes after the
  Gaussian draw, so the default path's random stream is unchanged; verified
  bit-identical on 10 000 transitions under five option sets
  (`dev_checks/check_default_unchanged.py`).
- New `experiments_alpha.py` (adds 6 arms to `experiments.ARMS` at import,
  without changing `experiments.py`) and `run_alpha.sh`. Seeds 110-119.
- New `dev_checks/` scripts for every check run so far (see `REPRODUCIBILITY.md`).

## 6. Results of the alpha / ending-decoding study (2026-11-04, seeds 110-119)

Full summary: `out/alpha_200k_summary.txt`; pooled comparisons and sample
diagnostics: `out/alpha_200k_pooled_and_diag.txt`.

| Arm (diagonal, fixed correction) | Whole run | Late (>=100k) | Ending rate |
|---|---|---|---|
| alpha 0.99 | 331 | 358 | 4.5% |
| alpha 0.9 | 285 | 348 | 3.5% |
| alpha 0.5 | 305 | 336 | 4.5% |
| alpha 0.1 | 294 | 376 | 1.9% |
| alpha 0.01 | 204 | 257 | 2.2% |
| exact | 282 | 298 | 0.7% |
| alpha 0.9, Bernoulli endings | 294 | 313 | 5.7% |
| **exact, Bernoulli endings** | **329** | **403** | 1.0% |
| FIFO 310 | 300 | 422 | -- |

- Claim 1 (alpha 0.9 beats alpha 0.01 and exact): held vs 0.01 (+81.5), failed vs exact (+3.0, tie).
- Claim 2 (replication vs FIFO): held. Pooled over 20 seeds: whole +0.4 [-26, +26], late -63 [-101, -27].
- Claim 3: decision rule did not apply. Exact + Bernoulli endings beat exact by +47 (9/10), late +105 (10/10),
  and was the only SOM not behind FIFO late (-20 [-108, +70]). Needs replication.
- The alpha sweep is not monotone and is confounded by the zero-start variance artefact,
  which lasts longer for small alpha (generated rewards [-1.73, 3.55] at alpha 0.01).

## 7. Replication of exact means + Bernoulli endings (2026-11-04, seeds 120-139)

No code change (new `experiments_replication.py`, `run_replication.sh`).
Summary `out/replication_200k_summary.txt`, pooled `out/replication_200k_pooled.txt`.

| Arm | Whole run | Late (>=100k) |
|---|---|---|
| **exact, Bernoulli endings** | **330** | **425** |
| exact, rounded endings | 272 | 353 |
| alpha 0.9, rounded endings | 304 | 357 |
| FIFO 310 | 302 | 408 |

Claims: 1 failed (vs alpha 0.9 whole +26.7 [-3.5, +55.1]); 2, 3, 4 held; not
predicted: beats FIFO over the whole run (+28.7 [+1.2, +56.1]). Over 30 seeds
it beats FIFO whole (+28.8 [+7.7, +49.9]) and ties it late (+5.1 [-36.5, +44.6]).
Recommended buffer from here: `StructuredSOMBuffer(averaging="exact", done_decode="bernoulli")`.

## 8. Changes for the burn-in study (2026-11-04)

- `structured_som.py`: `min_hits` (burn-in: only units with at least this many
  hits generate; falls back to all hit units if none qualifies; default 1 =
  previous behaviour) and `averaging="hybrid"` with `kappa_min` (rate
  `max(1/n, kappa_min)`: exact for the first `1/kappa_min` hits, then slow
  forgetting; spread stored directly, no correction). Defaults verified
  bit-identical to the previous version (`dev_checks/check_default_unchanged_v3.py`).
- New `experiments_burnin.py`, `run_burnin.sh`, `dev_checks/early_diagnostic.py`.

## 9. Results of the burn-in study (2026-11-04, seeds 140-159)

Summary `out/burnin_200k_summary.txt`; pooled `out/burnin_200k_pooled.txt`.

| Arm (Bernoulli endings) | Early (5-25k) | Whole run | Late | 5k |
|---|---|---|---|---|
| exact | 184 | 320 | 417 | 86 |
| exact, burn-in 5 | 202 | 321 | 405 | 86 |
| exact, burn-in 20 | 197 | 320 | 401 | 76 |
| hybrid (kappa_min 0.01) | 191 | 332 | 438 | 86 |
| hybrid, burn-in 20 | 214 | 334 | 423 | 75 |
| FIFO 310 | 204 | 284 | 383 | 186 |

- Claims 1-2 failed: neither burn-in nor hybrid averaging detectably raised the early score; claim 3 held.
- The 5k deficit is unchanged by both; only fast averaging (EMA alpha 0.9) scores higher at 5k.
- Pooled over 50 seeds (Studies 2-4): exact+Bernoulli beats FIFO over the whole run, +31.6 [+13.7, +48.7].

## 10. Changes for Study 5, averaging rate (2026-11-05)

- `structured_som.py`: `kappa_until` (with `averaging="hybrid"`, the floor rate
  `kappa_min` applies only while the memory has received at most this many
  transitions; exact averaging afterwards). Default `None` = unchanged;
  verified bit-identical (`dev_checks/check_default_unchanged_v4.py`) and
  checked for its two limits (`dev_checks/out_check_kappa_until.txt`).
- New `experiments_rate.py`, `run_rate.sh`. Seeds 160-179.

## 11. Results of Study 5, averaging rate (2026-11-05, seeds 160-179)

Summary `out/rate_200k_summary.txt`; pooled `out/rate_200k_pooled.txt`.

| Arm (Bernoulli, clean start) | 5k | Whole run | Late |
|---|---|---|---|
| rate 0.9 | 210 | 286 | 328 |
| rate 0.5 | 187 | 301 | 355 |
| rate 0.1 | 124 | 343 | 413 |
| rate 0.03 | 119 | 343 | 427 |
| rate 0.01 (hybrid) | 93 | 347 | 442 |
| exact | 94 | 338 | 424 |
| 0.9 for 10k transitions, then exact | 210 | 288 | 366 |
| FIFO 310 | 216 | 292 | 384 |

- Claims 1-2 held, claim 3 failed: the switch froze each unit near its last pre-switch
  transition (median 99 hits at the switch, so later updates move the mean by ~1%).
- Trade-off: fast rates start like FIFO but finish worse; slow/exact start slowly and finish best.
- Pooled 70 seeds: exact+Bernoulli beats FIFO over the whole run, +36.0 [+20.8, +50.9].
