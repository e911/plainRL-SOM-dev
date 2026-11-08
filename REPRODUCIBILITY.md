# Reproducibility log

Every command run in this copy, in order, with what it produced. Append a new
entry for every further run (commands, seeds, script hashes, outputs, where
the numbers went in the paper). Paths are relative to this folder
(`/home/pujan/Documents/plainRL-SOM-dev`) unless stated.

## Environment

| Item | Value |
|---|---|
| Machine | Intel Core i7-14700F, 28 threads; Linux 7.0.0-42-generic x86_64 |
| Python | 3.12.3, virtualenv `~/.virtualenvs/rlsom` (`PY=$HOME/.virtualenvs/rlsom/bin/python`) |
| Packages | torch 2.15.0+cu132 (run on CPU), gymnasium 1.4.0, numpy 2.5.4, matplotlib 3.11.2 |
| Not installed | `tyro` (needed by the original `train.py`), LaTeX (the paper is compiled on Overleaf) |
| Settings for every run | `PYTHONDONTWRITEBYTECODE=1` (the original repo tracks `__pycache__/*.pyc`), `CUDA_VISIBLE_DEVICES=` (CPU), `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` |

## Entry 1 — 2026-11-04: copy of the original repository

```sh
cd /home/pujan/Documents
cp -a RLSOM/plainRL-SOM plainRL-SOM-dev
cd plainRL-SOM-dev
git remote set-url --push origin DISABLED-no-push    # pushing disabled; nothing is ever committed here
```

Source: `sheeraja/RL-SOM`, branch `main`, commit `3e23ea1` ("Updated sample
generation"), clean working tree. The original folder
`/home/pujan/Documents/RLSOM/plainRL-SOM` is untouched.

## Entry 2 — 2026-11-04: development checks of the original code (before any change)

Run first in a scratchpad copy of the original files; the scripts are kept in
`dev_checks/` and re-run there on 2026-11-04 with identical output. Run from
this folder's root.

```sh
PY=$HOME/.virtualenvs/rlsom/bin/python
cd dev_checks
$PY check_bias_numeric.py > out_check_bias_numeric.txt   # EMA correction: 9x after 1 hit, 5.2x after 2, ... (paper Sec. 2.2)
$PY distance_shares.py    > out_distance_shares.txt      # block shares of WeightedSOMBuffer's distance
$PY action_purity.py      > out_action_purity.txt        # committed SOM: action purity 0.980 / 0.979
cd ..
```

500-episode online check (original agent loop, which still stores the time
limit as an ending; arms `orig`, `fixbias`, `fifo10k`, `fifo309`; seeds 0-4).
`driver_500ep.py` imports `agent`, `args` and `SOMBuffer` from the current
directory and writes `out_<arm>_<seed>.json` there:

```sh
for a in orig fixbias fifo10k fifo309; do for s in 0 1 2 3 4; do echo "$a $s"; done; done \
  | xargs -P 20 -n 2 sh -c '$HOME/.virtualenvs/rlsom/bin/python dev_checks/driver_500ep.py $0 $1 500'
```

Results (copied to `dev_checks/out_500ep/`): last-100-episode return
`orig` 58.6, `fixbias` 111.0, `fifo10k` 140.9, `fifo309` 221.0; sampled
rewards up to 14.6 with the original correction. Used for pre-registered
claim 3 and in the paper's Section 1 and Section 3 ("development check").

## Entry 3 — 2026-11-04: code changes

See `change.md` for every change. Summary: `agent.py` (truncation fix,
`train_steps`, `evaluate`), new `structured_som.py`, `replay_buffers.py`,
`experiments.py`, `run_experiments.sh`.

## Entry 4 — 2026-11-04: smoke test and pre-run validation

```sh
# all 16 arms, 6 000 steps, development seed 997, output outside this folder
O=/tmp/claude-1000/-home-pujan-Documents-RLSOM/bf28f151-c815-48ef-ad75-28cee34addc1/scratchpad/smoke
for a in $($PY -c "import experiments as E; print(' '.join(E.ARMS))"); do echo "$a"; done \
  | xargs -P 17 -I{} $PY experiments.py --arm {} --seed 997 --steps 6000 --out $O
$PY dev_checks/prerun_validation.py > dev_checks/out_prerun_validation.txt
```

All arms ran without errors. Validation output (`dev_checks/out_prerun_validation.txt`):
fixed correction returns the transition after one hit (buggy: reward 9.0,
none: 0.9); action purity 1.000; ending rate 0.028-0.049 (data 0.046);
physics error 0.0026-0.0053; **0% of endings in an ending unit** (paper
Section 5, "The ending term of the distance").

## Entry 5 — 2026-11-04: main run (pre-registered), 160 runs

Claims are in the docstring of `experiments.py`, written before launch.

```sh
sha256sum experiments.py structured_som.py replay_buffers.py agent.py SOMBuffer.py networks.py args.py > out/prerun_manifest.txt
date -u >> out/prerun_manifest.txt          # Wed Nov  4 09:40:33 PM UTC 2026
sh run_experiments.sh 26                    # 16 arms x seeds 100-109, 200 000 steps each, 26 in parallel
```

| Item | Value |
|---|---|
| Script hashes at launch | `out/prerun_manifest.txt`; re-checked after the run with `head -7 out/prerun_manifest.txt \| sha256sum -c -` (all OK) |
| Duration | 21:40 to 22:09 UTC (3-7 min per run) |
| Results | `out/online_200k/<arm>_<seed>.json` (160 files), runner log `out/runner.log`, 0 failures |
| Summary | `$PY experiments.py --summarize > out/online_200k_summary.txt` |

The first `online_200k_summary.txt` was cut short by a display pipe
(`| tee ... | head`); it was regenerated in full (329 lines) with the command above.

Results: claim 1 held (+150.1 [+77.9, +226.5]), claim 2 held (-29.2 [-70.2,
+12.4]), claim 3 failed (+15.8 [-28.8, +56.1], a tie). Written into
`change.md` §4 and the paper (Tables 2-6, Figures 1-3).

## Entry 6 — 2026-11-04: paper (ICML 2026 format, working draft)

```sh
# official ICML 2026 style files, downloaded outside this folder
curl -sSfL -o icml2026.zip https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip
# sha256 8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c
# copied: icml2026.sty icml2026.bst fancyhdr.sty algorithm.sty algorithmic.sty -> paper/
$PY paper/make_figures.py            # paper/figures/fig_curves.pdf, fig_gap.pdf, fig_ablation.pdf
cd paper && zip -r rlsom_icml_overleaf.zip rlsom_icml.tex references.bib *.sty *.bst figures/*.pdf
```

- Source: `paper/rlsom_icml.tex`; bibliography `paper/references.bib` (entries copied from
  `RLSOM/paper/rlsom_tmlr_overleaf/references.bib`, plus Pardo et al. 2018 and Welford 1962).
- Every number in the paper comes from `out/online_200k_summary.txt`
  (main run) or `dev_checks/out_*.txt` (pre-run checks).
- Figure colours: validated categorical slots 1-4 (CVD and normal-vision checks pass).
- No LaTeX is installed locally: a script checked balanced braces and
  environments, and that every `\ref` and `\cite` resolves. Compile on Overleaf
  by uploading `paper/rlsom_icml_overleaf.zip`.

## Script hashes (sha256) at the time of Entry 6

```
e375a2093548d78534883bf9df890f35b929fef28eaf1bb66cbca6b726f1d6e4  experiments.py
60a58e9ed96c2177ff736f0811f2454a151db7032c794d65738dd449d3801a82  structured_som.py
9431f27e10d652a604e4f35a0a024cdd18ec48aa0bc4bd5e9e9a6017cfd93401  replay_buffers.py
8eaf4c8d88ef7869604ce92dcf29786aeebf9e8416abdaca40840c52d92c1c1a  agent.py
470122f1f416aaa8c2268c029fb7c6f66adf357a668964d9282b1e2bf3018cd0  SOMBuffer.py
058bb8780ecc2b9bf50a500e181da20cedd8b60aadee0502982aa33187c8928d  dev_checks/action_purity.py
729b3b9bb74172f96666619b18f1d1f12da575f276c012384e97900a5b55b911  dev_checks/check_bias_numeric.py
cdf528592e43bbea492d766665dce268c13e65ab67a8b284987db776cfb76490  dev_checks/distance_shares.py
695ebf8656d7deff1be68a8bf78904691d7af54b6304e1137545005b0cebf26a  dev_checks/driver_500ep.py
cab56a30a1dcab94650b9dff3c88b13d1a14eced7c5ace59bcdb4786a3bd3b8c  dev_checks/load_data.py
0018bf2a0aedb9cb53d5b6830ae322c88ab2d9064e70f42221cb62f7a6f53822  dev_checks/prerun_validation.py
7e2b2f0cd9fd10a8860fa1353a086a4c1fb8b6a5714fa5c0b6b7712d171c7331  paper/make_figures.py
```

## Entry 7 — 2026-11-04: alpha / ending-decoding study (pre-registered), 90 runs

Question: is the fixed SOM's advantage over the exact-mean SOM recency (alpha)
or the generated ending rate? Claims are in the docstring of
`experiments_alpha.py`, written before launch.

Code change: `structured_som.StructuredSOMBuffer` gained
`done_decode="round" | "bernoulli"` (default `"round"`). Defaults verified
bit-identical to the Entry 5 version (`dev_checks/_structured_som_v1.py`):

```sh
$PY dev_checks/check_default_unchanged.py > dev_checks/out_check_default_unchanged.txt   # 5 option sets: identical
```

Smoke test (all 9 arms, 6 000 steps, seed 997; md-fixed-diag gives 110.6, the
same as in Entry 4):

```sh
O=/tmp/claude-1000/-home-pujan-Documents-RLSOM/bf28f151-c815-48ef-ad75-28cee34addc1/scratchpad/smoke_alpha
for a in $($PY -c "import experiments_alpha as A; print(' '.join(A.STUDY))"); do echo $a; done \
  | xargs -P 9 -I{} $PY experiments_alpha.py --arm {} --seed 997 --steps 6000 --out $O
```

Launch:

```sh
sha256sum experiments_alpha.py experiments.py structured_som.py replay_buffers.py agent.py SOMBuffer.py networks.py args.py run_alpha.sh \
  > out/alpha_200k/prerun_manifest.txt
date -u >> out/alpha_200k/prerun_manifest.txt
sh run_alpha.sh 26          # 9 arms x seeds 110-119, 200 000 steps
```

Arms: alpha 0.99 / 0.9 / 0.5 / 0.1 / 0.01 and exact (diagonal, fixed
correction, uniform units, rounded endings); alpha 0.9 and exact with
Bernoulli endings; FIFO 310. Results in `out/alpha_200k/<arm>_<seed>.json`.

Run finished 2026-11-04 (90/90 results, 0 failures; script hashes re-checked
after the run with `head -9 out/alpha_200k/prerun_manifest.txt | sha256sum -c -`: all OK).

```sh
$PY experiments_alpha.py --summarize > out/alpha_200k_summary.txt     # 221 lines
# pooled 20-seed comparisons (arms run in both studies) and sample diagnostics:
#   inline script recorded in this entry's output file
#   -> out/alpha_200k_pooled_and_diag.txt
```

The pooled/diagnostics numbers were produced by an inline script that loads
both result folders with `experiments.load`, merges `md-fixed-diag`,
`md-exact-diag` and `fifo-310` over seeds 100-119, and calls
`experiments.paired(..., rng=np.random.default_rng(0))` on the "curve" and
"late" scores; and prints `diag` (done_rate, reward_min, reward_max) per arm.

Results: claim 1 held for alpha 0.01 (+81.5 [+36.7, +127.5]) and failed for
the exact mean (+3.0 [-43.7, +45.2]); claim 2 held (whole -15.1 [-40.7, +13.7],
late -74.5 [-116.5, -25.3]); claim 3's decision rule did not apply (exact did
not lose with rounded endings). Not pre-registered as a conclusion: exact +
Bernoulli endings was the best SOM (whole 329, late 403; vs exact +47.2
[+21.4, +72.2], late +104.6 [+56.4, +152.0]; vs FIFO late -19.8 [-108.1, +70.2]).
Pooled 20 seeds: alpha 0.9 - FIFO whole +0.4 [-26.0, +26.2], late -63.0
[-100.9, -26.9]; exact - alpha 0.9 whole -44.6 [-81.3, -6.9].

Paper updated (new Sec. 4.4 "Recency or endings?", Table 6, Figure 4
`fig_alpha.pdf`, appendix Table 10; abstract, introduction, Sec. 4.3,
discussion, limitations and conclusion revised; previous version kept at
`/tmp/claude-1000/-home-pujan-Documents-RLSOM/bf28f151-c815-48ef-ad75-28cee34addc1/scratchpad/rlsom_icml_before_alpha.tex`):

```sh
$PY paper/make_figures.py      # adds fig_alpha.pdf
cd paper && zip -r rlsom_icml_overleaf.zip rlsom_icml.tex references.bib *.sty *.bst figures/*.pdf
```

Hashes after Entry 7:
```
    3809ec455c6ca36ee93d5f5c396d59f9c2a08be1ec2dfa7459ea863ea0732653  paper/make_figures.py
    03282676441d3a3f2939f148004e1919c160ea7ef2b959b262e6883d82073fb5  paper/rlsom_icml.tex
```

## Entry 8 — 2026-11-04: replication of exact + Bernoulli endings (pre-registered), 80 runs

Claims are in the docstring of `experiments_replication.py`, written before
launch. No library code changed; arms come from `experiments.py` and
`experiments_alpha.py`. New seeds 120-139 (20 seeds).

Smoke test (4 arms, 6 000 steps, seed 997; values identical to Entries 4 and 7:
fifo-310 166.8, md-exact-diag 42.9, md-fixed-diag 110.6, md-exact-diag-bern 85.3):

```sh
O=/tmp/claude-1000/-home-pujan-Documents-RLSOM/bf28f151-c815-48ef-ad75-28cee34addc1/scratchpad/smoke_repl
for a in md-exact-diag-bern md-exact-diag md-fixed-diag fifo-310; do echo $a; done \
  | xargs -P 4 -I{} $PY experiments_replication.py --arm {} --seed 997 --steps 6000 --out $O
```

Launch:

```sh
sha256sum experiments_replication.py experiments_alpha.py experiments.py structured_som.py replay_buffers.py agent.py \
  SOMBuffer.py networks.py args.py run_replication.sh > out/replication_200k/prerun_manifest.txt
date -u >> out/replication_200k/prerun_manifest.txt
sh run_replication.sh 26    # 4 arms x seeds 120-139, 200 000 steps
```

Run started 22:52:28 UTC and finished 2026-11-04 (80/80 results, 0 failures;
script hashes re-checked after the run with
`head -10 out/replication_200k/prerun_manifest.txt | sha256sum -c -`: all OK).

```sh
$PY experiments_replication.py --summarize > out/replication_200k_summary.txt
```

Pooled comparisons (`out/replication_200k_pooled.txt`) come from an inline
script that merges result folders with `experiments.load` (seeds 110-139 for
the four replication arms; seeds 100-139 for md-fixed-diag, md-exact-diag and
fifo-310) and calls `experiments.paired(..., rng=np.random.default_rng(0))`
on the "curve" and "late" scores.

Results (20 new seeds): claim 1 FAILED (exact+Bernoulli - alpha 0.9 whole
+26.7 [-3.5, +55.1]); claim 2 held (whole +58.8 [+23.7, +92.7], late +71.7
[+25.1, +115.0]); claim 3 held (late vs FIFO +17.6 [-22.0, +57.2]); claim 4
held (alpha 0.9 - FIFO late -50.5 [-86.0, -12.5]); 5 (not predicted):
exact+Bernoulli - FIFO whole +28.7 [+1.2, +56.1]. Pooled 30 seeds:
exact+Bernoulli - FIFO whole +28.8 [+7.7, +49.9], late +5.1 [-36.5, +44.6];
vs alpha 0.9 whole +32.5 [+6.6, +58.8]. Pooled 40 seeds: alpha 0.9 - FIFO
whole +1.2 [-15.3, +17.6], late -56.8 [-82.8, -29.6].

Paper updated: new Sec. 4.5 (Table 7, Figure 5 `fig_replication.pdf`),
appendix table of the replication, abstract (40-seed numbers), introduction,
discussion ("The late gap", new "The slow start"), limitations, conclusion.
Previous version: scratchpad `rlsom_icml_before_replication.tex`.

```sh
$PY paper/make_figures.py      # adds fig_replication.pdf
cd paper && zip -r rlsom_icml_overleaf.zip rlsom_icml.tex references.bib *.sty *.bst figures/*.pdf
```

Hashes after Entry 8:
```
    a5ab09688c6103dca80d42d5603d7b80f767423006cfd0a3d35349762d30317f  experiments_replication.py
    2ef213a4b3f8f52cd624266c982a9ab3fafe1cdaccc4e5aff9364c3ec4a0331f  run_replication.sh
    d16faf5260d50f7ac0a4106cbdfaffe4cc93d7ad43ffbf0295469c96a38a2517  paper/make_figures.py
    d5380c34e6151ce3d64214b0d67d501dc50afd391abadcb2317f58fc39423666  paper/rlsom_icml.tex
```

## Entry 9 — 2026-11-04: burn-in study (pre-registered), 120 runs

Diagnostic first (development seeds 997-999 only, 25 000 steps; arms
md-exact-diag-bern, md-fixed-diag, md-exact-diag):

```sh
for s in 997 998 999; do for a in md-exact-diag-bern md-fixed-diag md-exact-diag; do echo "$a $s"; done; done \
  | xargs -P 9 -n 2 sh -c "$PY -c \"import sys; sys.path.insert(0,'dev_checks'); import early_diagnostic as D; D.run('\$0', int('\$1'))\"" \
  > dev_checks/out_early_diagnostic.txt 2>&1
```

Finding: units with < 5 hits give 23-38% of generated transitions at 1k steps
and 4-10% at 5k, equally for all buffers; the exact-mean buffer's physics
error stays about 0.009-0.018 over the first 25k steps while alpha 0.9 falls
to about 0.005 (stale averages while the map moves). This motivated the
hybrid arm next to the requested burn-in.

Code change: `structured_som.StructuredSOMBuffer` gained `min_hits` (burn-in;
default 1) and `averaging="hybrid"` with `kappa_min` (default 0.01).
Previous version saved as `dev_checks/_structured_som_v2.py`; defaults
verified bit-identical under six option sets:

```sh
$PY dev_checks/check_default_unchanged_v3.py > dev_checks/out_check_default_unchanged_v3.txt
```

Also checked inline: hybrid with kappa_min = 1e-12 equals exact (means and
variances, 3 000 transitions); min_hits = 20 leaves 37 of 89 units eligible
after 3 000 transitions and samples correctly.

Smoke test (6 arms, 6 000 steps, seed 997; exact-bern 85.3 = md-exact-diag-bern
in Entry 8):

```sh
O=/tmp/claude-1000/-home-pujan-Documents-RLSOM/bf28f151-c815-48ef-ad75-28cee34addc1/scratchpad/smoke_burnin
for a in $($PY -c "import experiments_burnin as B; print(' '.join(B.STUDY))"); do echo $a; done \
  | xargs -P 6 -I{} $PY experiments_burnin.py --arm {} --seed 997 --steps 6000 --out $O
```

Launch (claims in the docstring of `experiments_burnin.py`, written before launch):

```sh
sha256sum experiments_burnin.py experiments.py structured_som.py replay_buffers.py agent.py SOMBuffer.py networks.py args.py \
  run_burnin.sh > out/burnin_200k/prerun_manifest.txt
date -u >> out/burnin_200k/prerun_manifest.txt
sh run_burnin.sh 26      # 6 arms x seeds 140-159, 200 000 steps
```

## Entry 10 — 2026-11-04: paper rewrite and citation verification

The paper was rewritten as a design study (four studies, no description of
code history); arms of Study 1 that are not design choices (the
`orig`/`md-buggy-diag` arms) are not reported in the paper but remain in
`out/online_200k/` and `out/online_200k_summary.txt`. Figures updated:
`fig_curves.pdf` now shows keep-endings instead of the `orig` arm, and
`fig_ablation.pdf` shows four variants. Previous paper versions are in the scratchpad
(`rlsom_icml_before_rewrite.tex`, `references_before_verify.bib`).

Citation verification (every entry of `paper/references.bib`, 20 entries):

```sh
# journal / book / conference DOIs: title, authors, venue, volume, issue, pages, year
curl -s "https://api.crossref.org/works/<DOI>"
#   10.1007/BF00992699  10.1038/nature14236  10.1007/BF00337288  10.1007/978-3-642-56927-2
#   10.1109/TSMC.1983.6313077  10.1145/3147.3165  10.3389/fnbot.2023.1127642
#   10.1177/1059712318818568  10.1080/00401706.1962.10490022  10.1609/aaai.v32i1.11694
# conference versions found by bibliographic query:
curl -s "https://api.crossref.org/works?rows=2&query.bibliographic=<title author>"
#   Isele & Cosgun -> 10.1609/aaai.v32i1.11595 (AAAI 2018); Hayes et al. -> 10.1109/ICRA.2019.8793982 (ICRA 2019)
# arXiv preprints and arXiv versions (titles and authors):
curl -sL "https://export.arxiv.org/api/query?id_list=1511.05952,1802.10269,1809.05922,1412.6980,2108.13264,1705.08690,2407.17032,1712.00378,2608.29860,2608.27662"
curl -s "https://api.datacite.org/dois/10.48550/arXiv.2108.13264"
# PMLR page of Pardo et al. (pages)
curl -s https://proceedings.mlr.press/v80/pardo18a.html | grep citation_
```

All 20 entries match their records. Corrections made: Isele & Cosgun and
Hayes et al. now carry their official conference DOIs instead of arXiv DOIs;
Karimpanal's name follows the publisher record (given name Thommen, family
name George Karimpanal). Pardo et al. keep pages 4045-4054 from the PMLR page
(the arXiv record's journal reference says 4042-4051, which disagrees with
PMLR). Karimpanal & Bouffanais keep year 2019 (print issue 27(2), April 2019;
online December 2018). Unused entries were removed.

Hashes after Entry 10:
```
    a06d90bb653385f0805df46e634b0dcb409aca9d58657d05b9ed7dc4870042f9  paper/rlsom_icml.tex
    41bff477b8fa33d27dec26bf7d5af7c8bd94aa2aaa9b39da8c8305222db79826  paper/references.bib
    ef0d6b2a00a3b9464c93ac73cbe74bbc174478b3160067ac7ff13fedcdaa2bbf  paper/make_figures.py
```

### Entry 9 (continued) — burn-in study results

Run started 23:57:27 UTC 2026-11-04; 120/120 results, 0 failures; script
hashes re-checked after the run with
`head -9 out/burnin_200k/prerun_manifest.txt | sha256sum -c -`: all OK.

```sh
$PY experiments_burnin.py --summarize > out/burnin_200k_summary.txt
```

Pooled comparison (`out/burnin_200k_pooled.txt`): inline script that merges
`md-exact-diag-bern` (Studies 2, 3) and `exact-bern` (Study 4, the same
configuration) with `fifo-310` over seeds 110-159 and calls
`experiments.paired(..., rng=np.random.default_rng(0))` on the curve, late
and checkpoint scores.

Results: claim 1 FAILED (burn-in 20 - exact, early +13.1 [-12.8, +39.5]);
claim 2 FAILED (hybrid - exact, early +7.6 [-20.3, +33.8]); claim 3 held
(whole +0.9 [-25.2, +27.8] and +12.9 [-18.4, +46.1]). At 5k every SOM arm
75-86 vs FIFO 186. Exploratory: every SOM arm beat FIFO over the whole run
(+35.9 to +50.7); hybrid beat FIFO late (+54.5 [+15.3, +95.7]). Pooled 50
seeds: exact+Bernoulli - FIFO whole +31.6 [+13.7, +48.7] (39/50), late +16.4
[-15.5, +47.7], 5k -87.8, 50k +117.9.

Paper: Study 4 section (Table 7, Figure 6 `fig_burnin.pdf`), appendix
table, discussion ("The slow start"), conclusion and limitations updated.
Previous version: scratchpad `rlsom_icml_before_study4.tex`.

```sh
$PY paper/make_figures.py      # adds fig_burnin.pdf
cd paper && zip -r rlsom_icml_overleaf.zip rlsom_icml.tex references.bib *.sty *.bst figures/*.pdf
```

Hashes after Entry 9 results:
```
    8dfb46ed2da09bd5aff2b0fcb7f9cc418c6634e4f5bfae16ed75d42e31e5b667  experiments_burnin.py
    400675f3ffec4cf4a6cb8eb7d8062f23561902d10528555ac008c026b0e354bb  run_burnin.sh
    0b50728190148bef19c6483a189de567c14fe0f718b46c1c47e72c9a932f37a5  structured_som.py
    09c93087527b176c3d855399423fd2ec43e0d7f6a3fb6077bb59e80af2e1aae3  paper/make_figures.py
    507bac9362705c49a97886b8f7a64dbe6e5977fd3e1374e2864117744decfd23  paper/rlsom_icml.tex
```

## Entry 11 — 2026-11-05: Study 5, averaging rate with Bernoulli endings (pre-registered), 160 runs

Code change: `structured_som.StructuredSOMBuffer` gained `kappa_until`
(hybrid floor rate only for the first `kappa_until` transitions of the
stream; default `None` = unchanged). Previous version saved as
`dev_checks/_structured_som_v3.py`. Checks:

```sh
$PY dev_checks/check_default_unchanged_v4.py > dev_checks/out_check_default_unchanged_v4.txt   # 8 option sets identical
# inline check -> dev_checks/out_check_kappa_until.txt:
#   kappa_until=0 equals exact; kappa_until=1e9 equals no limit; kappa_until=2000 differs from both
```

Smoke test (8 arms, 6 000 steps, seed 997; fifo-310 166.8, hybrid-bern 74.9 and
exact-bern 85.3 as in Entry 9; fast-early-exact = rate-0.9 = 106.2 before its
switch at 10 000 transitions):

```sh
O=/tmp/claude-1000/-home-pujan-Documents-RLSOM/bf28f151-c815-48ef-ad75-28cee34addc1/scratchpad/smoke_rate
for a in $($PY -c "import experiments_rate as R; print(' '.join(R.STUDY))"); do echo $a; done \
  | xargs -P 8 -I{} $PY experiments_rate.py --arm {} --seed 997 --steps 6000 --out $O
```

Launch (claims in the docstring of `experiments_rate.py`, written before launch):

```sh
sha256sum experiments_rate.py experiments_burnin.py experiments.py structured_som.py replay_buffers.py agent.py SOMBuffer.py \
  networks.py args.py run_rate.sh > out/rate_200k/prerun_manifest.txt
date -u >> out/rate_200k/prerun_manifest.txt
sh run_rate.sh 26        # 8 arms x seeds 160-179, 200 000 steps
```

### Entry 11 (continued) — Study 5 results

Run started 00:33:25 UTC 2026-11-05; 160/160 results, 0 failures; hashes
re-checked after the run with `head -10 out/rate_200k/prerun_manifest.txt | sha256sum -c -`: all OK.

```sh
$PY experiments_rate.py --summarize > out/rate_200k_summary.txt
```

Pooled (`out/rate_200k_pooled.txt`): inline script merging exact+Bernoulli
(`md-exact-diag-bern` in Studies 2-3, `exact-bern` in Studies 4-5) and
`fifo-310` over seeds 110-179, and `hybrid-bern` over seeds 140-179; uses
`experiments_burnin.paired(..., rng=np.random.default_rng(0))`.
Switch check (`dev_checks/out_switch_counts.txt`): inline script feeding the
first 10 000 recorded CartPole transitions to the fast-early-exact buffer
(seed 997): median 99 hits per unit at the switch.

Results: claim 1 held (rate 0.9 - exact late -96.2 [-148.7, -46.6]); claim 2
held (rate 0.03 +4.1 [-28.7, +37.4], rate 0.01 +8.9 [-20.4, +36.7] whole);
claim 3 FAILED (fast-early-exact +116.2 at 5k but -50.9 [-72.8, -28.9] whole).
Pooled 70 seeds: exact+Bernoulli - FIFO whole +36.0 [+20.8, +50.9] (57/70),
late +23.0 [-2.6, +48.0], 5k -97.8. Pooled 40 seeds: hybrid - FIFO whole
+52.2 [+29.2, +74.2], late +56.3 [+29.0, +83.0].

Paper: Study 5 section (Table 9, Figure 7 `fig_rate.pdf`), studies table,
abstract ("five studies", trade-off sentence), discussion ("Start against
finish"), conclusion, limitations, appendix table. Previous version: scratchpad
`rlsom_icml_before_study5.tex`.

```sh
$PY paper/make_figures.py      # adds fig_rate.pdf
cd paper && zip -r rlsom_icml_overleaf.zip rlsom_icml.tex references.bib *.sty *.bst figures/*.pdf
```

Hashes after Entry 11:
```
    6d2378fa5d61b2690f3c83568e2a8971bb3c12b5948c6204c27cb481a88513aa  experiments_rate.py
    34b2533f79232ae5a745c8bc20cd10765de117a96ca512099a60eb69209d4e97  run_rate.sh
    c1f93acb2e33d9a3aa1f85b7759e6909bd96672c4238fbd34badfc059138c4d4  structured_som.py
    064dcb585f7e6c5a197f2bffc51997141e6f45c7e7d649f9ffe2690387d2d220  paper/make_figures.py
    9e55c8c6b3b32280066d4a2b5ac6e74933b7a68121a88090dd490cbba4725026  paper/rlsom_icml.tex
```
