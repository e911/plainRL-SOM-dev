"""Figures of the ICML-format working paper, from ``out/online_200k/*.json``.

Run from the copy's root:  python paper/make_figures.py
Writes paper/figures/fig_curves.pdf, fig_gap.pdf, fig_ablation.pdf (main run, out/online_200k)
fig_alpha.pdf (alpha / ending-decoding study, out/alpha_200k) and
fig_replication.pdf (replication, out/replication_200k) and fig_burnin.pdf
(burn-in study, out/burnin_200k) and fig_rate.pdf (averaging-rate study, out/rate_200k).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import CHECKPOINTS, OUT, load, paired, score  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
# Validated categorical slots 1-4 (dataviz reference palette, light surface).
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STEPS_K = np.array(CHECKPOINTS) / 1000

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def curve(data: dict[str, dict[int, dict[str, Any]]], arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over seeds of an arm's greedy return at each checkpoint.

    Args:
        data: Results by arm and seed.
        arm: Arm name.

    Returns:
        ``(mean, se)`` arrays over the checkpoints.
    """
    m = np.array([[score(r, str(c)) for c in CHECKPOINTS] for r in data[arm].values()])
    return m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m))


def fig_curves(data: dict[str, dict[int, dict[str, Any]]]) -> None:
    """Learning curves of the fixed SOM, the committed SOM and two real buffers of the same memory.

    Args:
        data: Results by arm and seed.

    Returns:
        None. Saves ``fig_curves.pdf``.
    """
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    series = [("md-fixed-diag", "SOM (EMA)", C1, "o"), ("fifo-310", "FIFO (310)", C2, "s"), ("reservoir-310", "Reservoir (310)", C3, "^"),
              ("keep-endings-310", "Keep-endings (310)", C4, "D")]
    for arm, label, col, mk in series:
        m, se = curve(data, arm)
        ax.fill_between(STEPS_K, m - se, m + se, color=col, alpha=0.15, linewidth=0)
        ax.plot(STEPS_K, m, color=col, lw=1.5, marker=mk, ms=3.5, label=label)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Greedy return")
    ax.set_ylim(0, 500)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5, ncol=2, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_curves.pdf")
    plt.close(fig)


def fig_gap(data: dict[str, dict[int, dict[str, Any]]]) -> None:
    """Paired difference fixed SOM minus FIFO of the same memory at each checkpoint, with 95% intervals.

    Args:
        data: Results by arm and seed.

    Returns:
        None. Saves ``fig_gap.pdf``.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.25, 1.9))
    for som, fifo, label, col, mk, dx in [("md-fixed-diag", "fifo-310", "Diagonal SOM $-$ FIFO (310)", C1, "o", -1.2),
                                                         ("md-fixed-full", "fifo-810", "Full-cov. SOM $-$ FIFO (810)", C2, "s", 1.2)]:
        st = [paired(data, som, fifo, str(c), rng) for c in CHECKPOINTS]
        m = np.array([s["mean"] for s in st])
        lo, hi = np.array([s["lo"] for s in st]), np.array([s["hi"] for s in st])
        x = STEPS_K + dx
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=3.5, lw=1.2, elinewidth=0.8, capsize=0, label=label)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Return difference")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_gap.pdf")
    plt.close(fig)


def fig_ablation(data: dict[str, dict[int, dict[str, Any]]]) -> None:
    """Whole-run paired difference of each SOM variant from the fixed diagonal SOM, with 95% intervals.

    Args:
        data: Results by arm and seed.

    Returns:
        None. Saves ``fig_ablation.pdf``.
    """
    rng = np.random.default_rng(0)
    rows = [("md-none-diag", "Uncorrected EMA"), ("md-exact-diag", "Exact running mean"), ("md-fixed-full", "Full covariance"),
            ("md-fixed-diag-count", "Units by count")]
    st = [paired(data, arm, "md-fixed-diag", "curve", rng) for arm, _ in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 1.5))
    m = np.array([s["mean"] for s in st])
    lo, hi = np.array([s["lo"] for s in st]), np.array([s["hi"] for s in st])
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="o", color=C1, ms=4, elinewidth=1.0, capsize=0)
    for yi, mi in zip(y, m):
        ax.text(mi, yi + 0.28, f"{mi:+.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(y, [r for _, r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Difference from the EMA SOM")
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_ablation.pdf")
    plt.close(fig)


def fig_alpha() -> None:
    """Whole-run and late scores of every arm of the alpha / ending-decoding study (mean +- SE over seeds 110-119).

    Returns:
        None. Saves ``fig_alpha.pdf``.
    """
    import experiments_alpha as A

    data = load(A.OUT)
    arms = [a for a in A.STUDY if a in data]
    labels = {"md-a0.99": r"$\alpha$ 0.99", "md-fixed-diag": r"$\alpha$ 0.9", "md-a0.5": r"$\alpha$ 0.5", "md-a0.1": r"$\alpha$ 0.1",
              "md-a0.01": r"$\alpha$ 0.01", "md-exact-diag": "exact", "md-fixed-diag-bern": r"$\alpha$ 0.9, Bern.",
              "md-exact-diag-bern": "exact, Bern.", "fifo-310": "FIFO (310)"}
    y = np.arange(len(arms))[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    for kind, label, col, mk, dy in (("curve", "Whole run", C1, "o", 0.14), ("late", r"Late ($\geq$100k)", C2, "s", -0.14)):
        v = [np.array([score(r, kind) for r in data[a].values()]) for a in arms]
        m = np.array([x.mean() for x in v])
        se = np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in v])
        ax.errorbar(m, y + dy, xerr=se, fmt=mk, color=col, ms=3.5, elinewidth=0.9, capsize=0, label=label)
    ax.set_yticks(y, [labels[a] for a in arms])
    ax.set_xlabel("Greedy return (mean $\\pm$ SE, 10 seeds)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_alpha.pdf")
    plt.close(fig)


def fig_replication() -> None:
    """Learning curves of the replication study (mean +- SE over seeds 120-139).

    Returns:
        None. Saves ``fig_replication.pdf``.
    """
    import experiments_replication as R

    data = load(R.OUT)
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    series = [("md-exact-diag-bern", "SOM, exact, Bernoulli end", C1, "o"), ("fifo-310", "FIFO (310)", C2, "s"),
              ("md-fixed-diag", r"SOM, $\alpha$ 0.9", C3, "^"), ("md-exact-diag", "SOM, exact, rounded end", C4, "D")]
    for arm, label, col, mk in series:
        m, se = curve(data, arm)
        ax.fill_between(STEPS_K, m - se, m + se, color=col, alpha=0.15, linewidth=0)
        ax.plot(STEPS_K, m, color=col, lw=1.5, marker=mk, ms=3.5, label=label)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Greedy return")
    ax.set_ylim(0, 500)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", fontsize=6, ncol=1, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_replication.pdf")
    plt.close(fig)


def fig_burnin() -> None:
    """Learning curves of Study 4 (burn-in and hybrid averaging; mean +- SE over seeds 140-159).

    Returns:
        None. Saves ``fig_burnin.pdf``.
    """
    import experiments_burnin as B

    data = load(B.OUT)
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    series = [("exact-bern", "Exact", C1, "o"), ("fifo-310", "FIFO (310)", C2, "s"), ("exact-bern-k20", "Exact, burn-in 20", C3, "^"),
              ("hybrid-bern", "Hybrid", C4, "D")]
    for arm, label, col, mk in series:
        m, se = curve(data, arm)
        ax.fill_between(STEPS_K, m - se, m + se, color=col, alpha=0.15, linewidth=0)
        ax.plot(STEPS_K, m, color=col, lw=1.5, marker=mk, ms=3.5, label=label)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Greedy return")
    ax.set_ylim(0, 500)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_burnin.pdf")
    plt.close(fig)


def fig_rate() -> None:
    """Study 5: score at 5k, whole-run and late scores per averaging rate (mean +- SE over seeds 160-179).

    Returns:
        None. Saves ``fig_rate.pdf``.
    """
    import experiments_burnin as B
    import experiments_rate as R

    data = load(R.OUT)
    arms = [a for a in R.STUDY if a in data]
    labels = {"rate-0.9": r"rate 0.9", "rate-0.5": r"rate 0.5", "rate-0.1": r"rate 0.1", "rate-0.03": r"rate 0.03",
              "hybrid-bern": r"rate 0.01", "exact-bern": "exact", "fast-early-exact": "0.9, then exact", "fifo-310": "FIFO (310)"}
    y = np.arange(len(arms))[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    for kind, label, col, mk, dy in (("5000", "At 5k", C3, "^", 0.22), ("curve", "Whole run", C1, "o", 0.0), ("late", r"Late ($\geq$100k)", C2, "s", -0.22)):
        v = [np.array([B.score(r, kind) for r in data[a].values()]) for a in arms]
        m = np.array([x.mean() for x in v])
        se = np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in v])
        ax.errorbar(m, y + dy, xerr=se, fmt=mk, color=col, ms=3.2, elinewidth=0.9, capsize=0, label=label)
    ax.set_yticks(y, [labels[a] for a in arms])
    ax.set_xlabel("Greedy return (mean $\\pm$ SE, 20 seeds)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.3, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_rate.pdf")
    plt.close(fig)


def main() -> None:
    """Build all figures.

    Returns:
        None.
    """
    FIG.mkdir(parents=True, exist_ok=True)
    data = load(OUT)
    fig_curves(data)
    fig_gap(data)
    fig_ablation(data)
    fig_alpha()
    fig_replication()
    fig_burnin()
    fig_rate()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
