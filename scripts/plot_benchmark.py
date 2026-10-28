# file: scripts/plot_benchmark.py
r"""Two-panel benchmark figure: time per inverse and residual against $d$.

Usage
-----
From the rows returned by the benchmark::

    from scripts.benchmark_inverse import time_inverse
    from scripts.plot_benchmark import plot_rows
    plot_rows(time_inverse(), "benchmark_inverse.pdf")

Run as a script, it plots the stored run below (2026-10-25)::

    python scripts/plot_benchmark.py

The figure is sized for the birkjour text width and saved as a vector PDF,
ready for ``\includegraphics[width=\textwidth]{benchmark_inverse}``.  The
figure is grayscale by design: each case has its own line style and marker.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Grayscale by design: the tau-FLS cases in black, comparisons in gray.
# Line style and marker each identify a case on their own.
# case -> (label, ink, line style, marker, filled); the order is the legend order
STYLE = {
    "tau-FLS":    (r"$\uptau$-FLS",      "#000000", "-",                 "o", True),
    "tau-FLS+NS": (r"$\uptau$-FLS + NS", "#000000", (0, (4, 2)),        "o", False),
    "model":      ("model",              "#4d4d4d", "-",                 "D", True),
    "native FLS": ("native FLS",         "#808080", (0, (1, 1.5)),       "^", True),
    "pygae":      ("pygae",              "#808080", (0, (5, 1.5, 1, 1.5)), "v", True),
}

# Stored run, Cl(d, 0), 2026-10-25: seconds and residual per case, d = 4..11
DIMS = list(range(4, 12))
STORED = {
    "model":      ([21.17e-6, 42.12e-6, 72.40e-6, 212.1e-6, 0.7262e-3, 2.932e-3, 16.89e-3, 82.98e-3],
                   [4.4e-16, 5.6e-16, 2.9e-14, 1.0e-14, 1.8e-13, 7.4e-12, 2.8e-10, 2.1e-10]),
    "tau-FLS":    ([7.135e-6, 8.867e-6, 15.47e-6, 36.94e-6, 0.1480e-3, 0.6636e-3, 6.192e-3, 28.14e-3],
                   [2.2e-16, 7.8e-16, 1.5e-14, 1.7e-15, 1.6e-14, 7.3e-13, 5.0e-11, 5.0e-11]),
    "tau-FLS+NS": ([24.21e-6, 29.07e-6, 38.86e-6, 77.18e-6, 0.2358e-3, 1.478e-3, 8.843e-3, 39.08e-3],
                   [2.2e-16, 7.8e-16, 1.5e-14, 1.7e-15, 1.6e-14, 1.1e-14, 1.3e-15, 8.2e-16]),
    "native FLS": ([21.27e-6, 53.06e-6, 84.15e-6, 384.3e-6, 1.111e-3, 7.818e-3, 30.39e-3, 266.8e-3],
                   [2.2e-16, 1.3e-15, 3.1e-15, 3.2e-12, 5.7e-12, 1.6e-05, 9.8e-08, 1.1e-01]),
    "pygae":      ([21.52e-6, 56.97e-6, 142.9e-6, 561.2e-6, 4.395e-3, 26.52e-3, 181.2e-3, 1044e-3],
                   [2.2e-16, 6.3e-16, 1.1e-15, 7.3e-15, 8.7e-15, 1.8e-12, 2.9e-14, 2.5e-14]),
}

TEXT_W = 4.9        # inches; birkjour text width is about 12.5 cm
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _series(rows):
    """rows from time_inverse -> {case: (dims, seconds, residuals)}."""
    out = {}
    for case in STYLE:
        rs = sorted((r for r in rows if r["case"] == case), key=lambda r: r["d"])
        if rs:
            out[case] = ([r["d"] for r in rs], [r["seconds"] for r in rs],
                         [r["residual"] for r in rs])
    return out


def plot_rows(rows, path="benchmark_inverse.pdf"):
    data = _series(rows)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "text.usetex": False,
    })
    # mathtext has no \uptau; it sets \tau, which is close enough on screen
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(TEXT_W, 2.55), constrained_layout=True)

    for case, (ds, sec, res) in data.items():
        label, ink, ls, marker, filled = STYLE[case]
        label = label.replace(r"\uptau", r"\tau")
        kw = dict(color=ink, linestyle=ls, marker=marker, markersize=4, linewidth=1.0,
                  markerfacecolor=ink if filled else "white", markeredgecolor=ink,
                  markeredgewidth=0.8, label=label)
        ax_t.plot(ds, sec, **kw)
        ax_r.plot(ds, res, **kw)

    for ax in (ax_t, ax_r):
        ax.set_yscale("log")
        ax.set_xlabel("dimension $d$")
        ax.set_xticks(DIMS)
        ax.grid(True, which="major", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax_t.set_ylabel("time per inverse (s)")
    ax_t.set_title("Time", fontsize=8, loc="left", color=INK)
    ax_r.set_ylabel(r"residual $\max\,|AA^{-1} - 1|$")
    ax_r.set_title("Accuracy", fontsize=8, loc="left", color=INK)
    h, l = ax_t.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(l), fontsize=7,
               handlelength=1.6, columnspacing=1.0)

    fig.savefig(path)
    if path.endswith(".pdf"):
        fig.savefig(path[:-4] + ".png", dpi=200)
    return fig


def stored_rows():
    return [dict(d=d, case=c, seconds=s, residual=r)
            for c, (secs, ress) in STORED.items()
            for d, s, r in zip(DIMS, secs, ress)]


if __name__ == "__main__":
    plot_rows(stored_rows(), sys.argv[1] if len(sys.argv) > 1 else "benchmark_inverse.pdf")
