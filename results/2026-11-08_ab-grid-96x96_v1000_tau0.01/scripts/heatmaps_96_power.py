"""The 11-05 heatmaps for the v = 1000 grid: M(5 t*), M(t~) for the three t~, lambda_1, lambda_2.

    June-2026/.venv/bin/python results/2026-11-08_ab-grid-96x96_v1000_tau0.01/scripts/heatmaps_96_power.py

Reads the CSVs run_96_power.py wrote (summary, m_5t_star, m_t_tilde). Layout as
in 2026-11-05_ab-grid-96x96_modal_tau0.01/scripts/heatmap_m_5t_star.py: rows a,
columns b, one sequential blue ramp. M is coloured by log10 M when it spans a
decade or more across the grid and by M itself otherwise, on limits rounded out
to the leading digit of its span, with enough digits in each cell to tell the
pairs apart.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = [0.1, 1.0, 10.0, 100.0]
TAG = "v1000"
HEAD = r"96x96, v = 1000, N = 4 evenly spaced, $\tau$ = 0.01, power engine;  "
CMAP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def sci(x):
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def nice_limits(z):
    """Limits rounded out to the leading digit of the span, and that digit's step."""
    span = z.max() - z.min()
    step = 10.0 ** np.floor(np.log10(span)) if span > 0 else 1.0
    return np.floor(z.min() / step) * step, np.ceil(z.max() / step) * step, step


def grid(z, cells, lo, hi, cbar_label, title, stem):
    """A 4x4 (a, b) heatmap: rows a, columns b, colour z (4x4) on [lo, hi].

    ``cells[i][j]`` = (main, sub): the bold line and the smaller one under it
    (sub may be "", and main is then centred). Writes OUT/<stem>.png and .pdf.
    """
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, aspect="equal")
    for i in range(4):
        for j in range(4):
            main, sub = cells[i][j]
            ink = "white" if (z[i, j] - lo) / (hi - lo) > 0.5 else "#1a1a1a"
            ax.text(j, i - 0.13 if sub else i, main, ha="center", va="center",
                    color=ink, fontsize=11.5, fontweight="bold")
            if sub:
                ax.text(j, i + 0.2, sub, ha="center", va="center", color=ink, fontsize=9.5)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(4), [f"{b:g}" for b in AB])
    ax.set_yticks(range(4), [f"{a:g}" for a in AB])
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=150)
    plt.close(fig)


def arr(d, col):
    d = d.set_index(["a", "b"])
    return np.array([[d.loc[(a, b), col] for b in AB] for a in AB])


def heatmap(d, m_col, t_col, t_tex, subtitle, stem, t_def=""):
    """M at a per-pair time over the (a, b) grid; each cell shows M and the time."""
    m, t = arr(d, m_col), arr(d, t_col)
    log = m.max() / m.min() >= 10
    z = np.log10(m) if log else m
    if log:
        lo, hi = np.floor(z.min()), np.ceil(z.max())
        text = sci
    else:
        lo, hi, step = nice_limits(z)
        nd = max(3, int(-np.log10(step)) + 2)
        text = lambda x: f"{x:.{nd}f}"
    cells = [[(text(m[i, j]), rf"${t_tex}$ = {t[i, j]:.5f}") for j in range(4)] for i in range(4)]
    grid(z, cells, lo, hi, rf"$\log_{{10}} M({t_tex})$" if log else rf"$M({t_tex})$",
         rf"$M({t_tex})$ per $(a, b)$, with ${t_tex}$ in each cell" + (t_def and ";  " + t_def)
         + "\n" + HEAD + subtitle, stem)


summ = pd.read_csv(os.path.join(OUT, f"summary_96x96_{TAG}.csv"))
m5 = pd.read_csv(os.path.join(OUT, f"m_5t_star_96x96_{TAG}.csv"))
mt = pd.read_csv(os.path.join(OUT, f"m_t_tilde_96x96_{TAG}.csv"))
M_SRC = r"$M$ by exact powers of the one-step operator"

heatmap(m5, "m_at_5_t_star", "five_t_star", r"5t^*",
        r"$t^*$ from the reference $y''$" + "\n" + M_SRC, f"heatmap_m_5t_star_96x96_{TAG}")
for key, tex in (("5_over_re", r"5/\mathrm{Re}(\lambda_2)"), ("5_over_abs", r"5/|\lambda_2|"),
                 ("1_over_abs", r"1/|\lambda_2|")):
    heatmap(mt, f"m_at_t_tilde_{key}", f"t_tilde_{key}", r"\tilde{t}",
            r"$\lambda_2$ from the reduced operator's eigenvalues" + "\n" + M_SRC,
            f"heatmap_m_t_tilde_{key}_lambda2_96x96_{TAG}", t_def=rf"$\tilde{{t}} = {tex}$")

src = r"eigenvalues of the reduced one-step operator"
l1, re2, im2 = arr(summ, "lambda_1"), arr(summ, "lambda_2"), arr(summ, "lambda_2_imag")
grid(l1, [[(f"{l1[i, j]:.6f}", "") for j in range(4)] for i in range(4)],
     np.floor(l1.min()), np.ceil(l1.max()), r"$\lambda_1$",
     r"$\lambda_1$ per $(a, b)$: the slowest decay rate of $M$" + "\n" + HEAD + src,
     f"heatmap_lambda_1_96x96_{TAG}")
cells = [[(rf"{re2[i, j]:.3f} $\pm$ {im2[i, j]:.3f}$i$", rf"$|\lambda_2|$ = {np.hypot(re2[i, j], im2[i, j]):.4f}")
          if im2[i, j] > 0 else (f"{re2[i, j]:.6f}", "real")
          for j in range(4)] for i in range(4)]
grid(re2, cells, np.floor(re2.min()), np.ceil(re2.max()), r"$\mathrm{Re}(\lambda_2)$",
     r"$\lambda_2$ per $(a, b)$: the next excited decay rate; colour = $\mathrm{Re}(\lambda_2)$"
     + "\n" + HEAD + src, f"heatmap_lambda_2_96x96_{TAG}")
print("heatmaps written to", OUT)
