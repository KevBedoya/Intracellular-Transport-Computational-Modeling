"""lambda_1 and lambda_2 over (a, b) at N = 4 and N = 16 (v = 1000), as one 2x2 figure.

    June-2026/.venv/bin/python results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/scripts/heatmap_lambda_N4_vs_N16.py

Rows N = 4 (2026-11-08 run) and N = 16 (this run); columns lambda_1 and lambda_2.
Each panel is the heatmap_lambda_{1,2} table of its run, unchanged: same cells,
same colour ramp, same limits (each panel keeps its own scale).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.dirname(OUT)
AB = [0.1, 1.0, 10.0, 100.0]
RUNS = [(4, os.path.join(RES, "2026-11-08_ab-grid-96x96_v1000_tau0.01", "summary_96x96_v1000.csv")),
        (16, os.path.join(OUT, "summary_96x96_v1000_N16.csv"))]
CMAP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def arr(d, col):
    d = d.set_index(["a", "b"])
    return np.array([[d.loc[(a, b), col] for b in AB] for a in AB])


def panel(fig, ax, z, cells, lo, hi, cbar_label, title):
    """One 4x4 (a, b) table, drawn exactly as heatmaps_96_power.grid draws it."""
    im = ax.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, aspect="equal")
    for i in range(4):
        for j in range(4):
            main, sub = cells[i][j]
            ink = "white" if (z[i, j] - lo) / (hi - lo) > 0.5 else "#1a1a1a"
            ax.text(j, i - 0.13 if sub else i, main, ha="center", va="center",
                    color=ink, fontsize=11.5, fontweight="bold")
            if sub:
                ax.text(j, i + 0.2, sub, ha="center", va="center", color=ink, fontsize=9.5)
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
    ax.set_title(title, fontsize=12)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(18.4, 15.2))
    for r, (N, path) in enumerate(RUNS):
        s = pd.read_csv(path)
        l1, re2, im2 = arr(s, "lambda_1"), arr(s, "lambda_2"), arr(s, "lambda_2_imag")
        panel(fig, axes[r, 0], l1, [[(f"{l1[i, j]:.6f}", "") for j in range(4)] for i in range(4)],
              np.floor(l1.min()), np.ceil(l1.max()), r"$\lambda_1$",
              rf"$\lambda_1$, N = {N}: the slowest decay rate of $M$")
        cells = [[(rf"{re2[i, j]:.3f} $\pm$ {im2[i, j]:.3f}$i$",
                   rf"$|\lambda_2|$ = {np.hypot(re2[i, j], im2[i, j]):.4f}")
                  if im2[i, j] > 0 else (f"{re2[i, j]:.6f}", "real")
                  for j in range(4)] for i in range(4)]
        panel(fig, axes[r, 1], re2, cells, np.floor(re2.min()), np.ceil(re2.max()),
              r"$\mathrm{Re}(\lambda_2)$",
              rf"$\lambda_2$, N = {N}: the next excited decay rate; colour = $\mathrm{{Re}}(\lambda_2)$")
    fig.suptitle(r"$\lambda_1$ and $\lambda_2$ per $(a, b)$ at N = 4 (top) and N = 16 (bottom)" "\n"
                 r"96x96, v = 1000, N evenly spaced, $\tau$ = 0.01, power engine;  "
                 r"eigenvalues of the reduced one-step operator;  each panel has its own colour scale",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"heatmap_lambda_1_2_N4_vs_N16_96x96_v1000.{ext}"), dpi=120)
    print("written to", OUT)


if __name__ == "__main__":
    main()
