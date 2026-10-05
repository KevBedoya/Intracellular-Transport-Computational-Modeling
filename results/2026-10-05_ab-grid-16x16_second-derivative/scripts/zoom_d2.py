"""Zoomed copies of the central-difference second-derivative grids: y in [-1, max].

    python zoom_d2.py <grid>          reads the curves/roots CSVs grid_plots.py wrote

max is the panel's own maximum of d2 ln M/dt2, or 0 if the curve never turns
positive, with a 5% margin on both ends. Same curves and red first-root lines as the
unzoomed grid; no re-solve.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = int(sys.argv[1])
OUT = (f"N:/work/Intracellular-Transport-Computational-Modeling/results/"
       f"2026-10-05_ab-grid-{G}x{G}_second-derivative")
A = B = [0.1, 1.0, 10.0, 100.0]
LO = -1.0

d = pd.read_csv(os.path.join(OUT, f"curves_{G}x{G}.csv"))
roots = pd.read_csv(os.path.join(OUT, f"first_root_t_star_{G}x{G}.csv"))
t = d["t"].to_numpy()
info = {(r.a, r.b): r for r in roots.itertuples()}


def ylim(*curves):
    top = max(0.0, max(np.nanmax(c) for c in curves))
    pad = 0.05 * (top - LO)
    return LO - pad, top + pad


fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
for i, a in enumerate(A):
    for j, b in enumerate(B):
        ax = axes[i, j]
        q = d[f"d2lnM_a{a:g}_b{b:g}"].to_numpy()
        ax.plot(t, q, lw=1.0, color="C2")
        ax.axhline(0, color="k", lw=0.6)
        r = info[(a, b)]
        if not pd.isna(r.t_star_first_root):
            ax.axvline(r.t_star_first_root, color="red", lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim(q))
        ax.grid(True, alpha=0.3)
        root = "none" if pd.isna(r.t_star_first_root) else f"{r.t_star_first_root:.4f}"
        ax.set_title(f"a={a:g}, b={b:g}  root={root}  max={np.nanmax(q):.3g}", fontsize=9)
        if i == 3:
            ax.set_xlabel("t")
        if j == 0:
            ax.set_ylabel(r"$d^2 \ln M/dt^2$")
fig.suptitle(rf"$d^2 \ln M/dt^2$, {G}x{G}, v=1, N=4, T=1, y-window [{LO}, max]   "
             r"(red = first root of $d^2 \ln M/dt^2$)", fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, f"second_derivative_grid_{G}x{G}_zoom.png")
fig.savefig(path, dpi=110)
plt.close(fig)
print(path)
