"""The mass retained M(5 t*) over (a, b) at N = 4 and N = 16 (v = 1000): two tables side by side.

    June-2026/.venv/bin/python results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/scripts/heatmap_m_5t_star_N4_vs_N16.py

Sweep-method results only: t* is the right sweep (the last crossing of
|y''| = tau on the reference y''), and M(5 t*) the mass left at 5 t* -- the
numbers of t_star_m_5t_star_96x96_v1000_N16.pdf and of the N = 4 table, from the
m_5t_star CSVs. Left N = 4 (2026-11-08 run), right N = 16 (this run). Both
panels share one log10 colour scale, so equal shades mean equal M.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heatmap_lambda_N4_vs_N16 import OUT, RES, arr, panel

RUNS = [(4, os.path.join(RES, "2026-11-08_ab-grid-96x96_v1000_tau0.01", "m_5t_star_96x96_v1000.csv")),
        (16, os.path.join(OUT, "m_5t_star_96x96_v1000_N16.csv"))]


def sci(x):
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


data = [(N, arr(pd.read_csv(p), "m_at_5_t_star"), arr(pd.read_csv(p), "five_t_star")) for N, p in RUNS]
zs = np.log10(np.concatenate([m for _, m, _ in data]))
lo, hi = np.floor(zs.min()), np.ceil(zs.max())

fig, axes = plt.subplots(1, 2, figsize=(18.4, 7.9))
for ax, (N, m5, t5) in zip(axes, data):
    panel(fig, ax, np.log10(m5),
          [[(sci(m5[i, j]), rf"$5t^*$ = {t5[i, j]:.5f}") for j in range(4)] for i in range(4)],
          lo, hi, r"$\log_{10} M(5t^*)$", rf"N = {N}")
fig.suptitle(r"Mass retained $M(5t^*)$ per $(a, b)$, with $5t^*$ in each cell:  N = 4 (left), N = 16 (right)" "\n"
             r"96x96, v = 1000, N evenly spaced, $\tau$ = 0.01, power engine;  $t^*$ by the right sweep on the "
             r"reference $y''$;  $M$ by exact powers of the one-step operator;  one colour scale for both",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"heatmap_m_5t_star_N4_vs_N16_96x96_v1000.{ext}"), dpi=120)
print("written to", OUT)
