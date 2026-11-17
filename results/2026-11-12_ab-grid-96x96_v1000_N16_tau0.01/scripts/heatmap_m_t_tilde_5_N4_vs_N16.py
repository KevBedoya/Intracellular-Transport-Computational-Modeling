"""M(t~), t~ = 5/Re(lambda_2), over (a, b) at N = 4 and N = 16 (v = 1000): two tables side by side.

    June-2026/.venv/bin/python results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/scripts/heatmap_m_t_tilde_5_N4_vs_N16.py

From the m_t_tilde CSVs (column m_at_t_tilde_5_over_re). lambda_2 is real for
every pair at both N, so t~ = 5/Re(lambda_2) = 5/|lambda_2| and this is also the
5_over_abs map. Left N = 4 (2026-11-08 run), right N = 16 (this run). Both
panels share one colour scale. M spans less than a decade, so it is coloured by M
itself, on limits rounded out to the leading digit of the joint span, with the
cell digits of heatmaps_96_power.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heatmap_lambda_N4_vs_N16 import OUT, RES, arr, panel

RUNS = [(4, os.path.join(RES, "2026-11-08_ab-grid-96x96_v1000_tau0.01", "m_t_tilde_96x96_v1000.csv")),
        (16, os.path.join(OUT, "m_t_tilde_96x96_v1000_N16.csv"))]

data = [(N, arr(pd.read_csv(p), "m_at_t_tilde_5_over_re"), arr(pd.read_csv(p), "t_tilde_5_over_re"))
        for N, p in RUNS]
allm = np.concatenate([m for _, m, _ in data])
assert allm.max() / allm.min() < 10                  # linear colour, as heatmaps_96_power decides
span = allm.max() - allm.min()
step = 10.0 ** np.floor(np.log10(span))
lo, hi = np.floor(allm.min() / step) * step, np.ceil(allm.max() / step) * step
nd = max(3, int(-np.log10(step)) + 2)

fig, axes = plt.subplots(1, 2, figsize=(18.4, 7.9))
for ax, (N, m, t) in zip(axes, data):
    panel(fig, ax, m,
          [[(f"{m[i, j]:.{nd}f}", rf"$\tilde{{t}}$ = {t[i, j]:.5f}") for j in range(4)] for i in range(4)],
          lo, hi, r"$M(\tilde{t})$", rf"N = {N}")
fig.suptitle(r"$M(\tilde{t})$ per $(a, b)$, $\tilde{t} = 5/\mathrm{Re}(\lambda_2)$, with $\tilde{t}$ in each cell:  "
             r"N = 4 (left), N = 16 (right)" "\n"
             r"96x96, v = 1000, N evenly spaced, $\tau$ = 0.01, power engine;  $\lambda_2$ from the reduced "
             r"operator's eigenvalues (real for every pair);  $M$ by exact powers of the one-step operator;  "
             r"one colour scale for both",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"heatmap_m_t_tilde_5_over_re_lambda2_N4_vs_N16_96x96_v1000.{ext}"), dpi=120)
print("written to", OUT)
