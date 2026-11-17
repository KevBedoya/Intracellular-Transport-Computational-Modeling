"""M(5 t*) and M(t~), t~ = 5/Re(lambda_2), at N = 4 and N = 16 (v = 1000), in one 2x2 figure.

    June-2026/.venv/bin/python results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/scripts/heatmap_m_5t_star_t_tilde_5_N4_vs_N16.py

The two side-by-side figures together (heatmap_m_5t_star_N4_vs_N16.py, top row;
heatmap_m_t_tilde_5_N4_vs_N16.py, bottom row), with the same cells and limits.
Columns N = 4 (2026-11-08 run) and N = 16 (this run). Each row shares one colour
scale across N: log10 M for M(5 t*), M itself for M(t~).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heatmap_lambda_N4_vs_N16 import OUT, RES, arr, panel

N4 = os.path.join(RES, "2026-11-08_ab-grid-96x96_v1000_tau0.01")
RUNS = [(4, os.path.join(N4, "m_5t_star_96x96_v1000.csv"), os.path.join(N4, "m_t_tilde_96x96_v1000.csv")),
        (16, os.path.join(OUT, "m_5t_star_96x96_v1000_N16.csv"), os.path.join(OUT, "m_t_tilde_96x96_v1000_N16.csv"))]


def sci(x):
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


star = [(N, arr(pd.read_csv(p), "m_at_5_t_star"), arr(pd.read_csv(p), "five_t_star")) for N, p, _ in RUNS]
tilde = [(N, arr(pd.read_csv(p), "m_at_t_tilde_5_over_re"), arr(pd.read_csv(p), "t_tilde_5_over_re"))
         for N, _, p in RUNS]

# top row: log10 M(5 t*), integer limits over both N
zs = np.log10(np.concatenate([m for _, m, _ in star]))
lo_s, hi_s = np.floor(zs.min()), np.ceil(zs.max())
# bottom row: M(t~) linear (less than a decade), limits rounded out to the leading digit of the joint span
allm = np.concatenate([m for _, m, _ in tilde])
assert allm.max() / allm.min() < 10
step = 10.0 ** np.floor(np.log10(allm.max() - allm.min()))
lo_t, hi_t = np.floor(allm.min() / step) * step, np.ceil(allm.max() / step) * step
nd = max(3, int(-np.log10(step)) + 2)

fig, axes = plt.subplots(2, 2, figsize=(18.4, 15.4))
for c, ((N, m, t), (_, mt, tt)) in enumerate(zip(star, tilde)):
    panel(fig, axes[0, c], np.log10(m),
          [[(sci(m[i, j]), rf"$5t^*$ = {t[i, j]:.5f}") for j in range(4)] for i in range(4)],
          lo_s, hi_s, r"$\log_{10} M(5t^*)$",
          rf"$M(5t^*)$, N = {N}:  $t^*$ by the right sweep on the reference $y''$")
    panel(fig, axes[1, c], mt,
          [[(f"{mt[i, j]:.{nd}f}", rf"$\tilde{{t}}$ = {tt[i, j]:.5f}") for j in range(4)] for i in range(4)],
          lo_t, hi_t, r"$M(\tilde{t})$",
          rf"$M(\tilde{{t}})$, N = {N}:  $\tilde{{t}} = 5/\mathrm{{Re}}(\lambda_2)$ ($\lambda_2$ real for every pair)")
fig.suptitle(r"Mass retained per $(a, b)$ at $5t^*$ (top) and at $\tilde{t} = 5/\mathrm{Re}(\lambda_2)$ (bottom):  "
             r"N = 4 (left), N = 16 (right);  the time in each cell" "\n"
             r"96x96, v = 1000, N evenly spaced, $\tau$ = 0.01, power engine;  $M$ by exact powers of the "
             r"one-step operator;  each row has one colour scale for both N",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"heatmap_m_5t_star_and_t_tilde_5_N4_vs_N16_96x96_v1000.{ext}"), dpi=120)
print("written to", OUT)
