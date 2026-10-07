"""The original line-intercept t* on the 96x96 (a, b) grid, from job 0b8d4022's saved series.

The pre-2026-09-28 definition: a line through log10 M at t = 0.4 and t = 0.5
(the samples the old code picked: closest_multiple(compute_K(rg, ry, x), MCF)
// MCF), extended to where it reaches log10 M = 0; t* = -intercept / slope.
The old m* was M(10 t*). Plotted in ln M like the rest of the folder -- the
line reaches ln M = 0 at the same t.

Same ten T = 1 pairs as the tau = 0.01 analysis; the six T = 4 pairs are blank.
"""
import os, sys
R = "N:/work/Intracellular-Transport-Computational-Modeling"
os.environ["MPLBACKEND"] = "Agg"
sys.path[:0] = [R + "/src/intracellular_transport", R + "/src"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from launch_functions import launch as L

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, "results", "2026-10-07_ab-grid-96x96_tau0.01", "line_intercept_criterion")
os.makedirs(OUT, exist_ok=True)

G, MCF, T = 96, 5, 1.0
X1, X2 = 0.4, 0.5
A = B = [0.1, 1.0, 10.0, 100.0]
EXCLUDED = {(10.0, 0.1), (10.0, 1.0), (1.0, 0.1), (1.0, 1.0), (0.1, 0.1), (0.1, 1.0)}

h = MCF * L.num.compute_dT(G, G)
i1 = L.num.closest_multiple(L.num.compute_K(G, G, X1), MCF) // MCF
i2 = L.num.closest_multiple(L.num.compute_K(G, G, X2), MCF) // MCF
stride = 861                                    # plot spacing, ~1e-4

data, rows = {}, []
for a in A:
    for b in B:
        if (a, b) in EXCLUDED:
            continue
        m = np.fromfile(os.path.join(SP, f"tm_a{a:g}_b{b:g}.dat"), dtype=np.float64)
        t = np.arange(m.size) * h
        y1, y2 = np.log10(m[i1]), np.log10(m[i2])
        slope = (y2 - y1) / (X2 - X1)
        icpt = y2 - slope * X2
        ts = -icpt / slope
        k10 = L.num.closest_multiple(L.num.compute_K(G, G, 10 * ts), MCF) // MCF if ts > 0 else -1
        m10 = float(m[k10]) if 0 <= k10 < m.size else float("nan")
        mts = float(np.exp(np.interp(ts, t, np.log(m)))) if 0 <= ts <= t[-1] else float("nan")
        data[(a, b)] = (t[::stride], np.log(m[::stride]), ts, slope, icpt)
        rows.append(dict(a=a, b=b, t_star_intercept=ts, slope_log10=slope,
                         intercept_log10=icpt, M_at_0p4=float(m[i1]), M_at_0p5=float(m[i2]),
                         M_at_t_star=mts, m_star_old_M_at_10t_star=m10))
        print(f"a={a:g} b={b:g}: t*={ts:.6f}  slope={slope:.4f}  M(10t*)={m10:.4g}")

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "t_star_line_intercept_96x96.csv"), index=False)


def title(k):
    ts = data[k][2]
    return f"a={k[0]:g}, b={k[1]:g}  t*={ts:.4f}"


def panel(ax, k, zoom):
    t, y, ts, slope, icpt = data[k]
    ax.plot(t, y, lw=1.2, color="0.25", label=r"$\ln M$")
    # The fitted line, in ln M units: ln M = ln(10) * (icpt + slope * t).
    tl = np.linspace(min(0.0, ts) - 0.05, T, 400)
    ax.plot(tl, np.log(10) * (icpt + slope * tl), lw=1.0, ls="--", color="C0",
            label="line through t=0.4, 0.5")
    ax.plot([X1, X2], np.log(10) * (icpt + slope * np.array([X1, X2])), "o", ms=4,
            color="C0")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(ts, color="red", lw=1.2)
    if zoom:
        if 0 <= ts <= T:
            ax.set_xlim(ts - 0.15, ts + 0.15)
            ax.set_ylim(-0.5, 0.5)
        else:
            ax.set_xlim(min(0.0, ts) - 0.05, T)
            ax.text(0.5, 0.06, "t* outside [0, 1]: full view", ha="center", fontsize=8,
                    color="0.4", transform=ax.transAxes)
    else:
        ax.set_xlim(min(0.0, ts) - 0.05, T)


for name, zoom, extra in [("lnM_line_intercept_96x96.png", False, ""),
                          ("lnM_line_intercept_96x96_zoom.png", True,
                           r"  (window: t* $\pm$ 0.15, $\ln M$ in [-0.5, 0.5])")]:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            if (a, b) in EXCLUDED:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"a={a:g}, b={b:g}\nexcluded\n(T = 4 pair)", ha="center",
                        va="center", fontsize=9, color="0.5", transform=ax.transAxes)
                continue
            panel(ax, (a, b), zoom)
            ax.grid(True, alpha=0.3)
            ax.set_title(title((a, b)), fontsize=9)
            ax.set_xlabel("t")
            ax.set_ylabel(r"$\ln M(t)$")
    axes[3, 3].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"ln M(t), 96x96, v=1, N=4, T=1{extra}\n"
                 r"original criterion: line through $\log_{10}M$ at t = 0.4, 0.5 (blue dashed, "
                 r"points); red: t* where it reaches $M$ = 1 ($\ln M$ = 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)
print(OUT)
