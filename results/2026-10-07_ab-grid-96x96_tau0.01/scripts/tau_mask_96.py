"""Right-sweep tau-mask t* on the 96x96 (a, b) grid, from job 0b8d4022's saved series.

    python tau_mask_96.py [tau]          default tau = 0.01

Job 0b8d4022: 96x96, v = 1, N_LIST = [0, 24, 48, 72], T = 1, GPU,
MA_collection_factor = 5. Its checkpoints keep each point's total-mass series
(timeseries_4.dat, float64, M itself, sample i at t = i * 5 * dT). No re-solve.

y = ln M; y'' is the second-order central difference
(y[i+s] - 2 y[i] + y[i-s]) / H^2 with H = s * h_sample as close as possible
to 1e-4 -- the same approximation as the 16x16 analysis. t* is
launch._char_time_onset (the committed criterion): the sample just after the
last |y''| > tau. The six pairs studied to T = 4 at 16x16 are excluded here
and left as blank panels so the layout matches the earlier 4x4 images.
"""
import os, sys
R = "N:/work/Intracellular-Transport-Computational-Modeling"
os.environ["MPLBACKEND"] = "Agg"
sys.path[:0] = [R + "/src/intracellular_transport", R + "/src"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from launch_functions import launch as L

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, "results", f"2026-10-07_ab-grid-96x96_tau{TAU:g}")
os.makedirs(OUT, exist_ok=True)

G, MCF, T = 96, 5, 1.0
A = B = [0.1, 1.0, 10.0, 100.0]
EXCLUDED = {(10.0, 0.1), (10.0, 1.0), (1.0, 0.1), (1.0, 1.0), (0.1, 0.1), (0.1, 1.0)}

h = MCF * L.num.compute_dT(G, G)
s = max(1, int(round(L.CHAR_TIME_DIFF_SPACING / h)))
H = s * h
print(f"sample h={h:.4g}  stride s={s}  H={H:.6g}  tau={TAU:g}")

data, rows = {}, []
for a in A:
    for b in B:
        if (a, b) in EXCLUDED:
            continue
        m = np.fromfile(os.path.join(SP, f"tm_a{a:g}_b{b:g}.dat"), dtype=np.float64)
        t = np.arange(m.size) * h
        y = np.log(m)
        q = np.full_like(y, np.nan)
        q[s:-s] = (y[2 * s:] - 2 * y[s:-s] + y[:-2 * s]) / H ** 2
        r = L._char_time_onset(m, h, tau=TAU)
        ts = None if np.isnan(r["t_star"]) else r["t_star"]
        keep = slice(0, None, s)                 # plot at the differencing spacing
        data[(a, b)] = (t[keep], y[keep], q[keep], ts)
        rows.append(dict(a=a, b=b, t_star=ts, m_star=None if ts is None else r["m_star"],
                         M_at_T=float(m[-1]), fit_slope_log10=None if ts is None else r["fit_slope"],
                         tau=TAU))
        print(f"  a={a:g} b={b:g}: t*={'none' if ts is None else f'{ts:.6f}'}", flush=True)

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, f"t_star_96x96_tau{TAU:g}.csv"), index=False)

# Table as an image.
fmt = lambda v, f: "none" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, f)
cells = [[f"{r['a']:g}", f"{r['b']:g}", fmt(r["t_star"], ".6f"), fmt(r["m_star"], ".6e"),
          f"{r['M_at_T']:.6e}"] for r in rows]
fig, ax = plt.subplots(figsize=(9, 0.42 * len(cells) + 1.4))
ax.axis("off")
tb = ax.table(cellText=cells, colLabels=["a", "b", "t*", "m* = M(t*)", "M(T=1)"],
              loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1, 1.4)
ax.set_title(rf"t* by right-sweep mask, $\tau$ = {TAU:g}  --  96x96, v=1, N=4, T=1 "
             f"(job 0b8d4022), y'' step H={H:.3g}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"t_star_table_96x96_tau{TAU:g}.png"), dpi=130)
plt.close(fig)


def title(k):
    ts = data[k][3]
    return f"a={k[0]:g}, b={k[1]:g}  t*=" + ("none" if ts is None else f"{ts:.4f}")


def lnm_panel(ax, k, lim):
    t, y, _, ts = data[k]
    ax.plot(t, y, lw=1.2, color="0.25")
    if ts is not None:
        ax.axvline(ts, color="red", lw=1.2)


def lnm_zoom_panel(ax, k, clip):
    """ln M windowed to t* +- 0.15 and ln M(t*) +- 0.5; clip stops the x-window at [0, T]."""
    t, y, _, ts = data[k]
    lnm_panel(ax, k, None)
    if ts is None:
        ax.text(0.5, 0.92, "no t*: full view", ha="center", va="top", fontsize=8,
                color="0.4", transform=ax.transAxes)
        return
    yc = float(np.interp(ts, t, y))
    lo, hi = ts - 0.15, ts + 0.15
    if clip:
        lo, hi = max(lo, 0.0), min(hi, T)
    ax.set_xlim(lo, hi)
    ax.set_ylim(yc - 0.5, yc + 0.5)
    ax.axhline(yc, color="red", lw=0.6, ls=":")


def d2_panel(ax, k, lim):
    t, _, q, ts = data[k]
    ax.plot(t, q, lw=1.0, color="C2")
    ax.axhline(0, color="k", lw=0.6)
    if ts is not None:
        ax.axvline(ts, color="red", lw=1.2)
    if lim is not None:
        ax.set_ylim(-lim, lim)
        for v in (-TAU, TAU):
            ax.axhline(v, color="0.4", lw=0.8, ls=":")


NOTE = (rf"red: t* = sample after the last $|y''| > \tau$ = {TAU:g} (right sweep);  "
        rf"$y''$ = central difference of $\ln M$, H = {H:.3g};  "
        r"blank: pairs studied to T = 4 at 16x16, excluded")
figs = [(f"lnM_96x96_tau{TAU:g}.png", r"$\ln M(t)$", lnm_panel, None, ""),
        (f"second_derivative_96x96_tau{TAU:g}.png", r"$d^2\ln M/dt^2$", d2_panel, None, ""),
        (f"lnM_96x96_tau{TAU:g}_zoom.png", r"$\ln M(t)$", lnm_zoom_panel, False,
         r"  (window: t* $\pm$ 0.15, $\ln M(t^*)$ $\pm$ 0.5; dotted = $\ln M(t^*)$)"),
        (f"lnM_96x96_tau{TAU:g}_zoom_clipped.png", r"$\ln M(t)$", lnm_zoom_panel, True,
         r"  (window: t* $\pm$ 0.15 clipped to [0, T], $\ln M(t^*)$ $\pm$ 0.5)"),
        (f"second_derivative_96x96_tau{TAU:g}_ylim0.3.png", r"$d^2\ln M/dt^2$", d2_panel, 0.3,
         r"  (y-window $\pm$0.3; dotted = $\pm\tau$)"),
        (f"second_derivative_96x96_tau{TAU:g}_ylim0.2.png", r"$d^2\ln M/dt^2$", d2_panel, 0.2,
         r"  (y-window $\pm$0.2; dotted = $\pm\tau$)"),
        (f"second_derivative_96x96_tau{TAU:g}_ylim0.1.png", r"$d^2\ln M/dt^2$", d2_panel, 0.1,
         r"  (y-window $\pm$0.1; dotted = $\pm\tau$)")]
for name, ylab, panel, lim, extra in figs:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=(panel is not lnm_zoom_panel))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            if (a, b) in EXCLUDED:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"a={a:g}, b={b:g}\nexcluded\n(T = 4 pair)",
                        ha="center", va="center", fontsize=9, color="0.5",
                        transform=ax.transAxes)
                continue
            ax.set_xlim(0, T)
            panel(ax, (a, b), lim)
            ax.grid(True, alpha=0.3)
            ax.set_title(title((a, b)), fontsize=9)
            if i == 3:
                ax.set_xlabel("t")
            if j == 0 or (i, j - 1) in [(A.index(x), B.index(y)) for x, y in EXCLUDED]:
                ax.set_ylabel(ylab)
            ax.tick_params(labelbottom=True)
    fig.suptitle(f"{ylab}, 96x96, v=1, N=4, T=1{extra}\n{NOTE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)
print(OUT)
