"""The kappa criterion on the 96x96 (a, b) grid, from job 0b8d4022's saved series.

    python kappa_96.py [tau]          default tau = 0.01

kappa(t) = |y''| / y'^2 with y = ln M: the rate at which the local e-folding
time 1/lambda drifts (lambda = -y'). Same derivative approximation as the rest
of this study -- central differences across s samples, H = s * h_sample ~ 1e-4:

    y'  = (y[i+s] - y[i-s]) / (2H),   y'' = (y[i+s] - 2 y[i] + y[i-s]) / H^2

and the same right sweep: t* is the sample just after the last kappa > tau,
so kappa <= tau from t* to the end. (The criterion as used 2026-09-28 to
2026-10-05 also required at least 0.1 of run after t*; that is reported here as
a column, not applied.) All sixteen pairs, T = 1.
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
OUT = os.path.join(R, "results", f"2026-10-07_ab-grid-96x96_tau{TAU:g}", "kappa_criterion")
os.makedirs(OUT, exist_ok=True)

G, MCF, T, HOLD = 96, 5, 1.0, 0.1
A = B = [0.1, 1.0, 10.0, 100.0]
h = MCF * L.num.compute_dT(G, G)
s = max(1, int(round(L.CHAR_TIME_DIFF_SPACING / h)))
H = s * h
print(f"sample h={h:.4g}  stride s={s}  H={H:.6g}  tau={TAU:g}")

data, rows = {}, []
for a in A:
    for b in B:
        m = np.fromfile(os.path.join(SP, f"tm_a{a:g}_b{b:g}.dat"), dtype=np.float64)
        n = m.size
        t = np.arange(n) * h
        y = np.log(m)
        d1 = (y[2 * s:] - y[:-2 * s]) / (2 * H)
        d2 = (y[2 * s:] - 2 * y[s:-s] + y[:-2 * s]) / H ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = np.abs(d2) / d1 ** 2
        judged = np.arange(s, n - s)
        over = ~(kap <= TAU)                 # non-finite (y' = 0 at the start) counts as over
        if over[-1]:
            ts = None
        else:
            k = judged[np.nonzero(over)[0][-1] + 1] if over.any() else judged[0]
            ts = float(t[k])
        mstar = None if ts is None else float(m[int(round(ts / h))])
        full = np.full(n, np.nan)
        full[s:-s] = kap
        keep = slice(0, None, s)
        data[(a, b)] = (t[keep], y[keep], full[keep], ts)
        rows.append(dict(a=a, b=b, t_star=ts, m_star=mstar, M_at_T=float(m[-1]),
                         kappa_at_T=float(kap[-1]),
                         hold_ge_0p1=None if ts is None else bool(T - ts >= HOLD),
                         tau=TAU))
        print(f"  a={a:g} b={b:g}: t*={'none' if ts is None else f'{ts:.6f}'}  "
              f"kappa(T)={kap[-1]:.3g}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, f"t_star_kappa_96x96_tau{TAU:g}.csv"), index=False)

fmt = lambda v, f: "none" if v is None else format(v, f)
cells = [[f"{r['a']:g}", f"{r['b']:g}", fmt(r["t_star"], ".6f"), fmt(r["m_star"], ".6e"),
          f"{r['M_at_T']:.6e}", f"{r['kappa_at_T']:.3g}",
          "-" if r["hold_ge_0p1"] is None else ("yes" if r["hold_ge_0p1"] else "no")]
         for r in rows]
fig, ax = plt.subplots(figsize=(11, 0.4 * len(cells) + 1.4))
ax.axis("off")
tb = ax.table(cellText=cells, colLabels=["a", "b", "t*", "m* = M(t*)", "M(T=1)",
                                          "kappa(T)", ">= 0.1 run after t*"],
              loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1, 1.35)
ax.set_title(rf"t* by $\kappa = |y''|/y'^2 \leq \tau$ = {TAU:g} (right sweep) -- 96x96, v=1, "
             f"N=4, T=1 (job 0b8d4022), H={H:.3g}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"t_star_table_kappa_96x96_tau{TAU:g}.png"), dpi=130)
plt.close(fig)


def title(k):
    ts = data[k][3]
    return f"a={k[0]:g}, b={k[1]:g}  t*=" + ("none" if ts is None else f"{ts:.4f}")


def mark(ax, k):
    if data[k][3] is not None:
        ax.axvline(data[k][3], color="red", lw=1.2)


def step_label(ax):
    """The finite-difference step, in the panel corner."""
    ax.text(0.98, 0.04, f"h = {H:.4e}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.2",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))


STEP_LINE = (rf"finite-difference step h = {H:.6e}  (= {s} samples $\times$ {h:.6e}, the "
             rf"solver's sample spacing $5\,\Delta t$)")


def kappa_panel(ax, k, top):
    t, _, kap, _ = data[k]
    ax.plot(t, kap, lw=1.0, color="C1")
    ax.axhline(TAU, color="0.3", lw=0.9, ls=":")
    ax.set_ylim(0, top)
    mark(ax, k)
    step_label(ax)


def lnm_panel(ax, k, zoom):
    t, y, _, ts = data[k]
    ax.plot(t, y, lw=1.2, color="0.25")
    mark(ax, k)
    if zoom and ts is not None:
        yc = float(np.interp(ts, t, y))
        ax.set_xlim(ts - 0.15, ts + 0.15)
        ax.set_ylim(yc - 0.5, yc + 0.5)
        ax.axhline(yc, color="red", lw=0.6, ls=":")
    elif zoom:
        ax.text(0.5, 0.92, "no t*: full view", ha="center", va="top", fontsize=8,
                color="0.4", transform=ax.transAxes)


NOTE = (rf"red: t* = sample after the last $\kappa > \tau$ = {TAU:g} (right sweep);  "
        rf"$\kappa = |y''|/y'^2$, $y = \ln M$, central differences, H = {H:.3g};  all 16 pairs")
figs = [(f"kappa_96x96_tau{TAU:g}_ylim0.1.png", r"$\kappa(t)$",
         lambda ax, k: kappa_panel(ax, k, 0.1), r"  (y in [0, 0.1]; dotted = $\tau$)", True),
        (f"kappa_96x96_tau{TAU:g}_ylim{3 * TAU:g}.png", r"$\kappa(t)$",
         lambda ax, k: kappa_panel(ax, k, 3 * TAU), rf"  (y in [0, {3 * TAU:g}]; dotted = $\tau$)", True),
        (f"lnM_kappa_96x96_tau{TAU:g}.png", r"$\ln M(t)$",
         lambda ax, k: lnm_panel(ax, k, False), "", True),
        (f"lnM_kappa_96x96_tau{TAU:g}_zoom.png", r"$\ln M(t)$",
         lambda ax, k: lnm_panel(ax, k, True),
         r"  (window: t* $\pm$ 0.15, $\ln M(t^*)$ $\pm$ 0.5)", False)]
for name, ylab, panel, extra, share in figs:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=share)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            ax.set_xlim(0, T)
            panel(ax, (a, b))
            ax.grid(True, alpha=0.3)
            ax.set_title(title((a, b)), fontsize=9)
            if i == 3 or not share:
                ax.set_xlabel("t")
            if j == 0:
                ax.set_ylabel(ylab)
    head = f"{ylab}, 96x96, v=1, N=4, T=1{extra}"
    if "kappa_96x96" in name:
        head += "\n" + STEP_LINE
    fig.suptitle(f"{head}\n{NOTE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)
print(OUT)
