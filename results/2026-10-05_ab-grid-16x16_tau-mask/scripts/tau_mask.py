"""Right-sweep threshold mask on y'' = d2 ln M/dt2, 16x16 (a, b) grid.

    python tau_mask.py [tau]          default tau = 1e-3

l(t) = 1 if |y''(t)| <= tau else 0. Sweeping from the right, t^ is the last
sample with l = 0 (the largest t where |y''| > tau), and t* is the sample
immediately after it: |y''| <= tau from t* to the end of the run. No
interpolation -- t* is a sample time (spacing h = 1.5e-4).

The six pairs that only settle late come from the T = 4 runs; the other ten
from the T = 1 grid. Same second differences as those runs (central, h = 1.5e-4).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
R = "N:/work/Intracellular-Transport-Computational-Modeling/results"
T1 = pd.read_csv(os.path.join(R, "2026-10-05_ab-grid-16x16_second-derivative", "curves_16x16.csv"))
T4 = pd.read_csv(os.path.join(R, "2026-10-05_ab-pairs-16x16_T4", "curves_16x16_T4.csv"))
OUT = os.path.join(R, "2026-10-05_ab-grid-16x16_tau-mask")
os.makedirs(OUT, exist_ok=True)

A = B = [0.1, 1.0, 10.0, 100.0]
LATE = {(10.0, 0.1), (10.0, 1.0), (1.0, 0.1), (1.0, 1.0), (0.1, 0.1), (0.1, 1.0)}

curves, rows = {}, []
for a in A:
    for b in B:
        src, T = (T4, 4.0) if (a, b) in LATE else (T1, 1.0)
        key = f"a{a:g}_b{b:g}"
        t = src["t"].to_numpy()
        y = src[f"lnM_{key}"].to_numpy()
        q = src[f"d2lnM_{key}"].to_numpy()
        ok = np.isfinite(q)                       # the two endpoints have no central difference
        l = (np.abs(q) <= TAU) & ok
        bad = np.nonzero(~l & ok)[0]              # samples with l = 0
        if bad.size == 0:
            ts = t[np.nonzero(ok)[0][0]]          # never exceeds tau
        elif bad[-1] + 1 >= len(t) or not ok[bad[-1] + 1]:
            ts = None                             # exceeds tau at the last judged sample
        else:
            ts = float(t[bad[-1] + 1])
        curves[(a, b)] = (t, y, q, ts, T)
        rows.append(dict(a=a, b=b, T=T, tau=TAU, t_star=ts,
                         M_at_t_star=None if ts is None else float(np.exp(np.interp(ts, t, y)))))

pd.DataFrame(rows).to_csv(os.path.join(OUT, f"t_star_tau{TAU:g}_16x16.csv"), index=False)


def title(k):
    _, _, _, ts, T = curves[k]
    return f"a={k[0]:g}, b={k[1]:g}  T={T:g}  t*=" + ("none" if ts is None else f"{ts:.4f}")


def lnm_panel(ax, k):
    t, y, _, ts, T = curves[k]
    ax.plot(t, y, lw=1.2, color="0.25")
    if ts is not None:
        ax.axvline(ts, color="red", lw=1.2)
    ax.set_xlim(0, T)


def d2_panel(ax, k, zoom):
    t, _, q, ts, T = curves[k]
    ax.plot(t, q, lw=1.0, color="C2")
    ax.axhline(0, color="k", lw=0.6)
    if ts is not None:
        ax.axvline(ts, color="red", lw=1.2)
    ax.set_xlim(0, T)
    if zoom:
        ax.set_ylim(-0.1, 0.1)
        for s_ in (-TAU, TAU):
            ax.axhline(s_, color="0.4", lw=0.8, ls=":")


NOTE = (rf"red: t* = first sample after the last $|y''| > \tau$ = {TAU:g} (right sweep of "
        r"$l(t) = [\,|y''| \leq \tau\,]$);  b $\leq$ 1, a $\leq$ 10 from T = 4, others T = 1")
figs = [(f"lnM_tau{TAU:g}_16x16.png", r"$\ln M(t)$", lambda ax, k: lnm_panel(ax, k)),
        (f"second_derivative_tau{TAU:g}_16x16.png", r"$d^2\ln M/dt^2$",
         lambda ax, k: d2_panel(ax, k, False)),
        (f"second_derivative_tau{TAU:g}_16x16_ylim0.1.png", r"$d^2\ln M/dt^2$",
         lambda ax, k: d2_panel(ax, k, True))]
for name, ylab, panel in figs:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            panel(ax, (a, b))
            ax.grid(True, alpha=0.3)
            ax.set_title(title((a, b)), fontsize=9)
            if i == 3:
                ax.set_xlabel("t")
            if j == 0:
                ax.set_ylabel(ylab)
    extra = (r"  (y-window [-0.1, 0.1]; dotted = $\pm\tau$)" if "ylim0.1" in name else "")
    fig.suptitle(f"{ylab}, 16x16, v=1, N=4{extra}\n{NOTE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)

for r in rows:
    ts = "none" if r["t_star"] is None else f"{r['t_star']:.4f}"
    print(f"a={r['a']:g} b={r['b']:g} T={r['T']:g}: t*={ts}")
