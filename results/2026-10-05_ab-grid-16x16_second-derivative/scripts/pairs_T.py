"""ln M(t) to T (argv, default 2) on 16x16 for six (a, b) pairs, with its second difference.

Same setup as the T=1 grid (v=1, N_LIST=[0,4,8,12], MA_collection_factor=5),
same second difference (h = 1.5e-4, the rule-of-thumb step), same red line
(first genuine root, |q| > 1e-4 both sides) and blue line (committed
kappa criterion, launch._char_time_onset at tau=1e-3).
"""
import os, sys
R = "N:/work/Intracellular-Transport-Computational-Modeling"
os.environ["MPLBACKEND"] = "Agg"
sys.path[:0] = [R + "/src/intracellular_transport", R + "/src"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from launch_functions import launch as L

G, V, MCF = 16, 1.0, 5
T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
TT = f"T{T:g}"
PAIRS = [(10.0, 0.1), (10.0, 1.0), (1.0, 0.1), (1.0, 1.0), (0.1, 0.1), (0.1, 1.0)]
N_LIST = np.array([0, 4, 8, 12], dtype=np.int64)
NOISE, TAU = 1e-4, L.CHAR_TIME_TAU
OUT = os.path.join(R, "results", f"2026-10-05_ab-pairs-16x16_{TT}")
os.makedirs(OUT, exist_ok=True)

K = L.num.compute_K(G, G, T)
rk = int(np.floor(K / MCF))
h = MCF * L.num.compute_dT(G, G)
t = np.arange(rk) * h

lnM, d2, root, kap = {}, {}, {}, {}


def first_root(q):
    sig = np.where(np.abs(q) > NOISE, np.sign(q), 0.0)
    sig[~np.isfinite(q)] = 0.0
    idx = np.nonzero(sig)[0]
    flips = np.nonzero(sig[idx][1:] != sig[idx][:-1])[0]
    if not flips.size:
        return None
    lo, hi = idx[flips[0]], idx[flips[0] + 1]
    sg = np.sign(q[lo:hi + 1])
    ch = np.nonzero(sg[:-1] * sg[1:] < 0)[0]
    j = lo + (ch[-1] if ch.size else 0)
    q0, q1 = q[j], q[j + 1]
    return float(t[j] + (t[j + 1] - t[j]) * q0 / (q0 - q1))


for k in PAIRS:
    D_, A_ = L.sup.initialize_layers(G, G)
    _, series = L._setup_checkpoint("x", rk)
    L._solve_mass_analysis("cpu", series, G, G, k[0], k[1], V, T, N_LIST, D_, A_,
                           MCF, rk, 0.0, 1.0, 1.0, 10 ** 6, True, 0, 0, cp=None)
    m = np.array(series[4], dtype=np.float64)
    y = np.log(m)
    q = np.full_like(y, np.nan)
    q[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / h ** 2
    lnM[k], d2[k], root[k] = y, q, first_root(q)
    ts = L._char_time_onset(m, h, tau=TAU)["t_star"]
    kap[k] = None if np.isnan(ts) else float(ts)

data = {"t": t}
for k in PAIRS:
    data[f"lnM_a{k[0]:g}_b{k[1]:g}"] = lnM[k]
    data[f"d2lnM_a{k[0]:g}_b{k[1]:g}"] = d2[k]
pd.DataFrame(data).to_csv(os.path.join(OUT, f"curves_16x16_{TT}.csv"), index=False)
pd.DataFrame([dict(a=k[0], b=k[1], t_star_first_root=root[k],
                   t_star_kappa_criterion=kap[k], M_at_T=float(np.exp(lnM[k][-1])))
              for k in PAIRS]).to_csv(os.path.join(OUT, f"t_star_16x16_{TT}.csv"), index=False)


def fmt(v):
    return "none" if v is None else f"{v:.4f}"


def lines(ax, k):
    if root[k] is not None:
        ax.axvline(root[k], color="red", lw=1.2)
    if kap[k] is not None:
        ax.axvline(kap[k], color="blue", lw=1.2)


def grid(name, ylab, panel, note):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, k in zip(axes.flat, PAIRS):
        panel(ax, k)
        lines(ax, k)
        ax.set_xlim(0, T)
        ax.grid(True, alpha=0.3)
        ax.set_title(rf"a={k[0]:g}, b={k[1]:g}  root={fmt(root[k])}  $\kappa$-t*={fmt(kap[k])}",
                     fontsize=10)
        ax.set_ylabel(ylab)
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle(f"{ylab}, 16x16, v=1, N=4, T={T:g}\n{note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)


LINES = (r"red = first root of $d^2\ln M/dt^2$, blue = t* where $|y''|/y'^2$ "
         r"stays below $\tau$ = 1e-3")


def p_lnm(ax, k):
    ax.plot(t, lnM[k], lw=1.2, color="0.25")


def p_d2(ax, k):
    ax.plot(t, d2[k], lw=1.0, color="C2")
    ax.axhline(0, color="k", lw=0.6)


def p_d2_zoom(ax, k):
    p_d2(ax, k)
    top = max(0.0, np.nanmax(d2[k]))
    pad = 0.05 * (top + 1.0)
    ax.set_ylim(-1.0 - pad, top + pad)


grid(f"lnM_16x16_{TT}.png", r"$\ln M(t)$", p_lnm, LINES)
grid(f"second_derivative_16x16_{TT}.png", r"$d^2\ln M/dt^2$", p_d2,
     f"central difference, h={h:.3g};  " + LINES)
grid(f"second_derivative_16x16_{TT}_zoom.png", r"$d^2\ln M/dt^2$", p_d2_zoom,
     f"y-window [-1, max] + 5%;  central difference, h={h:.3g};  " + LINES)

for k in PAIRS:
    print(f"a={k[0]:g} b={k[1]:g}: M(T)={np.exp(lnM[k][-1]):.4g}  root={fmt(root[k])}  "
          f"kappa-t*={fmt(kap[k])}")
