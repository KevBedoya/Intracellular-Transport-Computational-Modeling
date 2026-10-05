"""ln M(t) and its second difference over the (a, b) grid, first root marked in red.

    python grid_plots.py <grid>          e.g. 32

The second difference is (y[i+s] - 2 y[i] + y[i-s]) / H^2 with y = ln M, taken
across s samples so that H = s * h_sample is as close as possible to the
double-precision rule of thumb H ~ 1e-4. The root is the first genuine sign
change (|q| > NOISE = 1e-4 on both sides; below that, early
rounding in M -- a few ulps summed over every cell -- reaches ~1e-6 at 32x32), linearly interpolated.
"""
import os, sys
R = "N:/work/Intracellular-Transport-Computational-Modeling"
os.environ["MPLBACKEND"] = "Agg"
sys.path[:0] = [R + "/src/intracellular_transport", R + "/src"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from launch_functions import launch as L

G = int(sys.argv[1])
OUT = os.path.join(R, "results", f"2026-10-05_ab-grid-{G}x{G}_second-derivative")
A = B = [0.1, 1.0, 10.0, 100.0]
V, T, MCF = 1.0, 1.0, 5
N_LIST = np.array([i * G // 4 for i in range(4)], dtype=np.int64)   # 4 evenly spaced tubes
H_TARGET, NOISE = 1e-4, 1e-4

K = L.num.compute_K(G, G, T)
rk = int(np.floor(K / MCF))
h = MCF * L.num.compute_dT(G, G)
s = max(1, int(round(H_TARGET / h)))
H = s * h
t = np.arange(rk) * h
print(f"grid {G}x{G}  N_LIST={N_LIST.tolist()}  sample h={h:.4g}  stride s={s}  H={H:.4g}  samples={rk}")

lnM, d2, kap_t = {}, {}, {}
TAU = launch_tau = L.CHAR_TIME_TAU
for a in A:
    for b in B:
        D_, A_ = L.sup.initialize_layers(G, G)
        _, series = L._setup_checkpoint("x", rk)
        L._solve_mass_analysis("cpu", series, G, G, a, b, V, T, N_LIST, D_, A_,
                               MCF, rk, 0.0, 1.0, 1.0, 10 ** 6, True, 0, 0, cp=None)
        y = np.log(np.array(series[4], dtype=np.float64))
        q = np.full_like(y, np.nan)
        q[s:-s] = (y[2 * s:] - 2 * y[s:-s] + y[:-2 * s]) / H ** 2
        lnM[(a, b)], d2[(a, b)] = y, q
        # t* by the committed criterion (|y''|/y'^2 stays below tau), on the
        # full-resolution mass series -- exactly what an API job would report.
        ts = L._char_time_onset(np.array(series[4], dtype=np.float64), h, tau=TAU)["t_star"]
        kap_t[(a, b)] = None if np.isnan(ts) else float(ts)
        print(f"  solved a={a:g} b={b:g}", flush=True)


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
    return float(t[j] + (t[j + 1] - t[j]) * q0 / (q0 - q1)) if q0 != q1 else float(t[j])


roots = {k: first_root(q) for k, q in d2.items()}

os.makedirs(os.path.join(OUT, "per_point"), exist_ok=True)
keep = slice(0, None, s)          # store the curves at the differencing spacing
data = {"t": t[keep]}
for k in lnM:
    data[f"lnM_a{k[0]:g}_b{k[1]:g}"] = lnM[k][keep]
    data[f"d2lnM_a{k[0]:g}_b{k[1]:g}"] = d2[k][keep]
pd.DataFrame(data).to_csv(os.path.join(OUT, f"curves_{G}x{G}.csv"), index=False)
pd.DataFrame([{"a": k[0], "b": k[1], "t_star_first_root": r,
               "t_star_kappa_criterion": kap_t[k]} for k, r in roots.items()]
             ).to_csv(os.path.join(OUT, f"first_root_t_star_{G}x{G}.csv"), index=False)


def label(k):
    r = roots[k]
    kt = kap_t[k]
    return (f"a={k[0]:g}, b={k[1]:g}  root=" + ("none" if r is None else f"{r:.4f}")
            + r"  $\kappa$-t*=" + ("none" if kt is None else f"{kt:.4f}"))


def mark(ax, k):
    if roots[k] is not None:
        ax.axvline(roots[k], color="red", lw=1.2)


def lnm_panel(ax, k):
    ax.plot(t[keep], lnM[k][keep], lw=1.2, color="0.25")
    mark(ax, k)
    if kap_t[k] is not None:
        ax.axvline(kap_t[k], color="blue", lw=1.2)
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3)


def d2_panel(ax, k):
    ax.plot(t[keep], d2[k][keep], lw=1.0, color="C2")
    ax.axhline(0, color="k", lw=0.6)
    mark(ax, k)
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3)


sub = f"{G}x{G}, v={V:g}, N=4, T={T:g}, H={H:.3g}"
for name, panel, ylab in [(f"lnM_grid_{G}x{G}.png", lnm_panel, r"$\ln M(t)$"),
                          (f"second_derivative_grid_{G}x{G}.png", d2_panel,
                           r"$d^2 \ln M / dt^2$")]:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            panel(ax, (a, b))
            ax.set_title(label((a, b)), fontsize=9)
            if i == 3:
                ax.set_xlabel("t")
            if j == 0:
                ax.set_ylabel(ylab)
    fig.suptitle(f"{ylab}, {sub}   (rows: a, columns: b; red = first root of "
                 r"$d^2 \ln M/dt^2$" + ("; blue = t* where $|y''|/y'^2$ stays below "
                 f"$\tau$={TAU:g}" if "lnM" in name else "") + ")", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)

for k in lnM:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    lnm_panel(ax1, k)
    ax1.set_ylabel(r"$\ln M(t)$")
    ax1.set_title(f"{sub}   " + label(k), fontsize=9)
    d2_panel(ax2, k)
    ax2.set_ylabel(r"$d^2 \ln M / dt^2$")
    ax2.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "per_point", f"a{k[0]:g}_b{k[1]:g}.png"), dpi=110)
    plt.close(fig)

for k, r in roots.items():
    print(k, r)
