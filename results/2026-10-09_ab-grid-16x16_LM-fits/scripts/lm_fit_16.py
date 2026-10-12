"""Levenberg-Marquardt fits of M(t) on the 16x16 (a, b) grid.

Data: the six b <= 1, a <= 10 pairs from the T = 4 runs, the other ten from T = 1
(same series as the tau-mask study). Fit parameters are written alpha, beta, ...
to keep them apart from the switch rates a, b.

  model 1 (proposed):  M = exp(-alpha t) cos(beta t)                 on [0, T]
  model 2:             M = A exp(-lam t) + B exp(-gam t) cos(om t + phi)   on [T0, T]

Both minimise the relative residual (model - M) / M with
scipy.optimize.least_squares(method="lm"), M spanning several decades.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

R = "N:/work/Intracellular-Transport-Computational-Modeling/results"
T1 = pd.read_csv(os.path.join(R, "2026-10-05_ab-grid-16x16_second-derivative", "curves_16x16.csv"))
T4 = pd.read_csv(os.path.join(R, "2026-10-05_ab-pairs-16x16_T4", "curves_16x16_T4.csv"))
OUT = os.path.join(R, "2026-10-09_ab-grid-16x16_LM-fits")
os.makedirs(OUT, exist_ok=True)

A_ = B_ = [0.1, 1.0, 10.0, 100.0]
LATE = {(10.0, 0.1), (10.0, 1.0), (1.0, 0.1), (1.0, 1.0), (0.1, 0.1), (0.1, 1.0)}
T0 = 0.15
STRIDE = 5                      # fit every 5th sample (h = 7.5e-4); plenty for a smooth curve


def m1(p, t):
    al, be = p
    return np.exp(-al * t) * np.cos(be * t)


def m2(p, t):
    A, lam, B, gam, om, phi = p
    return A * np.exp(-lam * t) + B * np.exp(-gam * t) * np.cos(om * t + phi)


def fit(model, p0, t, M):
    res = least_squares(lambda p: (model(p, t) - M) / M, p0, method="lm",
                        max_nfev=20000, xtol=1e-14, ftol=1e-14)
    r = (model(res.x, t) - M) / M
    return res, float(np.max(np.abs(r))), float(np.sqrt(np.mean(r ** 2)))


rows, data = [], {}
for a in A_:
    for b in B_:
        src, T = (T4, 4.0) if (a, b) in LATE else (T1, 1.0)
        key = f"a{a:g}_b{b:g}"
        t_all = src["t"].to_numpy()
        M_all = np.exp(src[f"lnM_{key}"].to_numpy())
        t, M = t_all[::STRIDE], M_all[::STRIDE]

        # Starting values from the tail: slope of ln M over the last 30%.
        tail = t > 0.7 * T
        slope, icpt = np.polyfit(t[tail], np.log(M[tail]), 1)
        lam0 = -slope

        r1, max1, rms1 = fit(m1, [lam0, 0.5], t, M)

        sel = t >= T0
        best = None
        for om0 in (1.0, 3.0, 6.0, 12.0):           # a few starts; LM is local
            p0 = [np.exp(icpt), lam0, 0.1, lam0 + 2.0, om0, 0.0]
            try:
                cand = fit(m2, p0, t[sel], M[sel])
            except Exception:
                continue
            if best is None or cand[2] < best[2]:
                best = cand
        r2, max2, rms2 = best

        data[(a, b)] = (t_all, M_all, T, r1.x, r2.x)
        A, lam, B, gam, om, phi = r2.x
        rows.append(dict(a=a, b=b, T=T,
                         m1_alpha=r1.x[0], m1_beta=r1.x[1], m1_rel_rms=rms1, m1_rel_max=max1,
                         m2_A=A, m2_lambda=lam, m2_B=B, m2_gamma=gam, m2_omega=om, m2_phi=phi,
                         m2_rel_rms=rms2, m2_rel_max=max2))
        print(f"a={a:g} b={b:g} T={T:g} | model1 alpha={r1.x[0]:.4f} beta={r1.x[1]:.4f} "
              f"rel rms={rms1:.2e} max={max1:.2e} | model2 lam={lam:.4f} gam={gam:.3f} "
              f"om={om:.3f} B/A={B / A:.2e} rel rms={rms2:.2e} max={max2:.2e}", flush=True)

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "lm_fits_16x16.csv"), index=False)


def grid(name, panel, ylab, note):
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, a in enumerate(A_):
        for j, b in enumerate(B_):
            ax = axes[i, j]
            panel(ax, (a, b))
            ax.grid(True, alpha=0.3)
            ax.set_title(f"a={a:g}, b={b:g}, T={data[(a, b)][2]:g}", fontsize=9)
            if i == 3:
                ax.set_xlabel("t")
            if j == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{ylab}, 16x16, v=1, N=4 -- Levenberg-Marquardt fits\n{note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)


NOTE = (r"grey: data;  blue dashed: model 1, $e^{-\alpha t}\cos\beta t$ on [0, T];  "
        r"red: model 2, $A e^{-\lambda t} + B e^{-\gamma t}\cos(\omega t + \phi)$ on "
        f"[{T0}, T]")


def p_M(ax, k):
    t, M, T, p1, p2 = data[k]
    ax.plot(t, M, lw=3, color="0.6", label="M(t)")
    ax.plot(t, m1(p1, t), lw=1.0, ls="--", color="C0", label="model 1")
    s = t >= T0
    ax.plot(t[s], m2(p2, t[s]), lw=1.0, color="C3", label="model 2")
    ax.set_xlim(0, T)


def p_lnM(ax, k):
    t, M, T, p1, p2 = data[k]
    ax.plot(t, np.log(M), lw=3, color="0.6", label=r"$\ln M$")
    with np.errstate(invalid="ignore"):
        ax.plot(t, np.log(m1(p1, t)), lw=1.0, ls="--", color="C0", label="model 1")
    s = t >= T0
    with np.errstate(invalid="ignore"):
        ax.plot(t[s], np.log(m2(p2, t[s])), lw=1.0, color="C3", label="model 2")
    ax.set_xlim(0, T)
    ax.set_ylim(np.log(M).min() - 0.5, 0.3)


def p_res(ax, k):
    t, M, T, p1, p2 = data[k]
    ax.plot(t, (m1(p1, t) - M) / M, lw=1.0, ls="--", color="C0", label="model 1")
    s = t >= T0
    ax.plot(t[s], (m2(p2, t[s]) - M[s]) / M[s], lw=1.0, color="C3", label="model 2")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.1, 0.1)


grid("M_with_fits_16x16.png", p_M, r"$M(t)$", NOTE)
grid("lnM_with_fits_16x16.png", p_lnM, r"$\ln M(t)$", NOTE + r";  model 1 undefined where $\cos\beta t \leq 0$")
grid("relative_residual_16x16.png", p_res, r"(model $-$ M) / M", NOTE + ";  y in [-0.1, 0.1]")
print(OUT)
