"""Chebyshev fit of ln M(t) on [0, T] for the 16x16 (a, b) grid, and its second derivative.

Reads curves_16x16.csv (ln M and the second-order central difference at h ~ 1.5e-4).
For each pair the degree is the smallest in DEGREES whose max residual reaches
TOL, or, if none does, the smallest within 1.5x of the best residual found --
higher degrees past that plateau only fit rounding noise and amplify it in y''.
Fit: numpy.polynomial.Chebyshev.fit (least squares in the Chebyshev basis);
y'' from .deriv(2), evaluated by Clenshaw recurrence.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

warnings.simplefilter("ignore", np.RankWarning)
OUT = "N:/work/Intracellular-Transport-Computational-Modeling/results/2026-10-05_ab-grid-16x16_second-derivative"
CH = os.path.join(OUT, "chebyshev_full_interval")
os.makedirs(os.path.join(CH, "per_point"), exist_ok=True)
A = B = [0.1, 1.0, 10.0, 100.0]
TOL, NOISE = 1e-14, 1e-4
DEGREES = list(range(80, 401, 20))

d = pd.read_csv(os.path.join(OUT, "curves_16x16.csv"))
t = d["t"].to_numpy()
dom = [t[0], t[-1]]
roots = pd.read_csv(os.path.join(OUT, "first_root_t_star_16x16.csv"))
fd_root = {(r.a, r.b): (None if pd.isna(r.t_star_first_root) else r.t_star_first_root)
           for r in roots.itertuples()}
# t* from the committed criterion (|y''|/y'^2 stays below tau), computed by
# grid_plots.py with launch._char_time_onset on the full-resolution series.
kap_t = {(r.a, r.b): (None if pd.isna(r.t_star_kappa_criterion) else r.t_star_kappa_criterion)
         for r in roots.itertuples()}
TAU = 1e-3

fits, summary = {}, []
for a in A:
    for b in B:
        k = (a, b)
        y = d[f"lnM_a{a:g}_b{b:g}"].to_numpy()
        trials = []
        for deg in DEGREES:
            f = Chebyshev.fit(t, y, deg, domain=dom)
            trials.append((deg, np.max(np.abs(f(t) - y)), f))
        best = min(r for _, r, _ in trials)
        hit = [tr for tr in trials if tr[1] <= TOL]
        deg, res, f = hit[0] if hit else next(tr for tr in trials if tr[1] <= 1.5 * best)
        r = f(t) - y
        late = t >= 0.02
        f1, f2 = f.deriv(1)(t), f.deriv(2)(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.abs(f2) / f1 ** 2
        fits[k] = dict(f=f, y=y, r=r, d2=f2, kappa=kappa,
                       fd=d[f"d2lnM_a{a:g}_b{b:g}"].to_numpy())
        summary.append(dict(a=a, b=b, degree=deg, max_residual=res,
                            max_residual_t_ge_0p02=float(np.max(np.abs(r[late]))),
                            reached_tol=bool(hit)))
        print(f"a={a:g} b={b:g}: degree {deg}, max residual {res:.2e} "
              f"(t>=0.02: {np.max(np.abs(r[late])):.2e})", flush=True)

pd.DataFrame(summary).to_csv(os.path.join(CH, "chebyshev_fit_summary_16x16.csv"), index=False)
deg_of = {(s["a"], s["b"]): s["degree"] for s in summary}


def title(k):
    r, kt = fd_root[k], kap_t[k]
    return (f"a={k[0]:g}, b={k[1]:g} deg={deg_of[k]} root="
            + ("none" if r is None else f"{r:.4f}")
            + r" $\kappa$-t*=" + ("none" if kt is None else f"{kt:.4f}"))


def mark(ax, k):
    if fd_root[k] is not None:
        ax.axvline(fd_root[k], color="red", lw=1.2)
    if kap_t[k] is not None:
        ax.axvline(kap_t[k], color="blue", lw=1.2)


def lnm_panel(ax, k):
    F = fits[k]
    ax.plot(t, F["y"], lw=4.0, color="C0", alpha=0.3, label="data")
    ax.plot(t, F["f"](t), lw=1.1, color="green", zorder=3, label="Chebyshev")
    mark(ax, k)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)


def d2_panel(ax, k):
    F = fits[k]
    ax.plot(t, F["fd"], lw=4.0, color="C0", alpha=0.3, label="central difference")
    ax.plot(t, F["d2"], lw=1.1, color="green", zorder=3, label="Chebyshev")
    ax.axhline(0, color="k", lw=0.6)
    mark(ax, k)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)


def d2_zoom_panel(ax, k):
    """d2_panel with y in [-1, max]: max of both curves (0 if never positive), 5% margin both ends."""
    d2_panel(ax, k)
    F = fits[k]
    top = max(0.0, np.nanmax(F["fd"]), np.nanmax(F["d2"]))
    pad = 0.05 * (top + 1.0)
    ax.set_ylim(-1.0 - pad, top + pad)


def res_panel(ax, k):
    ax.semilogy(t, np.abs(fits[k]["r"]) + 1e-18, lw=0.6, color="C1")
    ax.axhline(TOL, color="k", lw=0.8, ls=":")
    ax.set_ylim(1e-17, 1e-11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)


def coef_panel(ax, k):
    c = np.abs(fits[k]["f"].coef)
    ax.semilogy(np.arange(c.size), c + 1e-20, ".", ms=2, color="C4")
    ax.axhline(TOL, color="k", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)


grids = [
    ("lnM_with_chebyshev_16x16.png", lnm_panel, r"$\ln M(t)$", "t",
     "faint blue = data, green = Chebyshev fit, red = first root of central difference, blue = t* from |y''|/y'^2 < 1e-3"),
    ("second_derivative_with_chebyshev_16x16.png", d2_panel,
     r"$d^2 \ln M/dt^2$", "t",
     r"faint blue = central difference (h=1.5e-4), green = Chebyshev $f''$, "
     r"red = first root, blue = t* where $|y''|/y'^2$ stays below $\tau$ = 1e-3"),
    ("second_derivative_with_chebyshev_16x16_zoom.png", d2_zoom_panel,
     r"$d^2 \ln M/dt^2$", "t",
     r"y-window [-1, max] + 5%;  faint blue = central difference (h=1.5e-4), green = Chebyshev $f''$, "
     r"red = first root, blue = t* where $|y''|/y'^2$ stays below $\tau$ = 1e-3"),
    ("chebyshev_residual_16x16.png", res_panel, r"|fit $-$ data|", "t",
     "dotted = 1e-14"),
    ("chebyshev_coefficients_16x16.png", coef_panel, r"$|c_k|$", "k",
     "Chebyshev coefficient magnitudes; dotted = 1e-14"),
]
for name, panel, ylab, xlab, note in grids:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            panel(ax, (a, b))
            ax.set_title(title((a, b)), fontsize=9)
            if i == 3:
                ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel(ylab)
    fig.suptitle(f"{ylab}, 16x16, v=1, N=4, T=1, Chebyshev on [0, 1]\n{note}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, name), dpi=110)
    plt.close(fig)

for k in fits:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, panel, ylab, xlab in [(axes[0, 0], lnm_panel, r"$\ln M$", "t"),
                                  (axes[0, 1], d2_panel, r"$d^2\ln M/dt^2$", "t"),
                                  (axes[1, 0], res_panel, "|fit - data|", "t"),
                                  (axes[1, 1], coef_panel, r"$|c_k|$", "k")]:
        panel(ax, k)
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("16x16, v=1, N=4   " + title(k))
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "per_point", f"a{k[0]:g}_b{k[1]:g}.png"), dpi=100)
    plt.close(fig)

# How far the Chebyshev y'' is from the central difference, away from the ends.
for k, F in fits.items():
    m = (t > 0.02) & (t < 0.98) & np.isfinite(F["fd"])
    diff = np.abs(F["d2"][m] - F["fd"][m])
    print(f"{k}: max |cheb'' - fd''| on (0.02, 0.98) = {diff.max():.2e}, "
          f"median {np.median(diff):.2e}")
