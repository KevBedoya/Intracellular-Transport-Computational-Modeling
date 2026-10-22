"""|F[y'']|^2 on the 96x96 (a, b) grid, y = ln M, for two second derivatives.

    python psd_96.py [t1_dir]        t1_dir: folder with the T = 1 tm_a{a}_b{b}.dat files

Series come from t4_source.load_series (T = 4 from job 54cdf1f3 once a pair has
finished, T = 1 from job 0b8d4022 otherwise), sampled every h = 5 dT.

  central difference: y'' = (y[i+s] - 2 y[i] + y[i-s]) / H^2, H = s h ~ 1e-4,
                      kept at spacing H -- as tau_mask_96.py.
  Chebyshev:          ln M fitted on [0, T] at the same samples
                      (numpy Chebyshev.fit; degree = smallest in DEGREES whose
                      max residual reaches 1e-14, else the smallest within 1.5x
                      of the best -- the 16x16 rule), y'' from .deriv(2).

Both are evaluated on the same uniform grid t_n = n H (the interior samples
where the central difference exists), and

    P(w) = | H sum_n y''(t_n) exp(-i w t_n) |^2,   w = 2 pi k / (N H),

the squared magnitude of the Riemann-sum Fourier transform (np.fft.rfft), with
no window and no detrending. w is angular frequency (rad per unit t).
"""
import os, sys, warnings
R = "N:/work/Intracellular-Transport-Computational-Modeling"
SP = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [SP]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev
from t4_source import load_series

warnings.simplefilter("ignore", np.RankWarning)
T1_DIR = sys.argv[1] if len(sys.argv) > 1 else SP
OUT = os.path.join(R, "results", "2026-10-07_ab-grid-96x96_tau0.01", "psd")
os.makedirs(OUT, exist_ok=True)

G, MCF = 96, 5
A = B = [0.1, 1.0, 10.0, 100.0]
TOL = 1e-14
DEGREES = [100, 150, 200, 300, 400, 500, 600, 800]
ZOOM_BINS = 8

dRad, dThe = 1.0 / G, 2 * np.pi / G
h = MCF * 0.1 * min(dRad ** 2, dThe ** 2 * dRad ** 2) / 2.0     # compute_dT, D = 1
s = max(1, int(round(1e-4 / h)))                                   # CHAR_TIME_DIFF_SPACING
H = s * h
print(f"h={h:.6g}  s={s}  H={H:.6g}")

res, summary = {}, []
for a in A:
    for b in B:
        m, Tp, src = load_series(a, b, T1_DIR)
        y_full = np.log(m)
        y = y_full[::s]                                # samples at spacing H
        t = np.arange(y.size) * H
        fd = (y[2:] - 2 * y[1:-1] + y[:-2]) / H ** 2   # = y'' at stride s on the full series
        tc = t[1:-1]

        dom = [t[0], t[-1]]
        trials = []
        for deg in DEGREES:
            f = Chebyshev.fit(t, y, deg, domain=dom)
            trials.append((deg, float(np.max(np.abs(f(t) - y))), f))
        best = min(r for _, r, _ in trials)
        hit = [tr for tr in trials if tr[1] <= TOL]
        deg, rmax, f = hit[0] if hit else next(tr for tr in trials if tr[1] <= 1.5 * best)
        ch = f.deriv(2)(tc)

        w = 2 * np.pi * np.fft.rfftfreq(tc.size, H)
        P_fd = np.abs(H * np.fft.rfft(fd)) ** 2
        P_ch = np.abs(H * np.fft.rfft(ch)) ** 2
        res[(a, b)] = dict(T=Tp, w=w, fd=P_fd, ch=P_ch)
        inner = (tc > 0.02 * Tp) & (tc < 0.98 * Tp)
        summary.append(dict(a=a, b=b, T=Tp, source=src, n_samples=tc.size, dw=w[1],
                            cheb_degree=deg, cheb_max_residual=rmax, cheb_reached_tol=bool(hit),
                            max_abs_d2_diff_inner=float(np.max(np.abs(ch - fd)[inner])),
                            median_abs_d2_diff_inner=float(np.median(np.abs(ch - fd)[inner]))))
        print(f"a={a:g} b={b:g} T={Tp:g}: N={tc.size} deg={deg} resid={rmax:.1e} "
              f"|cheb''-fd''| inner max {summary[-1]['max_abs_d2_diff_inner']:.1e}", flush=True)

pd.DataFrame(summary).to_csv(os.path.join(OUT, "psd_summary_96x96.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "psd_96x96.npz"),
                    **{f"{kind}_a{a:g}_b{b:g}": res[(a, b)][kind]
                       for a in A for b in B for kind in ("w", "fd", "ch")})

methods = [("fd", "central difference", "central_difference"),
           ("ch", "Chebyshev", "chebyshev")]


def peak(r, key):
    """Largest |F|^2 over the plotted bins (k >= 1; omega = 0 is off a log axis)."""
    k = 1 + int(np.argmax(r[key][1:]))
    return k, r["w"][k], r[key][k]


peaks = []
for key, label, stem in methods:
    for zoom in (False, True):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                ax = axes[i, j]
                r = res[(a, b)]
                kp, wp, pp = peak(r, key)
                if zoom:
                    # neighbourhood of the peak: ZOOM_BINS bins either side
                    lo, hi = max(1, kp - ZOOM_BINS), kp + ZOOM_BINS + 1
                    ax.semilogy(r["w"][lo:hi], r[key][lo:hi], lw=1.0, color="C0",
                                marker=".", ms=4)
                    ax.set_xlim(r["w"][lo], r["w"][hi - 1])
                else:
                    ax.loglog(r["w"][1:], r[key][1:], lw=0.8, color="C0")
                    peaks.append(dict(method=label, a=a, b=b, T=r["T"], k=kp,
                                      omega_peak=wp, psd_peak=pp))
                ax.axvline(wp, color="red", lw=1.0)
                ax.grid(True, which="both", alpha=0.3)
                ax.set_title(f"a={a:g}, b={b:g}  T={r['T']:g}  "
                             rf"$\omega_{{peak}}$={wp:.3f}", fontsize=9)
                if i == 3:
                    ax.set_xlabel(r"$\omega$ (rad / unit t)")
                if j == 0:
                    ax.set_ylabel(r"$|\mathcal{F}[y'']|^2$")
        rng = (rf"zoom: $\pm${ZOOM_BINS} bins ($\Delta\omega = 2\pi/T$) about the peak" if zoom
               else "full range, log-log")
        fig.suptitle(rf"$|\mathcal{{F}}[d^2\ln M/dt^2](\omega)|^2$ -- {label} -- "
                     f"96x96, v=1, N=4  ({rng}); red = peak", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"psd_{stem}_96x96_{'peak_zoom' if zoom else 'full'}.png"),
                    dpi=110)
        plt.close(fig)
pd.DataFrame(peaks).to_csv(os.path.join(OUT, "psd_peaks_96x96.csv"), index=False)
print(OUT)
