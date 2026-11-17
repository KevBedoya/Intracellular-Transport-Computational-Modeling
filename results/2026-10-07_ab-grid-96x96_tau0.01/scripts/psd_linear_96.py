"""psd_central_difference_96x96_full on axes other than log-log, from psd_96.py's output.

    python psd_linear_96.py

Reads psd/psd_96x96.npz and psd/psd_summary_96x96.csv (written by psd_96.py) and
writes, next to them, the central-difference spectra on three axes:

  _linear    linear omega, linear |F|^2, omega = 0 included
  _semilogy  linear omega, log |F|^2,    omega = 0 included
  _semilogx  log omega,    linear |F|^2, k >= 1 (omega = 0 is off a log axis)

each over the full range (psd_central_difference_96x96_full_*.png) and cut at
omega = W_CUT (psd_central_difference_96x96_w100_*.png; only bins up to the first
one at or past W_CUT are drawn, so the y axis scales to what is shown).

The red line is the same peak as the log-log figure (largest |F|^2 over k >= 1,
psd_peaks_96x96.csv), so the figures agree; on the linear-omega axes the omega = 0
bin is drawn too and is the larger one wherever the spectrum falls monotonically.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
PSD = os.path.join(SP, "..", "psd")

G, MCF = 96, 5
A = B = [0.1, 1.0, 10.0, 100.0]
dRad, dThe = 1.0 / G, 2 * np.pi / G
h = MCF * 0.1 * min(dRad ** 2, dThe ** 2 * dRad ** 2) / 2.0     # as psd_96.py
s = max(1, int(round(1e-4 / h)))
H = s * h
W_CUT = 100.0

d = np.load(os.path.join(PSD, "psd_96x96.npz"))
T = {(r.a, r.b): r.T for r in pd.read_csv(os.path.join(PSD, "psd_summary_96x96.csv")).itertuples()}

axes_kinds = [("linear", "linear", "linear", "linear-linear"),
              ("semilogy", "linear", "log", "semilog-y"),
              ("semilogx", "log", "linear", "semilog-x")]

for wmax, (stem, xs, ys, desc) in [(wm, kind) for wm in (None, W_CUT) for kind in axes_kinds]:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ax = axes[i, j]
            w, P = d[f"w_a{a:g}_b{b:g}"], d[f"fd_a{a:g}_b{b:g}"]
            k = 1 + int(np.argmax(P[1:]))
            k0 = 1 if xs == "log" else 0
            if wmax is None:
                ax.plot(w[k0:], P[k0:], lw=0.8, color="C0")
            else:
                k1 = int(np.searchsorted(w, wmax)) + 1      # through the first bin >= wmax
                ax.plot(w[k0:k1], P[k0:k1], lw=1.0, color="C0", marker=".", ms=4)
                ax.set_xlim(0 if xs == "linear" else w[1], wmax)
            ax.set_xscale(xs)
            ax.set_yscale(ys)
            ax.axvline(w[k], color="red", lw=1.0)
            ax.text(0.97, 0.96, rf"H = {H:.4e},  $\Delta\omega$ = {w[1]:.4f}" + "\n"
                    + rf"$|F|^2(0)$ = {P[0]:.3g},  peak = {P[k]:.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color="0.2", bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                           ec="0.7", alpha=0.85))
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"a={a:g}, b={b:g}  T={T[(a, b)]:g}  "
                         rf"$\omega_{{peak}}$={w[k]:.3f}", fontsize=9)
            if i == 3:
                ax.set_xlabel(r"$\omega$ (rad / unit t)")
            if j == 0:
                ax.set_ylabel(r"$|\mathcal{F}[y'']|^2$")
    rng = "full range" if wmax is None else rf"$\omega \leq$ {wmax:g}"
    head = (rf"$|\mathcal{{F}}[d^2\ln M/dt^2](\omega)|^2$ -- central difference -- "
            f"96x96, v=1, N=4  ({rng}, {desc}); red = peak over k $\\geq$ 1"
            "\n" + rf"H = {H:.6e} = time between the $y''$ samples transformed "
            rf"({s} solver samples $\times$ {h:.6e})" + "\n"
            rf"$\Delta\omega = 2\pi/(NH) = 2\pi/T$ = spacing between frequency points;  "
            rf"Nyquist $\pi/H$ = {np.pi / H:.4g}")
    fig.suptitle(head, fontsize=12)
    fig.tight_layout()
    tag = "full" if wmax is None else f"w{wmax:g}"
    out = os.path.join(PSD, f"psd_central_difference_96x96_{tag}_{stem}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(os.path.normpath(out))
