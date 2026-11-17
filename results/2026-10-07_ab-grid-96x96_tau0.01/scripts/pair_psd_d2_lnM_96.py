"""One pair's |F[y'']|^2 (omega <= 100, linear axes) above its y'' and ln M.

    python pair_psd_d2_lnM_96.py [a b] [tau]      default a = 10, b = 0.1, tau = 0.01

Three panels, top to bottom: the pair's panel of
psd_central_difference_96x96_w100_linear.png (read from psd/psd_96x96.npz, and
checked against the spectrum recomputed here), y'' = d^2 ln M/dt^2 by the central
difference psd_96.py transforms (spacing H), and ln M, with t* from
launch._char_time_onset in red on the last two.

The series comes from t4_source.load_series, with job 54cdf1f3's checkpoints
looked up under this checkout's data_output/ -- R's folder on the server; on
another machine, download the pair's timeseries_4.dat and final_state.json into
data_output/jobs/<job>/checkpoints/ab_grid_char_time_a{a}_b{b}/ first.
"""
import os, sys
SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(SP, "..", "..", ".."))
os.environ["MPLBACKEND"] = "Agg"
sys.path[:0] = [SP, os.path.join(ROOT, "src", "intracellular_transport"), os.path.join(ROOT, "src")]

import numpy as np
import matplotlib.pyplot as plt
from launch_functions import launch as L
import t4_source
from t4_source import load_series

t4_source.T4_DIR = os.path.join(ROOT, "data_output", "jobs", t4_source.T4_JOB, "checkpoints")

args = sys.argv[1:]
a, b = (float(args[0]), float(args[1])) if len(args) >= 2 else (10.0, 0.1)
TAU = float(args[2]) if len(args) >= 3 else 1e-2
PSD = os.path.join(SP, "..", "psd")
G, MCF, W_CUT = 96, 5, 100.0

h = MCF * L.num.compute_dT(G, G)
s = max(1, int(round(L.CHAR_TIME_DIFF_SPACING / h)))
H = s * h

m, T, src = load_series(a, b, SP)
y = np.log(m)[::s]                                  # samples at spacing H, as psd_96.py
t = np.arange(y.size) * H
fd = (y[2:] - 2 * y[1:-1] + y[:-2]) / H ** 2
tc = t[1:-1]
r = L._char_time_onset(m, h, tau=TAU)
ts = None if np.isnan(r["t_star"]) else r["t_star"]

d = np.load(os.path.join(PSD, "psd_96x96.npz"))
w, P = d[f"w_a{a:g}_b{b:g}"], d[f"fd_a{a:g}_b{b:g}"]
P_here = np.abs(H * np.fft.rfft(fd)) ** 2
# |F| to 1e-8 of its peak: ln's last-bit rounding differs between machines and the
# second difference multiplies it by 1/H^2 ~ 1e8, which shows at the noise floor
if P_here.shape != P.shape or not np.allclose(np.sqrt(P_here), np.sqrt(P), rtol=0,
                                              atol=1e-8 * np.sqrt(P.max())):
    sys.exit(f"series does not reproduce psd_96x96.npz for ({a:g}, {b:g}) -- "
             "is it the one psd_96.py used?")
k = 1 + int(np.argmax(P[1:]))
k1 = int(np.searchsorted(w, W_CUT)) + 1             # through the first bin >= W_CUT


def half_max_crossings(w, P, k):
    """omega either side of bin k where P falls through P[k] / 2, linear between bins
    (None on a side where it never does, e.g. a peak at the first bin above omega = 0)."""
    half = P[k] / 2
    i = k
    while i > 0 and P[i] >= half:
        i -= 1
    j = k
    while j < P.size - 1 and P[j] >= half:
        j += 1
    lo = None if P[i] >= half else w[i] + (half - P[i]) / (P[i + 1] - P[i]) * (w[i + 1] - w[i])
    hi = None if P[j] >= half else w[j - 1] + (half - P[j - 1]) / (P[j] - P[j - 1]) * (w[j] - w[j - 1])
    return half, lo, hi


half, wl, wr = half_max_crossings(w, P, k)
fwhm = None if wl is None or wr is None else wr - wl
print(f"({a:g}, {b:g}) {src}: N={tc.size}  t*={'none' if ts is None else f'{ts:.6f}'}  "
      f"omega_peak={w[k]:.4f}  half max={half:.4f} at omega="
      f"{'none' if wl is None else f'{wl:.4f}'}, {'none' if wr is None else f'{wr:.4f}'}  "
      f"FWHM={'none' if fwhm is None else f'{fwhm:.4f}'}")

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12))

ax0.plot(w[:k1], P[:k1], lw=1.0, color="C0", marker=".", ms=4)
ax0.axvline(w[k], color="red", lw=1.0)
ax0.axhline(half, color="C1", lw=1.0, ls="--")
if fwhm is not None:
    ax0.plot([wl, wr], [half, half], color="C1", lw=2.0, marker="|", ms=12)
    ax0.annotate(f"FWHM = {fwhm:.3f}", (wr, half), xytext=(8, 6),
                 textcoords="offset points", ha="left", va="bottom", fontsize=9, color="C1")
fmt_w = lambda v: "none" if v is None else f"{v:.3f}"
ax0.set_xlim(0, W_CUT)
ax0.text(0.97, 0.96, rf"H = {H:.4e},  $\Delta\omega$ = {w[1]:.4f}" + "\n"
         + rf"$|F|^2(0)$ = {P[0]:.3g},  peak = {P[k]:.3g}" + "\n"
         + rf"$2\pi/\omega_{{peak}}$ = {2 * np.pi / w[k]:.4f}" + "\n"
         + f"half max = {half:.3f} at $\\omega$ = {fmt_w(wl)}, {fmt_w(wr)}" + "\n"
         + f"FWHM = {fmt_w(fwhm)}",
         transform=ax0.transAxes, ha="right", va="top", fontsize=8, color="0.2",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
ax0.set_title(rf"$|\mathcal{{F}}[y''](\omega)|^2$, $\omega \leq$ {W_CUT:g}, linear axes;  "
              rf"red = peak over k $\geq$ 1, $\omega_{{peak}}$ = {w[k]:.3f};  "
              "dashed = half maximum (crossings linear between bins)", fontsize=10)
ax0.set_xlabel(r"$\omega$ (rad / unit t)")
ax0.set_ylabel(r"$|\mathcal{F}[y'']|^2$")

ax1.plot(tc, fd, lw=1.0, color="C2")
ax1.axhline(0, color="k", lw=0.6)
ax1.set_title(rf"$y'' = d^2\ln M/dt^2$, central difference, H = {H:.4e}", fontsize=10)
ax1.set_ylabel(r"$d^2\ln M/dt^2$")

ax2.plot(t, y, lw=1.2, color="0.25")
ax2.set_title(r"$\ln M(t)$", fontsize=10)
ax2.set_ylabel(r"$\ln M(t)$")

for ax in (ax1, ax2):
    if ts is not None:
        ax.axvline(ts, color="red", lw=1.2)
    ax.set_xlim(0, T)
    ax.set_xlabel("t")
for ax in (ax0, ax1, ax2):
    ax.grid(True, alpha=0.3)

fig.suptitle(f"a={a:g}, b={b:g}  --  96x96, v=1, N=4, T={T:g} ({src})\n"
             rf"red on $y''$ and $\ln M$: t* = sample after the last $|y''| > \tau$ = {TAU:g} "
             "(right sweep), t* = " + ("none" if ts is None else f"{ts:.4f}"), fontsize=11)
fig.tight_layout()
out = os.path.normpath(os.path.join(PSD, f"psd_d2_lnM_96x96_a{a:g}_b{b:g}.png"))
fig.savefig(out, dpi=110)
plt.close(fig)
print(out)
