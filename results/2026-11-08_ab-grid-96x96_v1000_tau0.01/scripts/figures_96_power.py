"""The 11-05 figures for the v = 1000 grid: every (a, b) pair in a 4x4 panel grid.

    June-2026/.venv/bin/python results/2026-11-08_ab-grid-96x96_v1000_tau0.01/scripts/figures_96_power.py

Reads summary_96x96_v1000.csv and the per-pair series run_96_power.py cached in
data_output/ab-grid-96x96_power_v1000/ (M every H = 4305 dT, and the reference y'').
Same layout as 2026-11-05_ab-grid-96x96_modal_tau0.01/scripts/figures_96_modal.py,
with the reference y'' in place of the exact modal y''.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_96_power as rp               # sets the src path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from computational_tools import y2_spectrum as y2s

V = 1000.0
OUT = os.path.dirname(HERE)
AB, TAU = rp.AB, rp.TAU
summ = pd.read_csv(os.path.join(OUT, f"summary_96x96_{rp.tag(V)}.csv")).set_index(["a", "b"])


def label(a, b):
    return f"a={a:g}, b={b:g}"


series = {}
for a in AB:
    for b in AB:
        z = np.load(os.path.join(rp.cache_dir(V), f"a{a:g}_b{b:g}.npz"))
        t, H = z["t"], float(z["H"])
        y = np.log(z["M_DL"] + z["M_AL"])
        series[(a, b)] = dict(t=t, y=y, tc=t[1:-1], cd=(y[2:] - 2 * y[1:-1] + y[:-2]) / H ** 2,
                              t_ref=z["t_ref"], ref=z["y2_ref"], spec=y2s.spectrum(y, H), H=H)

head = f"96x96, v={V:g}, N=4, T=4, power engine"


def grid(name, title, draw, ylabel, xlabel="t"):
    """4x4 panels, rows a, columns b (the 10-07 layout); draw(ax, a, b, s) fills one."""
    fig, axes = plt.subplots(4, 4, figsize=(17.5, 13))
    for r, a in enumerate(AB):
        for c, b in enumerate(AB):
            ax = axes[r, c]
            ts = summ.loc[(a, b), "t_star"]
            ax.set_title(f"{label(a, b)}  t*=" + (f"{ts:.4f}" if np.isfinite(ts) else "none"),
                         fontsize=10)
            draw(ax, a, b, series[(a, b)])
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 3:
                ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, name), dpi=100)
    plt.close(fig)


def t_star_line(ax, a, b):
    ts = summ.loc[(a, b), "t_star"]
    if np.isfinite(ts):
        ax.axvline(ts, color="red", lw=1.4)


def tau_lines(ax):
    ax.axhline(0, color="k", lw=0.7)
    for sgn in (1, -1):
        ax.axhline(sgn * TAU, color="k", lw=0.8, ls=":", alpha=0.7)


def y2_panel(lim, reference=False):
    def draw(ax, a, b, s):
        if reference:
            ax.plot(s["tc"], s["cd"], color="C0", lw=3.0, alpha=0.35, label="central difference")
            ax.plot(s["t_ref"], s["ref"], color="C2", lw=1.0, label="reference")
            if (a, b) == (AB[0], AB[0]):
                ax.legend(fontsize=7, loc="lower right")
        else:
            ax.plot(s["tc"], s["cd"], color="C2", lw=1.0)
        tau_lines(ax)
        t_star_line(ax, a, b)
        ax.set_xlim(0, 4)
        ax.set_ylim(-lim, lim)
    return draw


H0 = series[(AB[0], AB[0])]["H"]
sub = (rf"red: $t^*$ ($\tau$ = {TAU:g}, reference $y''$);  dotted = $\pm\tau$;  "
       rf"central difference at H = {H0:.6e}")
for lim in (0.1, 0.01):
    grid(f"d2lnM_central_difference_ylim{lim:g}.png",
         rf"$y'' = d^2\ln M/dt^2$ by central difference -- {head}  (y-window $\pm${lim:g})" + "\n" + sub,
         y2_panel(lim), r"$d^2\ln M/dt^2$")
grid("d2lnM_reference_over_central_difference.png",
     rf"$y''$: reference (green; Richardson on central differences at 10H, 20H) over the central "
     rf"difference at H (wide blue) -- {head}  (y-window $\pm$0.1)" + "\n" + sub,
     y2_panel(0.1, reference=True), r"$d^2\ln M/dt^2$")


def lnM_panel(marked):
    def draw(ax, a, b, s):
        ax.plot(s["t"], s["y"], color="C0", lw=1.2)
        if marked:
            t_star_line(ax, a, b)
        ax.set_xlim(0, 4)
    return draw


for marked in (False, True):
    grid("lnM_t_star.png" if marked else "lnM.png",
         r"$\ln M(t)$ -- " + head + (rf";  red: $t^*$ ($\tau$ = {TAU:g})" if marked else ""),
         lnM_panel(marked), r"$\ln M$")

y2s.save_figure(os.path.join(OUT, "psd_y2.png"),
                [series[(a, b)]["spec"] for a in AB for b in AB],
                [label(a, b) for a in AB for b in AB], 4,
                rf"$|\mathcal{{F}}[y'']|^2$, y = ln M, linear axes, zoomed on the peak -- {head}"
                rf"  (H = {H0:.4e}, 8x zero-padded)")
print("figures written to", OUT)
