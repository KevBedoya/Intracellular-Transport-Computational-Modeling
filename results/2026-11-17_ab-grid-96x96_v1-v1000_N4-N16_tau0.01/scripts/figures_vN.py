"""Every figure of the four-(v, N) grid, from what run_vN.py wrote.

    June-2026/.venv/bin/python results/2026-11-17_ab-grid-96x96_v1-v1000_N4-N16_tau0.01/scripts/figures_vN.py

  m_5t_star/     M(5 t*) over (a, b): one heatmap per (v, N), and the four together
  m_5t_hat/      M(5 t^), t^ = 1/|lambda_2|: the same
  eigenvalues/   lambda_1 and lambda_2 over (a, b), per (v, N)
  mass_vs_time/  M(t) per (a, b) on [0, T], per (v, N): linear and log y
  d2lnM/         y'' = d^2 ln M/dt^2 by central difference at H with t* marked,
                 y-limits +-0.1 and +-0.01, per (v, N)

Heatmaps are drawn as heatmaps_96_power.py draws them (2026-11-12, the
reference heatmap_m_t_tilde_1_over_abs_lambda2_96x96_v1000_N16.png): rows a,
columns b, one sequential blue ramp, the value and its time in each cell. M is
coloured by log10 M when it spans a decade or more, by M itself otherwise. In the
four-panel figures rows are N (4, 16), columns are v (1, 1000), and the four
panels share one colour scale. Line figures are 4x4 panels in the 10-07 layout.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_vN as rv                       # sets the src path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT, AB, TAU, T = rv.OUT, rv.AB, rv.TAU, rv.T
CMAP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ALL = pd.read_csv(os.path.join(OUT, "data", "summary_96x96_all.csv"))
GRID_ORDER = [(4, 1.0), (4, 1000.0), (16, 1.0), (16, 1000.0)]   # 2x2: rows N, columns v


def case(v, N):
    return ALL[(ALL.v == v) & (ALL.N == N)].set_index(["a", "b"])


def arr(d, col):
    return np.array([[d.loc[(a, b), col] for b in AB] for a in AB])


def head(v, N):
    return rf"96x96, v = {v:g}, N = {N} evenly spaced, $\tau$ = 0.01;  "


# Titles do not name the engine: the modal and power engines agree to 7-10 digits
# at v = 1 (validation/v1_modal_vs_power.csv), so the figures describe the method only.
SRC = dict(t_star=r"$t^*$ by the right sweep of $|y''| = \tau$",
           lam=r"$\lambda_2$ from the eigenvalues of the one-step operator")


def sub(*paths):
    p = os.path.join(OUT, *paths)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save(fig, stem, exts=("png", "pdf"), dpi=150):
    for ext in exts:
        fig.savefig(sub(f"{stem}.{ext}"), dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- heatmaps

def sci(x):
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def nice_limits(z):
    """Limits rounded out to the leading digit of the span, and that digit's step."""
    span = z.max() - z.min()
    step = 10.0 ** np.floor(np.log10(span)) if span > 0 else 1.0
    return np.floor(z.min() / step) * step, np.ceil(z.max() / step) * step, step


def panel(fig, ax, z, cells, lo, hi, cbar_label, title, title_fs=11):
    """One 4x4 (a, b) table: rows a, columns b, colour z on [lo, hi].

    ``cells[i][j]`` = (main, sub): the bold line and the smaller one under it
    (sub may be "", and main is then centred).
    """
    im = ax.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, aspect="equal")
    for i in range(4):
        for j in range(4):
            main, sb = cells[i][j]
            ink = "white" if (z[i, j] - lo) / (hi - lo) > 0.5 else "#1a1a1a"
            ax.text(j, i - 0.13 if sb else i, main, ha="center", va="center",
                    color=ink, fontsize=11.5, fontweight="bold")
            if sb:
                ax.text(j, i + 0.2, sb, ha="center", va="center", color=ink, fontsize=9.5)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(4), [f"{b:g}" for b in AB])
    ax.set_yticks(range(4), [f"{a:g}" for a in AB])
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    ax.set_title(title, fontsize=title_fs)


def mass_scale(m):
    """(z, lo, hi, cell text, log?) for M values m (any shape), the reference rule."""
    if m.max() / m.min() >= 10:
        z = np.log10(m)
        return z, np.floor(z.min()), np.ceil(z.max()), sci, True
    lo, hi, step = nice_limits(m)
    nd = max(3, int(-np.log10(step)) + 2)
    return m, lo, hi, (lambda x: f"{x:.{nd}f}"), False


MASS_MAPS = [   # (folder, M column, time column, time TeX, definition, source key)
    ("m_5t_star", "m_at_5_t_star", "five_t_star", r"5t^*", "", "t_star"),
    ("m_5t_hat", "m_at_5_t_hat", "five_t_hat", r"5\hat{t}", r"$\hat{t} = 1/|\lambda_2|$", "lam"),
]

for folder, m_col, t_col, t_tex, t_def, src in MASS_MAPS:
    label = rf"M({t_tex})"
    # one heatmap per (v, N), on its own scale
    for v, N in rv.CASES:
        d = case(v, N)
        m, t = arr(d, m_col), arr(d, t_col)
        z, lo, hi, text, log = mass_scale(m)
        cells = [[(text(m[i, j]), rf"${t_tex}$ = {t[i, j]:.5f}") for j in range(4)] for i in range(4)]
        fig, ax = plt.subplots(figsize=(9.2, 7.6))
        panel(fig, ax, z, cells, lo, hi, rf"$\log_{{10}} {label}$" if log else rf"${label}$",
              rf"${label}$ per $(a, b)$, with ${t_tex}$ in each cell" + (t_def and ";  " + t_def)
              + "\n" + head(v, N) + SRC[src])
        fig.tight_layout()
        save(fig, os.path.join(folder, f"heatmap_{folder}_96x96_{rv.tag(v, N)}"))
    # the four together, on one scale
    ms = {(v, N): (arr(case(v, N), m_col), arr(case(v, N), t_col)) for v, N in rv.CASES}
    _, lo, hi, text, log = mass_scale(np.concatenate([m for m, _ in ms.values()]))
    fig, axes = plt.subplots(2, 2, figsize=(16.6, 15.4))
    for ax, (N, v) in zip(axes.flat, GRID_ORDER):
        m, t = ms[(v, N)]
        panel(fig, ax, np.log10(m) if log else m,
              [[(text(m[i, j]), rf"${t_tex}$ = {t[i, j]:.5f}") for j in range(4)] for i in range(4)],
              lo, hi, rf"$\log_{{10}} {label}$" if log else rf"${label}$",
              rf"v = {v:g}, N = {N}", title_fs=13)
    fig.suptitle(rf"${label}$ per $(a, b)$, with ${t_tex}$ in each cell" + (t_def and ";  " + t_def)
                 + r":  N = 4 (top), N = 16 (bottom);  v = 1 (left), v = 1000 (right)" + "\n"
                 + r"96x96, N evenly spaced, $\tau$ = 0.01;  " + SRC[src]
                 + r";  one colour scale for all four panels", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=3.5, w_pad=3.0)
    save(fig, os.path.join(folder, f"heatmap_{folder}_96x96_all4"), dpi=120)

# lambda_1 and lambda_2, per (v, N), each on its own scale
for v, N in rv.CASES:
    d = case(v, N)
    l1, re2, im2 = arr(d, "lambda_1"), arr(d, "lambda_2"), arr(d, "lambda_2_imag")
    src = "eigenvalues of the one-step operator"
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    panel(fig, ax, l1, [[(f"{l1[i, j]:.6f}", "") for j in range(4)] for i in range(4)],
          np.floor(l1.min()), np.ceil(l1.max()), r"$\lambda_1$",
          r"$\lambda_1$ per $(a, b)$: the slowest decay rate of $M$" + "\n" + head(v, N) + src)
    fig.tight_layout()
    save(fig, os.path.join("eigenvalues", f"heatmap_lambda_1_96x96_{rv.tag(v, N)}"))
    cells = [[(rf"{re2[i, j]:.3f} $\pm$ {im2[i, j]:.3f}$i$",
               rf"$|\lambda_2|$ = {np.hypot(re2[i, j], im2[i, j]):.4f}")
              if im2[i, j] > 0 else (f"{re2[i, j]:.6f}", "real")
              for j in range(4)] for i in range(4)]
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    panel(fig, ax, re2, cells, np.floor(re2.min()), np.ceil(re2.max()), r"$\mathrm{Re}(\lambda_2)$",
          r"$\lambda_2$ per $(a, b)$: the next excited decay rate; colour = $\mathrm{Re}(\lambda_2)$"
          + "\n" + head(v, N) + src)
    fig.tight_layout()
    save(fig, os.path.join("eigenvalues", f"heatmap_lambda_2_96x96_{rv.tag(v, N)}"))

# ---------------------------------------------------------------- 4x4 line panels

def load(v, N):
    out = {}
    for a in AB:
        for b in AB:
            z = np.load(os.path.join(rv.cache_dir(v, N), f"a{a:g}_b{b:g}.npz"))
            H = float(z["H"])
            y = np.log(z["M"])
            out[(a, b)] = dict(t=z["t"], M=z["M"], H=H, tc=z["t"][1:-1],
                               cd=(y[2:] - 2 * y[1:-1] + y[:-2]) / H ** 2)
    return out


def grid(stem, title, draw, ylabel, d, legend=None):
    """4x4 panels, rows a, columns b (the 10-07 layout); draw(ax, a, b, s, row) fills one."""
    fig, axes = plt.subplots(4, 4, figsize=(17.5, 13))
    for r, a in enumerate(AB):
        for c, b in enumerate(AB):
            ax = axes[r, c]
            row = d.loc[(a, b)]
            ts = row.t_star
            ax.set_title(f"a={a:g}, b={b:g}  t*=" + (f"{ts:.4f}" if np.isfinite(ts) else "none"),
                         fontsize=10)
            draw(ax, a, b, row)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 3:
                ax.set_xlabel("t")
    if legend:
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=len(legend),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.99, 0.925))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.905 if legend else 0.95))
    save(fig, stem, exts=("png",), dpi=100)


for v, N in rv.CASES:
    d, series, tg = case(v, N), load(v, N), rv.tag(v, N)
    H0 = series[(AB[0], AB[0])]["H"]
    hd = f"96x96, v={v:g}, N={N} evenly spaced, T=4"

    # y'' with t*, at two y-windows
    def y2_draw(lim):
        def draw(ax, a, b, row):
            s = series[(a, b)]
            ax.plot(s["tc"], s["cd"], color="C2", lw=1.0)
            ax.axhline(0, color="k", lw=0.7)
            for sgn in (1, -1):
                ax.axhline(sgn * TAU, color="k", lw=0.8, ls=":", alpha=0.7)
            if np.isfinite(row.t_star):
                ax.axvline(row.t_star, color="red", lw=1.4)
            ax.set_xlim(0, T)
            ax.set_ylim(-lim, lim)
        return draw

    for lim in (0.1, 0.01):
        grid(os.path.join("d2lnM", f"d2lnM_central_difference_ylim{lim:g}_96x96_{tg}"),
             rf"$y'' = d^2\ln M/dt^2$ by central difference -- {hd}  (y-window $\pm${lim:g})" + "\n"
             + rf"red: $t^*$ ($\tau$ = {TAU:g}, right sweep of $|y''| = \tau$);  dotted = $\pm\tau$;  "
             rf"central difference at H = {H0:.6e}",
             y2_draw(lim), r"$d^2\ln M/dt^2$", d)

    # M(t), with t*, 5 t* and 5 t^ where they fall inside [0, T]
    def m_draw(logy):
        def draw(ax, a, b, row):
            s = series[(a, b)]
            ax.plot(s["t"], s["M"], color="C0", lw=1.4, label=r"$M(t)$")
            for x, kw in ((row.t_star, dict(color="red", lw=1.4, label=r"$t^*$")),
                          (row.five_t_star, dict(color="red", lw=1.4, ls="--", label=r"$5t^*$")),
                          (row.five_t_hat, dict(color="#6a3d9a", lw=1.4, ls="-.",
                                                label=r"$5\hat{t}$, $\hat{t} = 1/|\lambda_2|$"))):
                if np.isfinite(x) and x <= T:
                    ax.axvline(x, **kw)
                elif (a, b) == (AB[0], AB[0]):
                    ax.plot([], [], **kw)          # keep the legend entry
            ax.set_xlim(0, T)
            if logy:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.02)
            ax.text(0.97, 0.95, rf"$5t^*$ = {row.five_t_star:.3f}" + "\n"
                    + rf"$5\hat{{t}}$ = {row.five_t_hat:.4f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        return draw

    for logy in (False, True):
        grid(os.path.join("mass_vs_time", f"M_vs_t{'_logy' if logy else ''}_96x96_{tg}"),
             r"$M(t)$" + (" (log y)" if logy else "") + f" -- {hd}" + "\n"
             + rf"$t^*$ ($\tau$ = {TAU:g}), $5t^*$ and $5\hat{{t}}$ are drawn where they fall "
             rf"inside [0, T]; their values are printed in each panel",
             m_draw(logy), r"$M$", d, legend=[1, 2, 3, 4])
    print("figures written for", tg, flush=True)

print("all figures written to", OUT)
