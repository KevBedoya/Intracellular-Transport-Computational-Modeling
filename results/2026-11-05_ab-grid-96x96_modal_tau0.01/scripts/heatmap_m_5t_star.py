"""M(5 t*) over the (a, b) grid as a heatmap, from the modal series.

    June-2026/.venv/bin/python results/2026-11-05_ab-grid-96x96_modal_tau0.01/scripts/heatmap_m_5t_star.py [--recompute]

t* per pair is read from summary_96x96_modal.csv (the char-time job's t*).
M(5 t*) comes from the spectral (eigen-modal) expansion of the 96x96 one-step operator
(mass_analysis_modal.solve_modes + ln_mass), as
M(10 t*) did in figures_96_modal.py: 5 t* runs from about 2 to 12, so it is past
T = 4 for most pairs. The same evaluation at t* and 10 t* is checked against the
summary's m* and M(10 t*).

The 16 modal solves (~20 s each on one core) are cached in m_5t_star_96x96_modal.csv;
--recompute redoes them.
"""
import os, sys
# one BLAS thread per worker: 8 workers each with a full BLAS pool oversubscribe the cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
R = os.path.abspath(os.path.join(OUT, "..", ".."))
sys.path[:0] = [os.path.join(R, "src", "intracellular_transport"), os.path.join(R, "src")]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

AB = [0.1, 1.0, 10.0, 100.0]
G, V, NL = 96, 1.0, [0, 24, 48, 72]
CSV = os.path.join(OUT, "m_5t_star_96x96_modal.csv")


def solve(args):
    a, b, ts = args
    import warnings
    warnings.filterwarnings("ignore")
    from computational_tools import mass_analysis_modal as mam
    modes = mam.solve_modes(G, G, a, b, V, NL)
    m1, m5, m10 = np.exp(mam.ln_mass(modes, [ts, 5 * ts, 10 * ts]))
    return dict(a=a, b=b, t_star=ts, five_t_star=5 * ts, m_at_5_t_star=m5,
                m_at_t_star_recomputed=m1, m_at_10_t_star_recomputed=m10)


def compute():
    s = pd.read_csv(os.path.join(OUT, "summary_96x96_modal.csv"))
    todo = [(r.a, r.b, r.t_star) for r in s.itertuples()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        d = pd.DataFrame(list(ex.map(solve, todo)))
    d = d.merge(s[["a", "b", "m_star", "m_at_10_t_star"]], on=["a", "b"])
    d["rel_diff_m_star"] = np.abs(d.m_at_t_star_recomputed / d.m_star - 1)
    d["rel_diff_m_10_t_star"] = np.abs(d.m_at_10_t_star_recomputed / d.m_at_10_t_star - 1)
    d = d[["a", "b", "t_star", "five_t_star", "m_at_5_t_star", "m_star", "m_at_t_star_recomputed",
           "rel_diff_m_star", "m_at_10_t_star", "m_at_10_t_star_recomputed", "rel_diff_m_10_t_star"]]
    d.to_csv(CSV, index=False)
    return d


def sci(x):
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


HEAD = r"96x96, v = 1, N = 4 evenly spaced, $\tau$ = 0.01, modal engine;  "
# sequential blue ramp, light (small) -> dark (large)
CMAP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def grid(z, cells, lo, hi, cbar_label, title, stem):
    """A 4x4 (a, b) heatmap: rows a, columns b, colour z (4x4) on [lo, hi].

    ``cells[i][j]`` = (main, sub): the bold line and the smaller one under it
    (sub may be "", and main is then centred). Writes OUT/<stem>.png and .pdf.
    """
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, aspect="equal")
    for i in range(4):
        for j in range(4):
            main, sub = cells[i][j]
            ink = "white" if (z[i, j] - lo) / (hi - lo) > 0.5 else "#1a1a1a"
            ax.text(j, i - 0.13 if sub else i, main, ha="center", va="center",
                    color=ink, fontsize=11.5, fontweight="bold")
            if sub:
                ax.text(j, i + 0.2, sub, ha="center", va="center", color=ink, fontsize=9.5)
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
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=150)
    plt.close(fig)


def heatmap(d, m_col, t_col, t_tex, subtitle, stem, t_def="", log=True):
    """M at a per-pair time over the 4x4 (a, b) grid, colour log10 M.

    Each cell shows M (column ``m_col``) and the time (``t_col``, typeset as ``t_tex``).
    ``t_def`` (e.g. the time's definition) follows the first title line.
    ``log=False`` colours by M itself, on the data range rounded out to 0.05 -- for
    M that spans well under a decade, where whole-decade log limits leave every cell
    the same shade. Writes OUT/<stem>.png and .pdf.
    """
    d = d.set_index(["a", "b"])
    m = np.array([[d.loc[(a, b), m_col] for b in AB] for a in AB])
    t = np.array([[d.loc[(a, b), t_col] for b in AB] for a in AB])
    z = np.log10(m) if log else m
    step = 1.0 if log else 0.05
    lo, hi = np.floor(z.min() / step) * step, np.ceil(z.max() / step) * step
    cells = [[(sci(m[i, j]), rf"${t_tex}$ = {t[i, j]:.4f}") for j in range(4)] for i in range(4)]
    grid(z, cells, lo, hi, rf"$\log_{{10}} M({t_tex})$" if log else rf"$M({t_tex})$",
         rf"$M({t_tex})$ per $(a, b)$, with ${t_tex}$ in each cell" + (t_def and ";  " + t_def)
         + "\n" + HEAD + subtitle, stem)


def plot(d):
    heatmap(d, "m_at_5_t_star", "five_t_star", r"5t^*",
            r"$t^*$ from the exact modal $y''$, $M$ from the spectral (eigen-modal) series",
            "heatmap_m_5t_star_96x96_modal")


if __name__ == "__main__":
    d = compute() if ("--recompute" in sys.argv or not os.path.exists(CSV)) else pd.read_csv(CSV)
    print(d.to_string(index=False))
    plot(d)
