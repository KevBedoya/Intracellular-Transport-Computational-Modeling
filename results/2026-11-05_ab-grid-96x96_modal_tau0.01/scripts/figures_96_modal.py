"""Figures for the modal 96x96 study: every (a, b) pair on one axes per figure.

    python figures_96_modal.py jobs_96_modal.json OUT_DIR

M(t) comes from the 16 Mass Analysis jobs (one sample every H ~ 1.0e-4, the
10-07 differencing step); t*, lambda_1/2 and the PSD columns from the (a, b)
char-time job. The exact modal y'' is recomputed here from the same kernel
(mass_analysis_modal), since no job writes that curve out.
"""
import glob, json, os, sys
R = r"N:\work\Intracellular-Transport-Computational-Modeling"
sys.path[:0] = [os.path.join(R, "src", "intracellular_transport"), os.path.join(R, "src")]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from computational_tools import mass_analysis_modal as mam, y2_spectrum as y2s
from launch_functions import launch

jobs = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
JOBS = os.path.join(R, "data_output", "jobs")
AB = [0.1, 1.0, 10.0, 100.0]
TAU, G, NL = 0.01, 96, [0, 24, 48, 72]
COLORS = dict(zip(AB, ["C0", "C1", "C2", "C3"]))          # by a
STYLES = dict(zip(AB, ["-", "--", "-.", ":"]))            # by b


def job_dir(job_id):
    hits = [d for d in os.listdir(JOBS) if d.startswith(job_id)]
    assert len(hits) == 1, (job_id, hits)
    return os.path.join(JOBS, hits[0])


def label(a, b):
    return f"a={a:g}, b={b:g}"


# --- load -------------------------------------------------------------------
ct_csv = glob.glob(os.path.join(job_dir(jobs["char_time"]), "**", "ab_grid_char_t_*.csv"),
                   recursive=True)
assert len(ct_csv) == 1, ct_csv
ct = pd.read_csv(ct_csv[0]).set_index(["a", "b"])

series = {}
for a in AB:
    for b in AB:
        jd = job_dir(jobs["mass_analysis"][f"{a:g},{b:g}"])
        csv = glob.glob(os.path.join(jd, "**", "total", "**", "MA_total_*.csv"), recursive=True)
        assert len(csv) == 1, (a, b, csv)
        df = pd.read_csv(csv[0])
        t, M = df["T"].to_numpy(), df["M"].to_numpy()
        H = t[1] - t[0]
        y = np.log(M)
        cd = (y[2:] - 2 * y[1:-1] + y[:-2]) / H ** 2
        modes = mam.solve_modes(G, G, a, b, 1.0, NL)
        _, _, exact, _ = mam.ln_mass_derivatives(modes, t[1:-1])
        spec = y2s.spectrum(y, H)
        # the 10-07 rule (right sweep on the central difference) on these samples
        rule = launch._char_time_onset(M, H, tau=TAU, spacing=H)
        ts = ct.loc[(a, b), "t_star"]
        # M at t* (checks against the job's m*) and at 10 t*, from the modal series
        m_ts, m_10ts = (np.exp(mam.ln_mass(modes, [ts, 10 * ts])) if np.isfinite(ts)
                        else (np.nan, np.nan))
        series[(a, b)] = dict(t=t, y=y, tc=t[1:-1], cd=cd, exact=exact, spec=spec, H=H,
                              cd_t_star=rule["t_star"], m_ts=m_ts, m_10ts=m_10ts)
        print(f"{label(a, b)}: H={H:.6e}, t*={ct.loc[(a, b), 't_star']:.6f}, "
              f"central-difference rule t*={rule['t_star']:.6f}", flush=True)


head = "96x96, v=1, N=4, T=4, modal engine"


def grid(name, title, draw, ylabel, xlabel="t", sharey=False):
    """4x4 panels, rows a, columns b (the 10-07 layout); draw(ax, a, b, s) fills one."""
    fig, axes = plt.subplots(4, 4, figsize=(17.5, 13), sharey=sharey)
    for r, a in enumerate(AB):
        for c, b in enumerate(AB):
            ax = axes[r, c]
            ts = ct.loc[(a, b), "t_star"]
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
    ts = ct.loc[(a, b), "t_star"]
    if np.isfinite(ts):
        ax.axvline(ts, color="red", lw=1.4)


def tau_lines(ax):
    ax.axhline(0, color="k", lw=0.7)
    for sgn in (1, -1):
        ax.axhline(sgn * TAU, color="k", lw=0.8, ls=":", alpha=0.7)


def y2_panel(lim, exact=False):
    def draw(ax, a, b, s):
        if exact:
            ax.plot(s["tc"], s["cd"], color="C0", lw=3.0, alpha=0.35, label="central difference")
            ax.plot(s["tc"], s["exact"], color="C2", lw=1.0, label="exact modal")
            if (a, b) == (AB[0], AB[0]):
                ax.legend(fontsize=7, loc="lower right")
        else:
            ax.plot(s["tc"], s["cd"], color="C2", lw=1.0)
        tau_lines(ax)
        t_star_line(ax, a, b)
        ax.set_xlim(0, 4)
        ax.set_ylim(-lim, lim)
    return draw


sub = (rf"red: $t^*$ ($\tau$ = {TAU:g}, exact modal $y''$);  dotted = $\pm\tau$;  "
       rf"central difference at H = {series[(AB[0], AB[0])]['H']:.6e}")
for lim in (0.1, 0.01):
    grid(f"d2lnM_central_difference_ylim{lim:g}.png",
         rf"$y'' = d^2\ln M/dt^2$ by central difference -- {head}  (y-window $\pm${lim:g})" + "\n" + sub,
         y2_panel(lim), r"$d^2\ln M/dt^2$")
grid("d2lnM_exact_over_central_difference.png",
     rf"$y''$: exact modal (green) over the central difference (wide blue) -- {head}  (y-window $\pm$0.1)"
     + "\n" + sub, y2_panel(0.1, exact=True), r"$d^2\ln M/dt^2$")


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

# --- PSD: the y2_spectrum panels (peak, half maximum, FWHM), zoomed on each peak ---------
y2s.save_figure(os.path.join(OUT, "psd_y2.png"),
                [series[(a, b)]["spec"] for a in AB for b in AB],
                [label(a, b) for a in AB for b in AB], 4,
                rf"$|\mathcal{{F}}[y'']|^2$, y = ln M, linear axes, zoomed on the peak -- {head}"
                rf"  (H = {series[(AB[0], AB[0])]['H']:.4e}, 8x zero-padded)")

# --- table --------------------------------------------------------------------------
rows = []
for (a, b), s in series.items():
    r = ct.loc[(a, b)]
    rows.append(dict(a=a, b=b, t_star=r["t_star"], m_star=r["m_star"],
                     m_at_t_star_recomputed=s["m_ts"], ten_t_star=10 * r["t_star"],
                     m_at_10_t_star=s["m_10ts"],
                     central_difference_t_star=s["cd_t_star"],
                     lambda_1=r["lambda_1"], lambda_2=r["lambda_2"], lambda_2_imag=r["lambda_2_imag"],
                     fit_slope=r["fit_slope"],
                     psd_omega_peak=r["psd_omega_peak"], psd_peak=r["psd_peak"],
                     psd_fwhm=r["psd_fwhm"], psd_fwhm_side=r["psd_fwhm_side"],
                     max_abs_exact_minus_cd_after_0p05=float(np.max(np.abs(
                         s["exact"] - s["cd"])[s["tc"] > 0.05]))))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary_96x96_modal.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(OUT)
