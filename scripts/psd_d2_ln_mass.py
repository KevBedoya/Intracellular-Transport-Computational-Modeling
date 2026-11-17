"""Power spectral density of g(t) = d^2/dt^2 ln M(t), and how it relates to t*.

M(t) is total mass (diffusive + advective layer). g is computed from the same
mass timeseries the characteristic-time kernel uses, and both definitions of t*
are recomputed from that series -- the original secant through log10 M at
t = 0.4, 0.5 (in force until 2026-09-28), and the current right sweep of
|g| <= tau, by calling launch._char_time_onset itself -- so both sides of every
comparison come from one solve.

    python scripts/psd_d2_ln_mass.py solve      # all runs -> data_output/psd_d2_ln_mass/raw
    python scripts/psd_d2_ln_mass.py report     # CSVs + figures -> results/2026-10-15_psd-d2-ln-mass

What it establishes (derivations in findings/psd_d2_ln_mass_report.pdf):

  * int g dt = -lambda, so the spectrum at f = 0 is S(0) = lambda^2, where
    lambda = -ln(10) * fit_slope is the asymptotic decay rate either kernel fits;
  * the original t* is the centroid of g -- exactly, for the finite fit window,
    once g is weighted by the window kernel below;
  * when g <= 0 (escape rate rising monotonically) |G(f)| <= |G(0)|, so the
    PSD peaks at f = 0. A PSD cannot see a time shift, so absolute times live
    in the phase, not the PSD;
  * the current t* is set by the slowest transient mode of g's tail,
    g ~ C exp(-gamma t) cos(omega t): t* = ln(C / tau) / gamma for a real mode,
    and within half a period of it for a complex one, so dt*/d ln(1/tau) =
    1/gamma. That mode is the PSD's resonance at omega / 2 pi when it dominates
    the spectrum, which it does only for slow switching (small a, b <= 1).
"""
import argparse
import glob
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_output" / "psd_d2_ln_mass" / "raw"
OUT_DIR = ROOT / "results" / "2026-10-15_psd-d2-ln-mass"

LN10 = np.log(10.0)
X1, X2 = 0.4, 0.5          # the original t* fit window (CHAR_TIME_FIT_X1/_X2 before 2026-09-28)
DIFF_DT = 2e-5             # sample spacing g is differenced at (see derivatives())

# The sweep: every (N, v, w) at 32x32 with a = b = w, run to T = 2 so the pulse in g
# has fully decayed, plus pure diffusion (one microtubule, switching off).
SWEEP_G, SWEEP_T = 32, 2.0
SWEEP = ([(SWEEP_G, 1, 0.0, 0.0, 0.0, SWEEP_T)] +
         [(SWEEP_G, n, v, w, w, SWEEP_T)
          for n in (4, 8, 16)
          for v in (10.0, 100.0, 1000.0, 10000.0)
          for w in (1.0, 10.0, 100.0, 1000.0)])
# The N = 16 grid-study configuration (results/2026-08-18_char-time-grid-study_N16)
# at increasing resolution, to T = 1 as in that study.
GRIDS = [(g, 16, 10000.0, 100.0, 100.0, 1.0) for g in (32, 48, 64, 80)]
# The (a, b) grid of the current t* studies (results/2026-10-07_ab-grid-96x96_tau0.01:
# v = 1, N = 4), at 32x32. T = 6 because with b <= 1 y'' oscillates for a long time
# before settling -- the regime where the PSD of g has a peak away from f = 0.
AB_RATES = (0.1, 1.0, 10.0, 100.0)
AB = ([(32, 4, 1.0, a, b, 6.0) for a in AB_RATES for b in AB_RATES] +
      # the oscillating b = 0.1 column again at 48^2, to check the tail-mode
      # relations are not a property of the coarse grid
      [(48, 4, 1.0, a, 0.1, 6.0) for a in AB_RATES])


# ----------------------------------------------------------------------------- solve

def _n_list(g, n):
    """Evenly spaced ray indices, as scripts/grid_study_char_time.py:n_list."""
    if n == 1:
        return [0]
    step = g - (g // n)
    return [int(i * step / (n - 1)) for i in range(n)]


def _raw_name(cfg):
    g, n, v, a, b, T = cfg
    return f"G{g}_N{n}_v{v:g}_a{a:g}_b{b:g}_T{T:g}.npz"


def _package_on_path():
    os.environ.setdefault("ITCM_OUTPUT_ROOT", os.path.join(tempfile.gettempdir(), "itcm_psd"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    for p in (ROOT / "src" / "intracellular_transport", ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _solve_one(cfg, cf=5):
    """One CPU solve through the same path as launch._char_time_point."""
    _package_on_path()
    from computational_tools import numerical_tools as num
    from computational_tools import supplements as sup
    from launch_functions import launch

    g, n, v, a, b, T = cfg
    n_list = np.array(_n_list(g, n), dtype=np.int64)
    rk = int(np.floor(num.compute_K(g, g, T) / cf))
    series = [np.zeros(rk) for _ in range(5)]
    d_layer, a_layer = sup.initialize_layers(g, g)
    t0 = time.time()
    launch._solve_mass_analysis("cpu", series, g, g, a, b, v, T, n_list, d_layer,
                                a_layer, cf, rk, 0.0, 1.0, 1.0, 10**12, True, 0, 0)
    out = RAW_DIR / _raw_name(cfg)
    np.savez(out, t=np.arange(rk) * cf * num.compute_dT(g, g), TM=series[4],
             AL=series[1], G=g, N=n, v=v, a=a, b=b, T=T, cf=cf,
             wall=time.time() - t0)
    return f"{out.name}  ({time.time() - t0:.0f} s)"


def cmd_solve(args):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    configs = {"all": SWEEP + GRIDS + AB, "sweep": SWEEP, "grids": GRIDS,
               "ab": AB}[args.set]
    todo = [c for c in configs
            if args.force or not (RAW_DIR / _raw_name(c)).exists()]
    # Heaviest first, so the large grids are not left running alone at the end.
    todo.sort(key=lambda c: -c[0] ** 6 * c[5])
    print(f"{len(todo)} solves, {args.workers} at a time")
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(_solve_one, todo):
            print("  done", line)


# ----------------------------------------------------------------------------- analyse

def derivatives(t, M, target_dt=DIFF_DT):
    """h = (ln M)' and g = (ln M)'' by central differences at ~target_dt.

    The solver samples far more finely than any feature of ln M, and differencing
    at that spacing only amplifies round-off (~eps / dt^2). Taking every k-th
    sample is exact -- no filtering -- so the signal itself is untouched.
    """
    stride = max(1, int(round(target_dt / (t[1] - t[0]))))
    ts, ms = t[::stride], M[::stride]
    dt = ts[1] - ts[0]
    ell = np.log(ms)
    h = np.gradient(ell, dt, edge_order=2)
    return ts, dt, ell, h, np.gradient(h, dt, edge_order=2)


def spectrum(g, dt, pad=16):
    """G(f) = int g(t) exp(-2 pi i f t) dt on a one-sided axis, zero padded.

    g is a transient, so this is its energy spectral density |G|^2; a periodogram
    differs only by the constant 1/T, which moves nothing.
    """
    nfft = 1 << int(np.ceil(np.log2(len(g) * pad)))
    return np.fft.rfftfreq(nfft, dt), np.fft.rfft(g, nfft) * dt


def analyse(path):
    d = np.load(path)
    ts, dt, ell, h, g = derivatives(d["t"], d["TM"])

    # The original t*: secant through log10 M at the window ends, evaluated at
    # the samples nearest 0.4 and 0.5 -- the same two points the window-kernel
    # identity below uses, so the two can be compared exactly.
    i1, i2 = int(round(X1 / dt)), int(round(X2 / dt))
    xa, xb = ts[i1], ts[i2]
    slope = (ell[i2] - ell[i1]) / LN10 / (xb - xa)
    t_star = -(ell[i2] / LN10 - slope * xb) / slope

    # Moments of g by parts, so differencing noise in g cannot bias them:
    #   int_0^T g = h(T) - h(0),   int_0^T t g = T h(T) - ell(T) + ell(0)
    i0 = h[-1] - h[0]
    t_centroid = (ts[-1] * h[-1] - ell[-1] + ell[0]) / i0
    # The exact finite-window counterpart of the secant (the report's eq. for t*_win).
    w = np.where(ts <= xa, 1.0, np.where(ts <= xb, (xb - ts) / (xb - xa), 0.0))
    t_kernel = np.trapz(np.minimum(ts, xa) * w * g, ts) / np.trapz(w * g, ts)
    var = np.trapz(ts ** 2 * g, ts) / i0 - t_centroid ** 2

    f, G = spectrum(g, dt)
    S = np.abs(G) ** 2
    f_half = f[np.argmax(S <= S[0] / 2)]
    phase = np.unwrap(np.angle(G))
    nlow = max(3, int(np.searchsorted(f, 0.05 * f_half)))
    group_delay = -np.polyfit(2 * np.pi * f[:nlow], phase[:nlow], 1)[0]

    k = int(np.argmin(g))
    big = np.abs(g) > 1e-3 * abs(g[k])
    return dict(
        G=int(d["G"]), N=int(d["N"]), v=float(d["v"]), a=float(d["a"]),
        b=float(d["b"]), T=float(d["T"]),
        t_star=t_star, fit_slope=slope, lam_fit=-slope * LN10,
        S0=S[0], sqrt_S0=np.sqrt(S[0]), f_peak=f[int(np.argmax(S))],
        f_half=f_half, t_centroid=t_centroid, t_kernel=t_kernel,
        group_delay=group_delay, sigma=np.sqrt(var) if var > 0 else np.nan,
        t_g_min=ts[k], g_min=g[k],
        g_sign_changes=int(np.sum(np.diff(np.sign(g[big])) != 0)),
        g_positive_fraction=np.trapz(np.clip(g, 0, None), ts) / np.trapz(np.abs(g), ts),
        _ts=ts, _ell=ell, _g=g, _f=f, _S=S,
    )


def continuum_diffusion(t_max=2.0, n_t=400_001, n_modes=400):
    """Point source at the centre of the unit disc, absorbing rim, D = 1.

    M(t) = sum c_n exp(-j_n^2 t), c_n = 2 / (j_n J1(j_n)), j_n the zeros of J0.
    Exact reference: t* = ln(c_1) / j_1^2 and lambda = j_1^2.
    """
    from scipy.special import j1, jn_zeros
    j = jn_zeros(0, n_modes)
    c, lam = 2 / (j * j1(j)), j ** 2
    t = np.linspace(0, t_max, n_t)
    E = np.exp(-np.outer(t[1:], lam))
    # Most of E underflows to zero by design; Accelerate's matmul flags that
    # spuriously as overflow/invalid, though the products are exact.
    with np.errstate(all="ignore"):
        M, M1, M2 = E @ c, -(E @ (c * lam)), E @ (c * lam ** 2)
    g = np.concatenate([[0.0], M2 / M - (M1 / M) ** 2])
    g[t < 2e-3] = 0.0   # series unconverged as t -> 0, where g is ~0 to all orders anyway
    dt = t[1] - t[0]
    f, G = spectrum(g, dt, pad=4)
    S = np.abs(G) ** 2
    return dict(t=t, g=g, f=f, S=S, lam=lam[0], t_star=np.log(c[0]) / lam[0],
                t_centroid=np.trapz(t * g, t) / np.trapz(g, t),
                f_half=f[np.argmax(S <= S[0] / 2)])


# ----------------------------------------------------------------------------- tail modes

TAUS = np.geomspace(1e-4, 3e-2, 25)   # thresholds the current t* is swept over


def matrix_pencil(y, dt, order):
    """Poles s_k and amplitudes A_k of y(t) ~ sum_k A_k exp(s_k t), t = 0, dt, ...

    The matrix-pencil method: an SVD of the Hankel matrix of y, truncated to
    `order`, whose shift-invariance gives the poles as eigenvalues.
    """
    n = len(y)
    L = n // 2
    Y = np.array([y[i:i + L + 1] for i in range(n - L)])
    V = np.linalg.svd(Y, full_matrices=False)[2][:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    s = np.log(z.astype(complex)) / dt
    A = np.linalg.lstsq(np.exp(np.outer(np.arange(n) * dt, s)), y, rcond=None)[0]
    return s, A


def tail_mode(ts, g, t_from, t_to, dt_mp=0.004, order=6):
    """The slowest-decaying mode of g on [t_from, t_to]: g ~ C exp(-gamma t) cos(omega t + phi).

    Returns (gamma, omega, C) with the envelope C referenced to t = 0. Poles with
    a negligible amplitude, or too close to the Nyquist rate to be real, are
    dropped before choosing.
    """
    sel = (ts >= t_from) & (ts <= t_to)
    stride = max(1, int(round(dt_mp / (ts[1] - ts[0]))))
    y = g[sel][::stride]
    dt = (ts[1] - ts[0]) * stride
    s, A = matrix_pencil(y, dt, order)
    env = np.abs(A)
    keep = (env > 1e-3 * env.max()) & (np.abs(s.imag) < 0.25 * np.pi / dt)
    s, env = s[keep], env[keep]
    k = int(np.argmax(s.real))
    gamma, omega = -s[k].real, abs(s[k].imag)
    c_from = env[k] * (2.0 if omega > 1e-6 else 1.0)    # a conjugate pair carries 2|A|
    return gamma, omega, c_from * np.exp(gamma * ts[sel][0])


def analyse_ab(path):
    """analyse() plus the current t* criterion and the tail mode that sets it."""
    _package_on_path()
    from launch_functions import launch

    r = analyse(path)
    d = np.load(path)
    mass, sample_dt = d["TM"], float(d["t"][1] - d["t"][0])

    def onset(tau):
        return launch._char_time_onset(mass, sample_dt, tau=tau)

    swept = np.array([onset(tau)["t_star"] for tau in TAUS])
    ok = np.isfinite(swept)
    gamma_tau = 1.0 / np.polyfit(np.log(1 / TAUS[ok]), swept[ok], 1)[0]
    gamma, omega, C = tail_mode(r["_ts"], r["_g"], np.nanmin(swept), np.nanmax(swept))
    cur2, cur3 = onset(1e-2), onset(1e-3)
    p = r["g_positive_fraction"]
    r.update(
        t_star_legacy=r["t_star"],
        t_star_tau1e2=cur2["t_star"], t_star_tau1e3=cur3["t_star"],
        lam_steady=-cur3["fit_slope"] * LN10,
        gamma_tau=gamma_tau, gamma_tail=gamma, f_tail=omega / (2 * np.pi), C_tail=C,
        t_env_tau1e2=np.log(C / 1e-2) / gamma, t_env_tau1e3=np.log(C / 1e-3) / gamma,
        half_period=np.pi / omega if omega > 1e-6 else 0.0,
        S_max_over_S0=float(r["_S"].max() / r["S0"]),
        S_bound=1.0 / (1.0 - 2.0 * p) ** 2,
        _swept=swept,
    )
    return r


# ----------------------------------------------------------------------------- report

# Reference categorical slots 1-3 (dataviz palette), one per microtubule count;
# three series is the all-pairs-safe limit for a scatter.
N_COLOR = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a"}
N_MARK = {4: "o", 8: "s", 16: "^"}
GRID_RAMP = {32: "#86b6ef", 48: "#3987e5", 64: "#256abf", 80: "#104281"}
INK, INK2, GRIDLINE = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(color=GRIDLINE, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)


def _fig_anatomy(grid_rows, cont, path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for r in grid_rows:
        col, lab = GRID_RAMP[r["G"]], f"{r['G']}²"
        ts = r["_ts"]
        line = r["fit_slope"] * (ts - r["t_star"])
        axs[0].plot(ts, r["_ell"] / LN10, color=col, lw=2, label=lab)
        axs[0].plot(ts[ts <= X2], line[ts <= X2], color=col, lw=1, ls="--")
        axs[0].plot([r["t_star"]], [0], "o", color=col, ms=6, mec="white", mew=1, zorder=4)
        axs[1].plot(ts, r["_g"], color=col, lw=2, label=lab)
        axs[1].axvline(r["t_star"], color=col, lw=1, ls=":")
        sel = r["_f"] <= 80
        axs[2].plot(r["_f"][sel][1:], (r["_S"] / r["S0"])[sel][1:], color=col, lw=2, label=lab)
    sel = cont["f"] <= 80
    axs[2].plot(cont["f"][sel][1:], (cont["S"] / cont["S"][0])[sel][1:], color=INK2,
                lw=1.2, ls="--", label="pure diffusion\n(continuum)")

    ax = axs[0]
    ax.set_xlim(0, 0.15)
    tmax = 0.15
    lowest = min(np.min(r["_ell"][r["_ts"] <= tmax]) for r in grid_rows) / LN10
    highest = max(-r["fit_slope"] * r["t_star"] for r in grid_rows)
    ax.set_ylim(1.1 * lowest, 1.6 * highest)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\log_{10} M(t)$")
    ax.set_title("(a) $\\log_{10}M$ near $t=0$. Dashed: the secant through the\n"
                 "fit window [0.4, 0.5], extrapolated; it crosses 0 at $t^*$ (dot)",
                 fontsize=10, color=INK, loc="left")

    ax = axs[1]
    ax.set_xlim(0, 0.2)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$g(t)=\mathrm{d}^2\ln M/\mathrm{d}t^2$")
    ax.set_title(r"(b) $g(t)$; dotted: $t^*$, which is the centroid of $g$",
                 fontsize=10, color=INK, loc="left")

    ax = axs[2]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 80)
    ax.set_ylim(1e-6, 2)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.text(0.12, 0.3, r"half power, defines $f_{1/2}$", fontsize=8.5, color=INK2)
    ax.set_xlabel(r"$f$  (cycles per unit $t$)")
    ax.set_ylabel(r"$S(f)/S(0)$,   $S(0)=\lambda^2$")
    ax.set_title(r"(c) PSD of $g$: maximum at $f=0$", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", title="grid", title_fontsize=8.5)
    for ax in axs:
        _style(ax)
    fig.suptitle(r"$N=16$, $v=10^4$, $a=b=100$ (the grid-study configuration)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fig_sweep(rows, fit, path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for n in (4, 8, 16):
        rs = [r for r in rows if r["N"] == n]
        kw = dict(color=N_COLOR[n], marker=N_MARK[n], s=34, label=f"N = {n}",
                  edgecolors="white", linewidths=0.8, zorder=3)
        axs[0].scatter([r["lam_fit"] for r in rs], [r["sqrt_S0"] for r in rs], **kw)
        axs[1].scatter([r["t_star"] for r in rs], [r["t_kernel"] for r in rs], **kw)
        axs[2].scatter([r["f_half"] for r in rs], [r["t_star"] for r in rs], **kw)

    lo, hi = 1e-3, 10
    axs[0].plot([lo, hi], [lo, hi], color=INK2, lw=0.8, zorder=1)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\lambda=-\ln 10\times$ (fit slope)")
    axs[0].set_ylabel(r"$\sqrt{S(0)}$")
    dev = max(abs(r["sqrt_S0"] / r["lam_fit"] - 1) for r in rows)
    axs[0].set_title(r"(a) PSD peak height $=\lambda^2$" + f"\nmax rel. dev. {dev:.1e}",
                     fontsize=10, color=INK, loc="left")

    lo, hi = 0.03, 0.105
    axs[1].plot([lo, hi], [lo, hi], color=INK2, lw=0.8, zorder=1)
    axs[1].set_xlim(lo, hi)
    axs[1].set_ylim(lo, hi)
    axs[1].set_xlabel(r"$t^*$ (secant through $\log_{10}M$ at 0.4, 0.5)")
    axs[1].set_ylabel(r"window-weighted centroid of $g$")
    dev = max(abs(r["t_kernel"] / r["t_star"] - 1) for r in rows)
    dev_c = max(abs(r["t_centroid"] / r["t_star"] - 1) for r in rows)
    axs[1].set_title(r"(b) $t^*$ is the centroid of $g$" +
                     f"\nmax rel. dev. {dev:.0e} (window), {dev_c:.0e} (plain)",
                     fontsize=10, color=INK, loc="left")

    A, B, rms = fit
    ff = np.geomspace(2.5, 14, 50)
    axs[2].plot(ff, A * ff ** B, color=INK2, lw=1, zorder=1)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel(r"$f_{1/2}$, half-power frequency of the PSD")
    axs[2].set_ylabel(r"$t^*$")
    axs[2].set_title(rf"(c) empirical: $t^*\approx{A:.3f}\,f_{{1/2}}^{{{B:.3f}}}$" +
                     f"\nrms scatter {100 * rms:.1f}%", fontsize=10, color=INK, loc="left")
    axs[2].legend(frameon=False, fontsize=8.5, loc="upper right")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    for axis, ticks in ((axs[2].xaxis, [3, 4, 6, 8, 10, 12]),
                        (axs[2].yaxis, [0.04, 0.05, 0.06, 0.08, 0.1])):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    for ax in axs:
        _style(ax)
    fig.suptitle(r"Sweep at $32^2$: $N\in\{4,8,16\}$, $v\in\{10,\dots,10^4\}$, "
                 r"$a=b\in\{1,\dots,10^3\}$ (48 runs)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Tail types for the (a, b) figures, in reference categorical slots 1-3.
TAIL_TYPES = (("real tail mode", "#2a78d6", "o"),
              ("oscillating tail mode", "#eb6834", "s"),
              ("two close tail modes", "#1baf7a", "^"))


def tail_type(r):
    if r["f_tail"] < 0.01:
        return TAIL_TYPES[0][0]
    if abs(r["gamma_tau"] / r["gamma_tail"] - 1) > 0.15:
        return TAIL_TYPES[2][0]
    return TAIL_TYPES[1][0]


AB_EXAMPLES = ((0.1, 100.0), (1.0, 0.1), (100.0, 0.1))


def _fig_ab_examples(rows, path):
    import matplotlib.pyplot as plt
    pick = {(r["a"], r["b"]): r for r in rows if r["G"] == 32}
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2))
    blurbs = {(0.1, 100.0): "real tail mode: g decays without oscillating",
              (1.0, 0.1): "oscillating tail: PSD resonance at the tail frequency",
              (100.0, 0.1): "biphasic early pulse dominates the PSD"}
    for j, ab in enumerate(AB_EXAMPLES):
        r = pick[ab]
        ts, g, f, S = r["_ts"], r["_g"], r["_f"], r["_S"]
        ax = axs[0, j]
        t_end = min(r["T"], 1.25 * r["t_env_tau1e3"] + 0.3)
        sel = ts <= t_end
        ax.semilogy(ts[sel], np.abs(g[sel]), color="#2a78d6", lw=1.4, label=r"$|g(t)|$")
        tt = np.linspace(0, t_end, 400)
        ax.semilogy(tt, r["C_tail"] * np.exp(-r["gamma_tail"] * tt), color=INK2, lw=1,
                    ls="--", label=r"tail envelope $Ce^{-\gamma t}$")
        for tau, key, mk in ((1e-2, "t_star_tau1e2", "o"), (1e-3, "t_star_tau1e3", "s")):
            ax.axhline(tau, color=INK2, lw=0.7, ls=":")
            ax.plot([r[key]], [tau], mk, color="#eb6834", ms=7, mec="white", mew=1, zorder=4,
                    label=rf"$t^*_\tau$, $\tau=10^{{{int(np.log10(tau))}}}$")
        ax.set_ylim(1e-5, 3e2)
        ax.set_xlim(0, t_end)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|g(t)|$")
        ax.set_title(f"({'abc'[j]}) a = {ab[0]:g}, b = {ab[1]:g}\n{blurbs[ab]}",
                     fontsize=10, color=INK, loc="left")
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax = axs[1, j]
        sel = f <= 8
        ax.plot(f[sel], S[sel] / r["S0"], color="#2a78d6", lw=1.6)
        k = int(np.argmax(S[sel]))
        ax.plot([f[sel][k]], [S[sel][k] / r["S0"]], "o", color="#2a78d6", ms=6,
                mec="white", mew=1, zorder=4)
        if r["f_tail"] > 0.01:
            ax.axvline(r["f_tail"], color="#eb6834", lw=1, ls="--")
            ax.text(r["f_tail"], 0.03, "  tail frequency", color=INK2, fontsize=8.5,
                    transform=ax.get_xaxis_transform(), va="bottom")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$f$")
        ax.set_ylabel(r"$S(f)/S(0)$")
        ax.set_title(f"({'def'[j]}) PSD: maximum at f = {f[sel][k]:.2f}, "
                     f"{S[sel][k] / r['S0']:.2g} × S(0)", fontsize=10, color=INK, loc="left")
    for ax in axs.flat:
        _style(ax)
    fig.suptitle(r"The $(a,b)$ grid of the current $t^*$ studies at $32^2$: $v=1$, $N=4$, $T=6$",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fig_ab_relations(rows, path):
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["G"] == 32]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for name, col, mk in TAIL_TYPES:
        rs = [r for r in rows if tail_type(r) == name]
        if not rs:
            continue
        kw = dict(color=col, marker=mk, s=38, edgecolors="white", linewidths=0.8, zorder=3)
        axs[0].scatter([r["gamma_tail"] for r in rs], [r["gamma_tau"] for r in rs],
                       label=name, **kw)
        osc = [r for r in rs if r["f_tail"] > 0.01]
        if osc:
            axs[1].scatter([r["f_tail"] for r in osc], [r["f_peak"] for r in osc], **kw)
        for key_m, key_p, fill in (("t_star_tau1e2", "t_env_tau1e2", col),
                                   ("t_star_tau1e3", "t_env_tau1e3", "white")):
            x = np.array([r[key_p] for r in rs])
            y = np.array([r[key_m] for r in rs])
            hp = np.array([r["half_period"] for r in rs])
            axs[2].errorbar(x - hp / 2, y, xerr=hp / 2, fmt="none", ecolor=col, elinewidth=1,
                            alpha=0.6, zorder=2)
            axs[2].scatter(x, y, color=fill, edgecolors=col, linewidths=1.2, marker=mk,
                           s=38, zorder=3)

    lo, hi = 1, 40
    axs[0].plot([lo, hi], [lo, hi], color=INK2, lw=0.8, zorder=1)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\gamma$, decay rate of the slowest tail mode of $g$")
    axs[0].set_ylabel(r"$\left[\mathrm{d}t^*_\tau/\mathrm{d}\ln(1/\tau)\right]^{-1}$")
    axs[0].set_title(r"(a) $t^*_\tau$ moves by $\ln 10/\gamma$ per decade of $\tau$",
                     fontsize=10, color=INK, loc="left")
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    # b = 0.1 and b = 1 land nearly on top of each other at every a, so they are
    # labelled together, one label per a.
    osc = [r for r in rows if r["f_tail"] > 0.01]
    for a in AB_RATES:
        grp = [r for r in osc if r["a"] == a and r["b"] <= 1]
        if grp:
            axs[1].annotate(f"a = {a:g}, b ≤ 1",
                            (max(r["f_tail"] for r in grp), np.mean([r["f_peak"] for r in grp])),
                            xytext=(7, -3), textcoords="offset points", fontsize=7.5, color=INK2)
    for r in osc:
        if r["b"] > 1:
            # (10, 10) sits just above the a = 10 group, so its label goes left
            left = (r["a"], r["b"]) == (10.0, 10.0)
            axs[1].annotate(f"({r['a']:g}, {r['b']:g})", (r["f_tail"], r["f_peak"]),
                            xytext=(-7 if left else 7, -3), textcoords="offset points",
                            fontsize=7.5, color=INK2, ha="right" if left else "left")
    axs[1].plot([0, 3.5], [0, 3.5], color=INK2, lw=0.8, zorder=1)
    axs[1].set_xlim(0, 1.3)
    axs[1].set_ylim(-0.1, 3.6)
    axs[1].set_xlabel(r"tail frequency $\omega/2\pi$")
    axs[1].set_ylabel(r"frequency of the PSD maximum")
    axs[1].set_title("(b) the PSD maximum is the tail frequency\nonly for slow switching",
                     fontsize=10, color=INK, loc="left")

    axs[2].plot([0, 5.2], [0, 5.2], color=INK2, lw=0.8, zorder=1)
    axs[2].set_xlabel(r"$\ln(C/\tau)/\gamma$, when the tail envelope reaches $\tau$")
    axs[2].set_ylabel(r"$t^*_\tau$ from launch._char_time_onset")
    axs[2].set_title("(c) $t^*_\\tau$ vs the envelope prediction; filled $\\tau=10^{-2}$,\n"
                     "hollow $10^{-3}$; bars: the half period before it",
                     fontsize=10, color=INK, loc="left")
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sci(x, digits=1):
    """1.2e-04 -> 1.2\\times10^{-4}, for LaTeX tables."""
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def _write_tex_tables(sweep, grid_rows, ab_rows):
    """LaTeX fragments the findings report \\input's, so its numbers cannot drift
    from the data they summarise."""
    out = OUT_DIR / "tex"
    out.mkdir(exist_ok=True)

    def stat(vals):
        a = np.abs(np.asarray(vals))
        return f"${_sci(a.max())}$ & ${_sci(np.median(a))}$"

    rows = [
        (r"$\sqrt{S(0)}$ against $\lambda=-\ln10\times$ fit slope",
         stat([r["sqrt_S0"] / r["lam_fit"] - 1 for r in sweep])),
        (r"window-weighted centroid of $g$ against $t^*_{\rm sec}$",
         stat([r["t_kernel"] / r["t_star"] - 1 for r in sweep])),
        (r"plain centroid of $g$ against $t^*_{\rm sec}$",
         stat([r["t_centroid"] / r["t_star"] - 1 for r in sweep])),
        (r"group delay of $G$ at $f=0$ against $t^*_{\rm sec}$",
         stat([r["group_delay"] / r["t_star"] - 1 for r in sweep])),
    ]
    n_dc = sum(r["f_peak"] == 0 for r in sweep)
    lines = [r"\begin{tabular}{lll}", r"\toprule",
             r"check & max.\ rel.\ deviation & median \\", r"\midrule"]
    lines += [f"{name} & {s} \\\\" for name, s in rows]
    lines += [rf"PSD maximum at $f=0$ & {n_dc} of {len(sweep)} runs & \\",
              r"\bottomrule", r"\end{tabular}"]
    (out / "table_identities.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
             r"grid & $t^*_{\rm sec}$ & centroid of $g$ & $\lambda=\sqrt{S(0)}$ & $S(0)$ "
             r"& $f_{1/2}$ & $\sigma$ & $t^*_{\rm sec}f_{1/2}$ \\", r"\midrule"]
    for r in grid_rows:
        lines.append(rf"${r['G']}^2$ & {r['t_star']:.6f} & {r['t_centroid']:.6f} & "
                     rf"{r['sqrt_S0']:.4f} & {r['S0']:.4f} & {r['f_half']:.2f} & "
                     rf"{r['sigma']:.4f} & {r['t_star'] * r['f_half']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_grid.tex").write_text("\n".join(lines) + "\n")

    short = {TAIL_TYPES[0][0]: "real", TAIL_TYPES[1][0]: "osc.", TAIL_TYPES[2][0]: "two"}

    def num(x, fmt):
        return "--" if not np.isfinite(x) else format(x, fmt)

    ab32 = sorted([r for r in ab_rows if r["G"] == 32], key=lambda r: (r["a"], r["b"]))
    lines = [r"\begin{tabular}{rrlrrrrrrrrrrr}", r"\toprule",
             r"$a$ & $b$ & tail & $p$ & $f_{\max}$ & $S_{\max}/S_0$ & $\omega/2\pi$ & "
             r"$\gamma$ & $\gamma_\tau$ & $t^*_{10^{-2}}$ & $t_{\rm env}$ & "
             r"$t^*_{10^{-3}}$ & $t_{\rm env}$ & $t^*_{\rm sec}$ \\", r"\midrule"]
    for r in ab32:
        lines.append(
            rf"{r['a']:g} & {r['b']:g} & {short[tail_type(r)]} & "
            rf"{r['g_positive_fraction']:.3f} & {r['f_peak']:.2f} & "
            rf"{r['S_max_over_S0']:.2f} & {num(r['f_tail'], '.3f') if r['f_tail'] > 0.01 else '0'} & "
            rf"{r['gamma_tail']:.2f} & {r['gamma_tau']:.2f} & "
            rf"{r['t_star_tau1e2']:.3f} & {r['t_env_tau1e2']:.3f} & "
            rf"{r['t_star_tau1e3']:.3f} & {r['t_env_tau1e3']:.3f} & "
            rf"{r['t_star_legacy']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_ab.tex").write_text("\n".join(lines) + "\n")

    by = {(r["G"], r["a"], r["b"]): r for r in ab_rows}
    lines = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
             r"$a$ & grid & $f_{\max}$ & $\omega/2\pi$ & $\gamma$ & $\gamma_\tau$ & "
             r"$t^*_{10^{-2}}$ & $t_{\rm env}$ & $t^*_{10^{-3}}$ & $t_{\rm env}$ \\", r"\midrule"]
    for a in AB_RATES:
        for g in (32, 48):
            r = by.get((g, a, 0.1))
            if r is None:
                continue
            lines.append(
                rf"{a:g} & ${g}^2$ & {r['f_peak']:.2f} & {r['f_tail']:.3f} & "
                rf"{r['gamma_tail']:.2f} & {r['gamma_tau']:.2f} & "
                rf"{r['t_star_tau1e2']:.3f} & {r['t_env_tau1e2']:.3f} & "
                rf"{r['t_star_tau1e3']:.3f} & {r['t_env_tau1e3']:.3f} \\")
        if a != AB_RATES[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_ab_grid.tex").write_text("\n".join(lines) + "\n")


def cmd_report(args):
    import warnings
    import pandas as pd
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(RAW_DIR / "*.npz")))
    ab_names = {_raw_name(c) for c in AB}
    rows = [analyse(p) for p in paths if Path(p).name not in ab_names]
    ab_rows = [analyse_ab(p) for p in paths if Path(p).name in ab_names]
    if not rows:
        sys.exit(f"no solves under {RAW_DIR}; run `solve` first")
    cont = continuum_diffusion()

    if ab_rows:
        cols = ["G", "a", "b", "t_star_tau1e2", "t_star_tau1e3", "t_env_tau1e2",
                "t_env_tau1e3", "half_period", "gamma_tau", "gamma_tail", "f_tail",
                "C_tail", "f_peak", "S_max_over_S0", "S_bound", "g_positive_fraction",
                "lam_steady", "sqrt_S0", "t_star_legacy", "t_kernel", "t_centroid"]
        dfa = pd.DataFrame([{k: r[k] for k in cols} for r in ab_rows])
        dfa["tail_type"] = [tail_type(r) for r in ab_rows]
        dfa.sort_values(["G", "a", "b"]).to_csv(OUT_DIR / "ab_metrics.csv", index=False)
        _fig_ab_examples(ab_rows, OUT_DIR / "ab_examples.png")
        _fig_ab_relations(ab_rows, OUT_DIR / "ab_relations.png")
        print(dfa.sort_values(["G", "a", "b"]).to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df["tstar_x_fhalf"] = df.t_star * df.f_half
    df["tstar_over_sigma"] = df.t_star / df.sigma
    df = df.sort_values(["G", "N", "v", "a", "T"])
    df.to_csv(OUT_DIR / "psd_metrics.csv", index=False)

    sweep = [r for r in rows if r["G"] == SWEEP_G and r["T"] == SWEEP_T and r["N"] > 1]
    x, y = np.log([r["f_half"] for r in sweep]), np.log([r["t_star"] for r in sweep])
    B, logA = np.polyfit(x, y, 1)
    rms = float(np.std(y - logA - B * x))
    _fig_sweep(sweep, (np.exp(logA), B, rms), OUT_DIR / "sweep_identities.png")

    grid_rows = sorted([r for r in rows if (r["G"], r["N"], r["v"], r["a"], r["T"]) in
                        {(c[0], c[1], c[2], c[3], c[5]) for c in GRIDS}], key=lambda r: r["G"])
    if grid_rows:
        _fig_anatomy(grid_rows, cont, OUT_DIR / "anatomy_grid_study_config.png")
    sweep_all = [r for r in rows if r["G"] == SWEEP_G and r["T"] == SWEEP_T]
    _write_tex_tables(sweep_all, grid_rows, ab_rows)

    print(f"{len(rows)} runs -> {OUT_DIR}")
    print(f"continuum diffusion: t* = ln(c1)/lam1 = {cont['t_star']:.6f}, centroid = "
          f"{cont['t_centroid']:.6f}, lam1 = {cont['lam']:.6f}, sqrt S(0) = "
          f"{np.sqrt(cont['S'][0]):.6f}, f_half = {cont['f_half']:.4f}")
    print(f"sweep fit: t* = {np.exp(logA):.4f} f_half^{B:.4f}, rms {100 * rms:.2f}%")
    cols = ["G", "N", "v", "a", "t_star", "lam_fit", "sqrt_S0", "t_centroid", "t_kernel",
            "group_delay", "f_peak", "f_half", "sigma", "tstar_x_fhalf"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.6g}"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="run the sweep and grid solves")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--force", action="store_true", help="re-solve runs already on disk")
    s.add_argument("--set", choices=("all", "sweep", "grids", "ab"), default="all",
                   help="which solves to run (grids: 64^2 ~8 min, 80^2 ~30 min on 1 core)")
    s.set_defaults(fn=cmd_solve)
    r = sub.add_parser("report", help="analyse saved solves; write CSV and figures")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
