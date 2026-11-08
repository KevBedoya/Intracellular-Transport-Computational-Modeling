"""The 96x96 (a, b) grid at one velocity on the power engine: every per-pair number
of the 2026-11-05 modal study.

    June-2026/.venv/bin/python results/2026-11-08_ab-grid-96x96_v1000_tau0.01/scripts/run_96_power.py V OUT_DIR

Fixed as on 11-05: 96x96, N = 4 evenly spaced (N_LIST = [0, 24, 48, 72]),
tau = 0.01, T = 4, (a, b) in {0.1, 1, 10, 100}^2.

Per pair (power_engine.PowerEngine):

  * M(t) every H = 4305 dT ~ 1.0005e-4 (the 10-07 differencing step and the
    11-05 Mass Analysis sampling) over [0, T], here at exactly k = j * 4305 steps;
  * the reference y'' = d^2 ln M / dt^2: Richardson extrapolation of the central
    differences of ln M at 10 H and 20 H, (4 D(10H) - D(20H)) / 3, whose error
    (round-off ~1e-13 / (10H)^2 plus O((10H)^4) truncation) stays near 1e-6.
    It takes the place of the modal engine's exact y'';
  * t*: the right sweep of the 11-05 char-time job on that y'' -- the last
    crossing of |y''| = tau, linearly interpolated between samples;
  * central_difference_t_star: launch._char_time_onset (the 10-07 time-stepper
    rule) on the samples at spacing H, as figures_96_modal.py did;
  * m* = M(t*) from the samples (cubic spline of ln M), and again by binary
    powers (m_at_t_star_recomputed); M at 5 t*, 10 t*, 5/Re(lambda_2),
    5/|lambda_2| and 1/|lambda_2| by binary powers;
  * lambda_1, lambda_2 (PowerEngine.rates) and their condition numbers;
  * fit_slope: least squares of log10 M over [t*, T], as the modal t* path;
  * the PSD columns, y2_spectrum.spectrum of ln M at H over [0, T].

Writes OUT_DIR/summary_96x96_<tag>.csv, m_5t_star_96x96_<tag>.csv and
m_t_tilde_96x96_<tag>.csv (tag = v<V>), and per pair the series (t, M_DL, M_AL)
and the reference y'' to data_output/ab-grid-96x96_power_v<V>/ for the figures.
"""
import os, sys
# one BLAS thread per worker: 8 workers each with a full BLAS pool oversubscribe the cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(R, "src", "intracellular_transport"), os.path.join(R, "src")]

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

AB = [0.1, 1.0, 10.0, 100.0]
G, NL, T, TAU = 96, [0, 24, 48, 72], 4.0, 0.01
S = 4305                 # steps per sample: H = S dT ~ 1.0005e-4
RICH = 10                # reference y'' from central differences at RICH*H and 2*RICH*H
T_TILDE = [("5_over_re", 5.0, "re"), ("5_over_abs", 5.0, "abs"), ("1_over_abs", 1.0, "abs")]


def tag(v):
    return f"v{v:g}"


def cache_dir(v):
    return os.path.join(R, "data_output", f"ab-grid-96x96_power_{tag(v)}")


def reference_y2(y, H, m=RICH):
    """(sample indices j in [2m, J - 2m), y'' there): Richardson on spacings mH and 2mH."""
    J = y.size
    d1 = (y[2 * m:] - 2 * y[m:-m] + y[:-2 * m]) / (m * H) ** 2              # j = m .. J-m-1
    d2 = (y[4 * m:] - 2 * y[2 * m:-2 * m] + y[:-4 * m]) / (2 * m * H) ** 2  # j = 2m .. J-2m-1
    return np.arange(2 * m, J - 2 * m), (4 * d1[m:-m] - d2) / 3


def last_crossing(t, y2, tau):
    """The right sweep: the last crossing of |y''| = tau, linear between samples.

    NaN when |y''| > tau at the last sample (the decay has not settled by T).
    """
    over = ~(np.abs(y2) <= tau)
    if over[-1]:
        return float("nan")
    if not over.any():
        print(f"*** |y''| <= tau from t = {t[0]:g} on; t* reported there")
        return float(t[0])
    i = int(np.nonzero(over)[0][-1])
    level = tau * np.sign(y2[i])
    return float(t[i] + (level - y2[i]) * (t[i + 1] - t[i]) / (y2[i + 1] - y2[i]))


def analyse(args):
    a, b, v = args
    warnings.filterwarnings("ignore")
    import power_engine as pe
    from computational_tools import y2_spectrum as y2s
    from launch_functions import launch

    t0 = time.perf_counter()
    e = pe.PowerEngine(G, a, b, v, NL)
    H = S * e.dT
    J = int(T / H) + 1
    DA = e.samples(S, J)
    M = DA.sum(axis=1)
    t = np.arange(J) * H
    y = np.log(M)
    jR, y2 = reference_y2(y, H)
    tR = t[jR]
    ts = last_crossing(tR, y2, TAU)
    rates = e.rates()
    l2, l2abs = rates["lambda_2"], np.hypot(rates["lambda_2"], rates["lambda_2_imag"])
    t_tilde = {key: K / (l2 if part == "re" else l2abs) for key, K, part in T_TILDE}

    row = dict(a=a, b=b, t_star=ts)
    if np.isfinite(ts):
        m_ts, m_5, m_10, *m_tt = e.mass([ts, 5 * ts, 10 * ts] + list(t_tilde.values()))
        m_star = float(np.exp(CubicSpline(t, y)(ts)))
        tail = t > ts
        slope = np.polyfit(np.concatenate([[ts], t[tail]]),
                           np.log10(np.concatenate([[m_star], M[tail]])), 1)[0]
    else:
        m_tt = list(e.mass(list(t_tilde.values())))
        m_ts = m_5 = m_10 = m_star = slope = float("nan")
    rule = launch._char_time_onset(M, H, tau=TAU, spacing=H)
    spec = y2s.spectrum(y, H)
    cd = (y[2:] - 2 * y[1:-1] + y[:-2]) / H ** 2                           # at t[1:-1]
    late = tR > 0.05
    row.update(m_star=m_star, m_at_t_star_recomputed=m_ts, ten_t_star=10 * ts,
               m_at_10_t_star=m_10, five_t_star=5 * ts, m_at_5_t_star=m_5,
               central_difference_t_star=rule["t_star"], **rates,
               lambda_2_abs=l2abs, fit_slope=float(slope), **y2s.summary(spec),
               max_abs_reference_minus_cd_after_0p05=float(np.max(np.abs(y2 - cd[jR - 1])[late])),
               M_at_T=float(M[-1]))
    for key, _, _ in T_TILDE:
        row[f"t_tilde_{key}"] = t_tilde[key]
    for (key, _, _), m in zip(T_TILDE, m_tt):
        row[f"m_at_t_tilde_{key}"] = float(m)

    os.makedirs(cache_dir(v), exist_ok=True)
    np.savez(os.path.join(cache_dir(v), f"a{a:g}_b{b:g}.npz"),
             t=t, M_DL=DA[:, 0], M_AL=DA[:, 1], t_ref=tR, y2_ref=y2, H=H, dT=e.dT)
    row["wall_s"] = time.perf_counter() - t0
    print(f"v={v:g} a={a:g} b={b:g}: t*={ts:.6f} lambda_1={rates['lambda_1']:.6f} "
          f"lambda_2={l2:.4f}+-{rates['lambda_2_imag']:.4f}i  ({row['wall_s']:.1f} s)", flush=True)
    return row


SUMMARY_COLUMNS = ["a", "b", "t_star", "m_star", "m_at_t_star_recomputed", "ten_t_star",
                   "m_at_10_t_star", "central_difference_t_star", "lambda_1", "lambda_2",
                   "lambda_2_imag", "kappa_lambda_1", "kappa_lambda_2", "fit_slope",
                   "psd_omega_peak", "psd_peak", "psd_fwhm", "psd_fwhm_side",
                   "max_abs_reference_minus_cd_after_0p05", "M_at_T"]


def run(v, out_dir, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        d = pd.DataFrame(list(ex.map(analyse, [(a, b, v) for a in AB for b in AB])))
    os.makedirs(out_dir, exist_ok=True)
    d[SUMMARY_COLUMNS].to_csv(os.path.join(out_dir, f"summary_96x96_{tag(v)}.csv"), index=False)
    d[["a", "b", "t_star", "five_t_star", "m_at_5_t_star"]].to_csv(
        os.path.join(out_dir, f"m_5t_star_96x96_{tag(v)}.csv"), index=False)
    d[["a", "b", "lambda_2", "lambda_2_imag", "lambda_2_abs"]
      + [c for key, _, _ in T_TILDE for c in (f"t_tilde_{key}", f"m_at_t_tilde_{key}")]].to_csv(
        os.path.join(out_dir, f"m_t_tilde_96x96_{tag(v)}.csv"), index=False)
    return d


if __name__ == "__main__":
    v, out = float(sys.argv[1]), sys.argv[2]
    t0 = time.perf_counter()
    d = run(v, out)
    print(d[SUMMARY_COLUMNS].to_string(index=False))
    print(f"total {time.perf_counter() - t0:.1f} s; wrote {out}")
