"""The 96x96 (a, b) grid at four (v, N): v in {1, 1000} x N in {4, 16} evenly spaced, tau = 0.01.

    June-2026/.venv/bin/python results/2026-11-17_ab-grid-96x96_v1-v1000_N4-N16_tau0.01/scripts/run_vN.py

Per (v, N, a, b), with (a, b) in {0.1, 1, 10, 100}^2 and T = 4:

  * t*: the characteristic-time rule of the modal char-time job -- the last
    crossing of |y''| = tau, y = ln M, on the right sweep;
  * lambda_1, lambda_2 (Re, |Im|): the slowest decay rate and the next excited
    one, from the eigenvalues of the symmetry-reduced one-step operator
    (launch._slowest_rates's rule);
  * t^ = 1/|lambda_2|, and M at 5 t* and at 5 t^;
  * M(t) every H = 4305 dT ~ 1.0005e-4 over [0, T], cached for the figures.

Engine:

  * v = 1: the modal expansion itself (mass_analysis_modal.solve_modes):
    t*, lambda_1, lambda_2 from launch._char_time_onset_modal on the exact modal
    y''; M from the modal sum (mam.ln_mass).
  * v = 1000: the modal sum cannot be evaluated -- cond(V) is ~1e22-1e25, so the
    modal amplitudes are round-off (2026-11-08 and 2026-11-12 READMEs; cond(V) is
    recorded here again in the column cond_V). The power engine (power_engine.py,
    copied unchanged from 2026-11-12) evaluates the same operator by exact powers:
    t* by the same right sweep on the reference y'' (Richardson on central
    differences at 10H and 20H), lambda_1, lambda_2 from the eigenvalues of the
    same operator, M by binary powers.

At v = 1 the power engine is run as well, and the two are compared in
validation/v1_modal_vs_power.csv. validation/vs_earlier_runs.csv compares each
(v, N) with the run already in results/ (11-05, 11-08, 11-12).

Writes data/summary_96x96_<tag>.csv per (v, N) (tag = v<V>_N<N>),
data/summary_96x96_all.csv, the two validation CSVs, and per pair the series
(t, M, and the reference y'' of the power engine) to
data_output/ab-grid-96x96_vN_<tag>/ for the figures.
"""
import os, sys
# one BLAS thread per worker: 8 workers each with a full BLAS pool oversubscribe the cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.dirname(OUT)
R = os.path.dirname(RES)
sys.path[:0] = [HERE, os.path.join(R, "src", "intracellular_transport"), os.path.join(R, "src")]

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

AB = [0.1, 1.0, 10.0, 100.0]
CASES = [(1.0, 4), (1000.0, 4), (1.0, 16), (1000.0, 16)]     # (v, N)
ENGINE = {1.0: "modal", 1000.0: "power"}
G, T, TAU = 96, 4.0, 0.01
S = 4305                 # steps per sample: H = S dT ~ 1.0005e-4
RICH = 10                # reference y'' from central differences at RICH*H and 2*RICH*H
NUMBERS = ["t_star", "five_t_star", "m_at_5_t_star", "m_star", "lambda_1", "lambda_2",
           "lambda_2_imag", "lambda_2_abs", "t_hat", "five_t_hat", "m_at_5_t_hat"]


def n_list(N):
    return list(range(0, G, G // N))


def tag(v, N):
    return f"v{v:g}_N{N}"


def cache_dir(v, N):
    return os.path.join(R, "data_output", f"ab-grid-96x96_vN_{tag(v, N)}")


def reference_y2(y, H, m=RICH):
    """(sample indices j in [2m, J - 2m), y'' there): Richardson on spacings mH and 2mH."""
    d1 = (y[2 * m:] - 2 * y[m:-m] + y[:-2 * m]) / (m * H) ** 2              # j = m .. J-m-1
    d2 = (y[4 * m:] - 2 * y[2 * m:-2 * m] + y[:-4 * m]) / (2 * m * H) ** 2  # j = 2m .. J-2m-1
    return np.arange(2 * m, y.size - 2 * m), (4 * d1[m:-m] - d2) / 3


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


def derived(ts, l1, l2, l2i, mass_at):
    """The NUMBERS columns, with mass_at(times) -> M at those times."""
    l2abs = float(np.hypot(l2, l2i))
    times = [x for x in (ts, 5 * ts, 5 / l2abs) if np.isfinite(x)]
    m = dict(zip(times, mass_at(times))) if times else {}
    nan = float("nan")
    return dict(t_star=ts, five_t_star=5 * ts, m_at_5_t_star=m.get(5 * ts, nan), m_star=m.get(ts, nan),
                lambda_1=l1, lambda_2=l2, lambda_2_imag=l2i, lambda_2_abs=l2abs,
                t_hat=1 / l2abs, five_t_hat=5 / l2abs, m_at_5_t_hat=m.get(5 / l2abs, nan))


def analyse(args):
    v, N, a, b = args
    warnings.filterwarnings("ignore")
    import power_engine as pe
    from computational_tools import mass_analysis_modal as mam
    from launch_functions import launch

    t0 = time.perf_counter()
    NL = n_list(N)
    e = pe.PowerEngine(G, a, b, v, NL)
    H = S * e.dT
    J = int(T / H) + 1
    t = np.arange(J) * H

    # power engine: the v = 1000 numbers, and the cross-check at v = 1
    Mp = e.samples(S, J).sum(axis=1)
    jR, y2R = reference_y2(np.log(Mp), H)
    r = e.rates()
    power = derived(last_crossing(t[jR], y2R, TAU), r["lambda_1"], r["lambda_2"], r["lambda_2_imag"],
                    lambda ts: e.mass(ts))

    modes = mam.solve_modes(G, G, a, b, v, NL)
    if ENGINE[v] == "modal":
        cm = launch._char_time_onset_modal(modes, T, tau=TAU)
        modal = derived(cm["t_star"], cm["lambda_1"], cm["lambda_2"], cm["lambda_2_imag"],
                        lambda ts: np.exp(mam.ln_mass(modes, np.sort(ts)))[np.argsort(np.argsort(ts))])
        M = np.empty(J)
        M[0] = e.w @ e.x0                       # the modal sum drops the beta = 0 modes, so not at k = 0
        M[1:] = np.exp(mam.ln_mass(modes, t[1:]))
        main = modal
    else:
        modal, M, main = None, Mp, power

    row = dict(v=v, N=N, a=a, b=b, engine=ENGINE[v], **main,
               central_difference_t_star=launch._char_time_onset(M, H, tau=TAU, spacing=H)["t_star"],
               M_at_T=float(M[-1]), cond_V=modes["cond_V"],
               kappa_lambda_1=r["kappa_lambda_1"], kappa_lambda_2=r["kappa_lambda_2"])
    if modal is not None:
        row.update({f"power_{k}": power[k] for k in NUMBERS})

    os.makedirs(cache_dir(v, N), exist_ok=True)
    np.savez(os.path.join(cache_dir(v, N), f"a{a:g}_b{b:g}.npz"),
             t=t, M=M, H=H, dT=e.dT, t_ref=t[jR], y2_ref=y2R)
    row["wall_s"] = time.perf_counter() - t0
    print(f"v={v:g} N={N} a={a:g} b={b:g} [{ENGINE[v]}]: t*={row['t_star']:.6f} "
          f"lambda_1={row['lambda_1']:.6f} lambda_2={row['lambda_2']:.4f}+-{row['lambda_2_imag']:.4f}i "
          f"M(5t*)={row['m_at_5_t_star']:.4e} M(5t^)={row['m_at_5_t_hat']:.6f} "
          f"cond(V)={row['cond_V']:.1e} ({row['wall_s']:.1f} s)", flush=True)
    return row


COLUMNS = ["v", "N", "a", "b", "engine"] + NUMBERS + [
    "central_difference_t_star", "M_at_T", "cond_V", "kappa_lambda_1", "kappa_lambda_2"]

EARLIER = {   # (v, N): (directory, file tag) of the run already in results/
    (1.0, 4): ("2026-11-05_ab-grid-96x96_modal_tau0.01", "modal"),
    (1000.0, 4): ("2026-11-08_ab-grid-96x96_v1000_tau0.01", "v1000"),
    (1000.0, 16): ("2026-11-12_ab-grid-96x96_v1000_N16_tau0.01", "v1000_N16"),
}


def rel(x, y):
    return np.abs(np.asarray(x, float) / np.asarray(y, float) - 1)


def validate(d):
    os.makedirs(os.path.join(OUT, "validation"), exist_ok=True)
    # 1. v = 1: the modal expansion against the power engine
    m = d[d.engine == "modal"]
    c = m[["v", "N", "a", "b"]].copy()
    c["t_star_modal"], c["t_star_power"] = m.t_star, m.power_t_star
    c["abs_diff_t_star"] = np.abs(m.t_star - m.power_t_star)
    for k in ("m_at_5_t_star", "m_at_5_t_hat", "m_star", "lambda_1", "lambda_2", "lambda_2_abs"):
        c[f"rel_diff_{k}"] = rel(m[k], m[f"power_{k}"])
    c["abs_diff_lambda_2_imag"] = np.abs(m.lambda_2_imag - m.power_lambda_2_imag)
    c["cond_V"] = m.cond_V
    c.to_csv(os.path.join(OUT, "validation", "v1_modal_vs_power.csv"), index=False)
    print("\nv = 1, modal vs power -- maxima:\n" + c.drop(columns=["v", "N", "a", "b", "t_star_modal",
                                                                    "t_star_power"]).max().to_string())
    # 2. each (v, N) against the run already in results/
    rows = []
    for (v, N), (dname, ftag) in EARLIER.items():
        p = os.path.join(RES, dname)
        old = (pd.read_csv(os.path.join(p, f"m_5t_star_96x96_{ftag}.csv"))[["a", "b", "t_star", "m_at_5_t_star"]]
               .merge(pd.read_csv(os.path.join(p, f"summary_96x96_{ftag}.csv"))[["a", "b", "lambda_1"]])
               .merge(pd.read_csv(os.path.join(p, f"m_t_tilde_96x96_{ftag}.csv"))
                      [["a", "b", "lambda_2", "lambda_2_imag", "t_tilde_5_over_abs", "m_at_t_tilde_5_over_abs"]]))
        new = d[(d.v == v) & (d.N == N)].merge(old, on=["a", "b"], suffixes=("", "_old"))
        for _, q in new.iterrows():
            rows.append(dict(v=v, N=N, a=q.a, b=q.b, earlier_run=dname,
                             abs_diff_t_star=abs(q.t_star - q.t_star_old),
                             rel_diff_m_at_5_t_star=rel(q.m_at_5_t_star, q.m_at_5_t_star_old),
                             rel_diff_lambda_1=rel(q.lambda_1, q.lambda_1_old),
                             rel_diff_lambda_2=rel(q.lambda_2, q.lambda_2_old),
                             abs_diff_lambda_2_imag=abs(q.lambda_2_imag - q.lambda_2_imag_old),
                             abs_diff_five_t_hat=abs(q.five_t_hat - q.t_tilde_5_over_abs),
                             rel_diff_m_at_5_t_hat=rel(q.m_at_5_t_hat, q.m_at_t_tilde_5_over_abs)))
    # v = 1, N = 16 has no study of its own; its t* was checked live in the 11-12 validation
    vp = pd.read_csv(os.path.join(RES, "2026-11-12_ab-grid-96x96_v1000_N16_tau0.01", "validation",
                                  "v1_power_vs_modal.csv"))
    new = d[(d.v == 1.0) & (d.N == 16)].merge(vp[["a", "b", "t_star_modal"]], on=["a", "b"])
    for _, q in new.iterrows():
        rows.append(dict(v=1.0, N=16, a=q.a, b=q.b,
                         earlier_run="2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/validation (modal t*)",
                         abs_diff_t_star=abs(q.t_star - q.t_star_modal)))
    e = pd.DataFrame(rows)
    e.to_csv(os.path.join(OUT, "validation", "vs_earlier_runs.csv"), index=False)
    print("\nagainst the earlier runs -- maxima per (v, N):\n"
          + e.drop(columns=["a", "b", "earlier_run"]).groupby(["v", "N"]).max().to_string())


def main(workers=8):
    t0 = time.perf_counter()
    jobs = [(v, N, a, b) for v, N in CASES for a in AB for b in AB]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        d = pd.DataFrame(list(ex.map(analyse, jobs)))
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    for v, N in CASES:
        d[(d.v == v) & (d.N == N)][COLUMNS].to_csv(
            os.path.join(OUT, "data", f"summary_96x96_{tag(v, N)}.csv"), index=False)
    d[COLUMNS].to_csv(os.path.join(OUT, "data", "summary_96x96_all.csv"), index=False)
    print(d[COLUMNS].to_string(index=False))
    validate(d)
    print(f"\ntotal {time.perf_counter() - t0:.1f} s; wrote {OUT}")


if __name__ == "__main__":
    main()
