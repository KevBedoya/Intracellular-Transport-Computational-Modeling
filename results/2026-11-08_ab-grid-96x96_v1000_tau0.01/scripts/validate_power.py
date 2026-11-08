"""Two checks of the power engine, written to ../validation/.

    June-2026/.venv/bin/python results/2026-11-08_ab-grid-96x96_v1000_tau0.01/scripts/validate_power.py modal
    June-2026/.venv/bin/python results/2026-11-08_ab-grid-96x96_v1000_tau0.01/scripts/validate_power.py stepper

modal    -- the whole pipeline at v = 1, where the modal engine is well conditioned
            (cond(V) ~ 1e8), against the 2026-11-05 modal study: every per-pair
            number, and the reference y'' against the exact modal y''.
            -> validation/v1_power_vs_modal_11-05.csv
stepper  -- the layer masses at v = 1000 against the time-stepper itself
            (launch._solve_mass_analysis, CPU), over a short window T = 0.05 with
            one sample every 4305 steps. The time-stepper stores in slot j >= 1 the
            masses of x_{j*4305 - 1}, so the power engine is evaluated at those steps.
            -> validation/v1000_power_vs_time_stepper.csv
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_96_power as rp               # sets the src path

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(HERE), "validation")
REF = os.path.join(rp.R, "results", "2026-11-05_ab-grid-96x96_modal_tau0.01")


def rel(x, y):
    return np.abs(np.asarray(x, float) / np.asarray(y, float) - 1)


def modal_y2_gap(args):
    """max |reference y'' - exact modal y''| for t > 0.05, at v = 1."""
    a, b = args
    warnings.filterwarnings("ignore")
    from computational_tools import mass_analysis_modal as mam
    z = np.load(os.path.join(rp.cache_dir(1.0), f"a{a:g}_b{b:g}.npz"))
    late = z["t_ref"] > 0.05
    modes = mam.solve_modes(rp.G, rp.G, a, b, 1.0, rp.NL)
    _, _, exact, noise = mam.ln_mass_derivatives(modes, z["t_ref"][late])
    return float(np.max(np.abs(z["y2_ref"][late] - exact))), float(np.max(noise))


def check_modal():
    p = rp.run(1.0, tempfile.mkdtemp()).set_index(["a", "b"]).sort_index()
    s = pd.read_csv(os.path.join(REF, "summary_96x96_modal.csv")).set_index(["a", "b"]).sort_index()
    m5 = pd.read_csv(os.path.join(REF, "m_5t_star_96x96_modal.csv")).set_index(["a", "b"]).sort_index()
    mt = pd.read_csv(os.path.join(REF, "m_t_tilde_96x96_modal.csv")).set_index(["a", "b"]).sort_index()
    d = pd.DataFrame(index=p.index)
    d["t_star_power"], d["t_star_modal"] = p.t_star, s.t_star
    d["abs_diff_t_star"] = np.abs(p.t_star - s.t_star)
    d["rel_diff_m_star"] = rel(p.m_star, s.m_star)
    d["rel_diff_m_at_t_star_recomputed"] = rel(p.m_at_t_star_recomputed, s.m_star)
    d["rel_diff_m_at_5_t_star"] = rel(p.m_at_5_t_star, m5.m_at_5_t_star)
    d["rel_diff_m_at_10_t_star"] = rel(p.m_at_10_t_star, s.m_at_10_t_star)
    for key, _, _ in rp.T_TILDE:
        d[f"rel_diff_m_at_t_tilde_{key}"] = rel(p[f"m_at_t_tilde_{key}"], mt[f"m_at_t_tilde_{key}"])
    for c in ("lambda_1", "lambda_2"):
        d[f"rel_diff_{c}"] = rel(p[c], s[c])
    d["abs_diff_lambda_2_imag"] = np.abs(p.lambda_2_imag - s.lambda_2_imag)
    d["central_difference_t_star_power"] = p.central_difference_t_star
    d["central_difference_t_star_modal"] = s.central_difference_t_star
    d["rel_diff_fit_slope"] = rel(p.fit_slope, s.fit_slope)
    d["rel_diff_psd_omega_peak"] = np.where(s.psd_omega_peak == 0, np.abs(p.psd_omega_peak),
                                            rel(p.psd_omega_peak, s.psd_omega_peak))
    d["rel_diff_psd_fwhm"] = rel(p.psd_fwhm, s.psd_fwhm)
    with ProcessPoolExecutor(max_workers=8) as ex:
        gaps = list(ex.map(modal_y2_gap, list(d.index)))
    d["max_abs_reference_minus_exact_y2_after_0p05"] = [g for g, _ in gaps]
    d["modal_y2_noise_bound_after_0p05"] = [n for _, n in gaps]
    os.makedirs(OUT, exist_ok=True)
    d.reset_index().to_csv(os.path.join(OUT, "v1_power_vs_modal_11-05.csv"), index=False)
    print(d.reset_index().to_string(index=False))
    print("\nmaxima:\n" + d.drop(columns=[c for c in d if c.startswith(("t_star_", "central"))]).max().to_string())


def stepper_pair(args):
    a, b, v, T, s = args
    warnings.filterwarnings("ignore")
    import time
    import power_engine as pe
    from launch_functions import launch
    K = launch.num.compute_K(rp.G, rp.G, T, 1.0, 1.0)
    rk = int(np.floor(K / s))
    ser = [np.zeros(rk) for _ in range(5)]
    D_LAYER, A_LAYER = launch.sup.initialize_layers(rp.G, rp.G)
    t0 = time.perf_counter()
    launch._solve_mass_analysis(launch.DEVICE_CPU, ser, rp.G, rp.G, a, b, v, T, list(rp.NL),
                                D_LAYER, A_LAYER, s, rk, 0.0, 1.0, 1.0, 10 ** 6, True, 0, 0)
    wall = time.perf_counter() - t0
    e = pe.PowerEngine(rp.G, a, b, v, rp.NL)
    DA = e.layer_masses(np.arange(1, rk) * s - 1)
    dl, al = ser[0][1:], ser[1][1:]
    return dict(a=a, b=b, v=v, T=T, samples=rk, steps=int(K), stepper_wall_s=wall,
                M_DL_at_T=dl[-1], M_AL_at_T=al[-1],
                max_rel_diff_M_DL=float(np.max(rel(DA[:, 0], dl))),
                max_rel_diff_M_AL=float(np.max(rel(DA[:, 1], al))),
                max_rel_diff_M_total=float(np.max(rel(DA.sum(1), dl + al))))


def check_stepper(T=0.05, pairs=((0.1, 100.0), (100.0, 0.1), (10.0, 10.0))):
    with ProcessPoolExecutor(max_workers=len(pairs)) as ex:
        d = pd.DataFrame(list(ex.map(stepper_pair, [(a, b, 1000.0, T, rp.S) for a, b in pairs])))
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "v1000_power_vs_time_stepper.csv"), index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    {"modal": check_modal, "stepper": check_stepper}[sys.argv[1]]()
