"""Two checks of the power engine at N = 16, written to ../validation/.

    June-2026/.venv/bin/python results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/scripts/validate_power.py modal
    June-2026/.venv/bin/python results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/scripts/validate_power.py stepper

modal    -- the whole pipeline at v = 1, against the modal engine run live on the
            same N_LIST (there is no 11-05 study at this N): t*, m*, fit_slope and
            lambda_1, lambda_2 from launch._char_time_onset_modal, the modal M at
            5 t*, 10 t* and the three t~, and the reference y'' against the exact
            modal y''. cond(V) is recorded at v = 1 and at v = 1000.
            -> validation/v1_power_vs_modal.csv
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


def rel(x, y):
    return np.abs(np.asarray(x, float) / np.asarray(y, float) - 1)


def modal_pair(args):
    """The modal engine's numbers for one pair at v = 1, and its y'' against the reference."""
    a, b, times = args
    warnings.filterwarnings("ignore")
    from computational_tools import mass_analysis_modal as mam
    from launch_functions import launch
    modes = mam.solve_modes(rp.G, rp.G, a, b, 1.0, rp.NL)
    row = launch._char_time_onset_modal(modes, rp.T, tau=rp.TAU)
    ts = row["t_star"]
    m = np.exp(mam.ln_mass(modes, [ts, 5 * ts, 10 * ts] + list(times)))
    z = np.load(os.path.join(rp.cache_dir(1.0), f"a{a:g}_b{b:g}.npz"))
    late = z["t_ref"] > 0.05
    _, _, exact, noise = mam.ln_mass_derivatives(modes, z["t_ref"][late])
    cond_1000 = mam.solve_modes(rp.G, rp.G, a, b, 1000.0, rp.NL)["cond_V"]
    return dict(t_star=ts, m_star=row["m_star"], fit_slope=row["fit_slope"],
                lambda_1=row["lambda_1"], lambda_2=row["lambda_2"], lambda_2_imag=row["lambda_2_imag"],
                m_at_t_star=m[0], m_at_5_t_star=m[1], m_at_10_t_star=m[2],
                m_at_t_tilde=list(m[3:]),
                y2_gap=float(np.max(np.abs(z["y2_ref"][late] - exact))), y2_noise=float(np.max(noise)),
                cond_V_v1=modes["cond_V"], cond_V_v1000=cond_1000)


def check_modal():
    p = rp.run(1.0, tempfile.mkdtemp()).set_index(["a", "b"]).sort_index()
    keys = [key for key, _, _ in rp.T_TILDE]
    args = [(a, b, [p.loc[(a, b), f"t_tilde_{k}"] for k in keys]) for a, b in p.index]
    with ProcessPoolExecutor(max_workers=8) as ex:
        mo = list(ex.map(modal_pair, args))
    d = pd.DataFrame(index=p.index)
    d["t_star_power"] = p.t_star
    d["t_star_modal"] = [r["t_star"] for r in mo]
    d["abs_diff_t_star"] = np.abs(d.t_star_power - d.t_star_modal)
    d["rel_diff_m_star"] = rel(p.m_star, [r["m_star"] for r in mo])
    d["rel_diff_m_at_t_star_recomputed"] = rel(p.m_at_t_star_recomputed, [r["m_at_t_star"] for r in mo])
    d["rel_diff_m_at_5_t_star"] = rel(p.m_at_5_t_star, [r["m_at_5_t_star"] for r in mo])
    d["rel_diff_m_at_10_t_star"] = rel(p.m_at_10_t_star, [r["m_at_10_t_star"] for r in mo])
    for i, k in enumerate(keys):
        d[f"rel_diff_m_at_t_tilde_{k}"] = rel(p[f"m_at_t_tilde_{k}"], [r["m_at_t_tilde"][i] for r in mo])
    for c in ("lambda_1", "lambda_2"):
        d[f"rel_diff_{c}"] = rel(p[c], [r[c] for r in mo])
    d["abs_diff_lambda_2_imag"] = np.abs(p.lambda_2_imag - [r["lambda_2_imag"] for r in mo])
    d["rel_diff_fit_slope"] = rel(p.fit_slope, [r["fit_slope"] for r in mo])
    d["max_abs_reference_minus_exact_y2_after_0p05"] = [r["y2_gap"] for r in mo]
    d["modal_y2_noise_bound_after_0p05"] = [r["y2_noise"] for r in mo]
    d["cond_V_v1"] = [r["cond_V_v1"] for r in mo]
    d["cond_V_v1000"] = [r["cond_V_v1000"] for r in mo]
    os.makedirs(OUT, exist_ok=True)
    d.reset_index().to_csv(os.path.join(OUT, "v1_power_vs_modal.csv"), index=False)
    print(d.reset_index().to_string(index=False))
    print("\nmaxima:\n" + d.drop(columns=["t_star_power", "t_star_modal"]).max().to_string())
    print("minima of cond_V:\n" + d[["cond_V_v1", "cond_V_v1000"]].min().to_string())


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
    return dict(a=a, b=b, v=v, N=rp.N, T=T, samples=rk, steps=int(K), stepper_wall_s=wall,
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
