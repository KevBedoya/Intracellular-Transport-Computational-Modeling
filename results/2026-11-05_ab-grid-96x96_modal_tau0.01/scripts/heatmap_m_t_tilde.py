"""M(t~) over the (a, b) grid as heatmaps, for t~ = 5/Re(lambda_2), 5/|lambda_2| and 1/|lambda_2|.

    June-2026/.venv/bin/python results/2026-11-05_ab-grid-96x96_modal_tau0.01/scripts/heatmap_m_t_tilde.py [--recompute]

lambda_2 is the char-time job's: the slowest excited decay rate after lambda_1
(launch._slowest_rates), real part `lambda_2` and |imaginary part| `lambda_2_imag`
in summary_96x96_modal.csv, so |lambda_2| = sqrt(lambda_2^2 + lambda_2_imag^2).
For b = 100 lambda_2 is real and 5/Re(lambda_2) = 5/|lambda_2|.

M(t~) comes from the spectral (eigen-modal) expansion of the 96x96 one-step operator
(mass_analysis_modal.solve_modes + ln_mass), re-solved here per pair. lambda_2 is
recomputed from the same solve as a check against the summary.

The solves are cached in m_t_tilde_96x96_modal.csv; --recompute redoes them.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heatmap_m_5t_star as h           # sets one BLAS thread per worker and the src path

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# (key, K, which part of lambda_2, TeX): t~ = K / Re(lambda_2) or K / |lambda_2|
VARIANTS = [("5_over_re", 5.0, "re", r"5/\mathrm{Re}(\lambda_2)"),
            ("5_over_abs", 5.0, "abs", r"5/|\lambda_2|"),
            ("1_over_abs", 1.0, "abs", r"1/|\lambda_2|")]
CSV = os.path.join(h.OUT, "m_t_tilde_96x96_modal.csv")


def solve(args):
    a, b, times = args
    import warnings
    warnings.filterwarnings("ignore")
    from computational_tools import mass_analysis_modal as mam
    from launch_functions import launch
    modes = mam.solve_modes(h.G, h.G, a, b, h.V, h.NL)
    _, l2_re, l2_im = launch._slowest_rates(modes)
    M = np.exp(mam.ln_mass(modes, times))
    return dict(a=a, b=b, lambda_2_local=l2_re, lambda_2_imag_local=l2_im,
                **{f"m_at_t_tilde_{key}": m for (key, *_), m in zip(VARIANTS, M)})


def compute():
    s = pd.read_csv(os.path.join(h.OUT, "summary_96x96_modal.csv"))
    d = s[["a", "b", "lambda_2", "lambda_2_imag"]].copy()
    d["lambda_2_abs"] = np.hypot(d.lambda_2, d.lambda_2_imag)
    for key, K, part, _ in VARIANTS:
        d[f"t_tilde_{key}"] = K / (d.lambda_2 if part == "re" else d.lambda_2_abs)
    todo = [(r.a, r.b, [r[f"t_tilde_{key}"] for key, *_ in VARIANTS]) for _, r in d.iterrows()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        m = pd.DataFrame(list(ex.map(solve, todo)))
    d = d.merge(m, on=["a", "b"])
    d["rel_diff_lambda_2"] = np.abs(d.lambda_2_local / d.lambda_2 - 1)
    d["abs_diff_lambda_2_imag"] = np.abs(d.lambda_2_imag_local - d.lambda_2_imag)
    d = d[["a", "b", "lambda_2", "lambda_2_imag", "lambda_2_abs"]
          + [c for key, *_ in VARIANTS for c in (f"t_tilde_{key}", f"m_at_t_tilde_{key}")]
          + ["lambda_2_local", "lambda_2_imag_local", "rel_diff_lambda_2", "abs_diff_lambda_2_imag"]]
    d.to_csv(CSV, index=False)
    return d


def plot(d):
    src = r"$\lambda_2$ from the char-time job" + "\n" + r"$M$ from the spectral (eigen-modal) series"
    for key, _, _, tex in VARIANTS:
        # 1/|lambda_2| keeps M in 0.74 - 1.00: colour by M itself, not log10 M
        h.heatmap(d, f"m_at_t_tilde_{key}", f"t_tilde_{key}", r"\tilde{t}", src,
                  f"heatmap_m_t_tilde_{key}_lambda2_96x96_modal",
                  t_def=rf"$\tilde{{t}} = {tex}$", log=(key != "1_over_abs"))


if __name__ == "__main__":
    d = compute() if ("--recompute" in sys.argv or not os.path.exists(CSV)) else pd.read_csv(CSV)
    print(d.to_string(index=False))
    plot(d)
