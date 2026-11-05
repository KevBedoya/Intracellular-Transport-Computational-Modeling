"""The modal-expansion mass analysis must reproduce the time-stepper.

computational_tools.mass_analysis_modal diagonalises the solver's own one-step
operator, so the two differ only by round-off. These tests run both on small
grids -- centred and off-centre seeds, symmetric and asymmetric tube sets, and
a nonzero d_tube -- and require every one of the five mass series to agree to
1e-11 absolute, along with an exact (round-off) symmetry reduction.

Run directly (``python tests/test_mass_analysis_modal.py``) or under pytest.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
_PKG = os.path.join(_SRC, "intracellular_transport")
for _p in (_PKG, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep the router test's CSVs and figures out of the repository's data_output.
os.environ.setdefault("ITCM_OUTPUT_ROOT",
                      os.path.join(tempfile.gettempdir(), "itcm_mass_analysis_modal_tests"))

from computational_tools import analysis_tools as ant        # noqa: E402
from computational_tools import mass_analysis_modal as mam   # noqa: E402
from computational_tools import numerical_tools as num       # noqa: E402
from computational_tools import supplements as sup           # noqa: E402

MA_FACTOR = 5
TOL = 1e-11
SERIES = ("DL", "AL", "ALoI", "ALoT", "TM")


def _both(G, a, b, v, T, N_LIST, **kw):
    K = num.compute_K(G, G, T)
    relative_k = K // MA_FACTOR
    stepped = [np.zeros(relative_k) for _ in SERIES]
    modal = [np.zeros(relative_k) for _ in SERIES]
    D_LAYER, A_LAYER = sup.initialize_layers(G, G)
    ant.comp_mass_analysis_respect_to_time(G, G, a, b, v, T, np.array(N_LIST), D_LAYER, A_LAYER,
                                           *stepped, MA_FACTOR, relative_k, **kw)
    info = mam.comp_mass_analysis_modal_expansion(G, G, a, b, v, T, np.array(N_LIST), *modal,
                                                  MA_FACTOR, relative_k, **kw)
    return stepped, modal, info


def _check(G, a, b, v, T, N_LIST, **kw):
    stepped, modal, info = _both(G, a, b, v, T, N_LIST, **kw)
    assert info["invariance_residual"] < mam.INVARIANCE_TOL
    for name, x, y in zip(SERIES, stepped, modal):
        err = np.max(np.abs(x - y))
        assert err < TOL, f"{name}: max |stepped - modal| = {err:.2e}"
    return info


def test_centred_symmetric_tubes():
    info = _check(16, 1.0, 1.0, 1.0, 0.5, [0, 4, 8, 12])
    assert info["n_reduced"] < info["n_state"] // 4


def test_unequal_switch_rates():
    _check(16, 10.0, 0.1, 2.0, 0.5, [0, 4, 8, 12])


def test_asymmetric_tubes():
    _check(12, 1.0, 10.0, 1.0, 0.5, [0, 5])


def test_off_centre_seed():
    _check(16, 1.0, 1.0, 1.0, 0.5, [0, 4, 8, 12], center_init_cond=False, m_init=3, n_init=2)


def test_extraction_region():
    _check(16, 1.0, 1.0, 1.0, 0.5, [0, 8], d_tube=0.05)


def test_without_symmetry_matches_with():
    G, T, N_LIST = 12, 0.5, [0, 3, 6, 9]
    relative_k = num.compute_K(G, G, T) // MA_FACTOR
    out = []
    for use_symmetry in (True, False):
        s = [np.zeros(relative_k) for _ in SERIES]
        mam.comp_mass_analysis_modal_expansion(G, G, 1.0, 1.0, 1.0, T, np.array(N_LIST), *s,
                                               MA_FACTOR, relative_k, use_symmetry=use_symmetry)
        out.append(s[-1])
    assert np.max(np.abs(out[0] - out[1])) < TOL


def test_mass_analysis_engine_through_the_router():
    """The "Mass Analysis" computation, as the API worker and desktop app run it.

    engine="modal" must pass validation (any case), write the same CSVs as the
    default time-stepper to round-off, and say which engine produced them; an
    unknown engine is refused at submission.
    """
    from multiprocessing_tools import computation_router as router

    name = "Mass Analysis"
    base = dict(rg_param=12, ry_param=12, v_param=1.0, a_param=10.0, b_param=0.1,
                T_param=0.5, N_LIST=[0, 3, 6, 9], save_png=False)
    assert router.validate_params(name, base) == {}
    assert router.validate_params(name, {**base, "engine": "Modal"}) == {}
    assert set(router.validate_params(name, {**base, "engine": "eigen"})) == {"engine"}

    def total_mass(params):
        result = router.run_selected_computation(name, dict(params))
        total = [d for d in result["output_dirs"] if os.path.normpath(d).split(os.sep)[-2] == "total"]
        csv = [f for f in os.listdir(total[0]) if f.endswith(".csv")]
        return result, pd.read_csv(os.path.join(total[0], csv[0]))

    res_s, stepped = total_mass({**base, "compute_psd": True})
    res_m, modal = total_mass({**base, "engine": "modal", "device": "gpu", "compute_psd": True})
    assert (res_s["engine"], res_m["engine"]) == ("time_stepper", "modal")
    assert np.array_equal(stepped["T"].to_numpy(), modal["T"].to_numpy())
    assert np.max(np.abs(stepped["M"].to_numpy() - modal["M"].to_numpy())) < TOL

    # the PSD of y'' is read off the same samples, so the engines agree on it too
    for key in ("psd_omega_peak", "psd_peak", "psd_fwhm"):
        assert res_s["psd_y2"][key] is not None
        assert abs(res_s["psd_y2"][key] - res_m["psd_y2"][key]) <= 1e-6 * abs(res_s["psd_y2"][key])
    assert "psd_y2" not in router.run_selected_computation(name, dict(base))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
