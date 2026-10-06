"""The right-sweep |y''| criterion that defines t*, checked against analytic curves.

``launch._char_time_onset`` is pure numpy -- a mass timeseries in, t* and the
fit over the steady segment out -- so it is tested here on curves whose answer
is known in closed form, with no solver involved. The solver-backed properties
(both kernels agree, workers do not change the numbers, rows are labelled
correctly) are in tests/test_ab_grid_char_time.py.

The rule: with y = ln M, mark l(t) = 1 where |y''(t)| <= tau. Sweeping from the
right, t* is the sample just after the last one with l = 0, so |y''| <= tau
from t* to the end. The exact answer for a closed-form y is therefore the last
time |y''| exceeds tau, found here on a dense grid and compared with the
sampled t*, which is resolved to one sample.

Two reference curves:

  * a two-mode decay M = (1 - c) e^{-l1 t} + c e^{-l2 t}, what a mass timeseries
    looks like once the higher modes have died, for which |y''| decreases
    monotonically;
  * a damped oscillation in the decay rate, y = -3t + A e^{-g t} sin(w t), the
    shape seen for slow switching onto the advective layer, where y'' passes
    through zero repeatedly before it settles -- the case the sweep from the
    right exists for.

Run directly (``python tests/test_char_time_criterion.py``) or under pytest.
"""
import math
import os
import sys
import tempfile

os.environ.setdefault("ITCM_OUTPUT_ROOT",
                      os.path.join(tempfile.gettempdir(), "itcm_ab_grid_tests"))
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src", "intracellular_transport"),
           os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                             # noqa: E402

from launch_functions import launch                            # noqa: E402

L1, L2, C = 4.0, 50.0, 0.3
DT = 1e-5
T_END = 1.0
# Damped oscillation in the decay rate.
OA, OG, OW = 0.5, 8.0, 10.0


def _grid(dt=DT, t_end=T_END):
    return np.arange(int(round(t_end / dt)) + 1) * dt


def _two_mode(t, l1=L1, l2=L2, c=C):
    return (1 - c) * np.exp(-l1 * t) + c * np.exp(-l2 * t)


def _two_mode_d2(t, l1=L1, l2=L2, c=C):
    e1, e2 = (1 - c) * np.exp(-l1 * t), c * np.exp(-l2 * t)
    m = e1 + e2
    dm = -l1 * e1 - l2 * e2
    d2m = l1 ** 2 * e1 + l2 ** 2 * e2
    return d2m / m - (dm / m) ** 2


def _osc_y(t):
    return -3.0 * t + OA * np.exp(-OG * t) * np.sin(OW * t)


def _osc_d2(t):
    e = np.exp(-OG * t)
    return OA * e * ((OG ** 2 - OW ** 2) * np.sin(OW * t)
                     - 2 * OG * OW * np.cos(OW * t))


def _exact_last_exceedance(d2, tau, t_end=T_END):
    """Last time |y''| > tau, on a grid far finer than any sampling tested."""
    t = np.linspace(0, t_end, 2_000_001)
    over = np.nonzero(np.abs(d2(t)) > tau)[0]
    return t[over[-1]] if over.size else 0.0


def test_onset_is_the_last_exceedance_two_mode():
    t = _grid()
    for tau in (1e-1, 1e-2, 1e-3):
        got = launch._char_time_onset(_two_mode(t), DT, tau=tau)["t_star"]
        want = _exact_last_exceedance(_two_mode_d2, tau)
        # Resolved to one sample, plus the O(H^2) error of the difference.
        assert want < got <= want + 2 * DT, f"tau={tau}: t*={got}, exact {want}"


def test_sweep_from_the_right_skips_an_early_dip():
    """y'' crosses zero many times; t* is after the last swing, not the first.

    A rule taking the first time |y''| <= tau would stop at the first zero
    crossing of the oscillation, near t = 0.18. The right sweep must not.
    """
    t = _grid()
    tau = 1e-1
    got = launch._char_time_onset(np.exp(_osc_y(t)), DT, tau=tau)["t_star"]
    want = _exact_last_exceedance(_osc_d2, tau)
    first_small = t[np.nonzero(np.abs(_osc_d2(t)) <= tau)[0][0]]
    assert first_small < 0.2 < want, (first_small, want)
    assert want < got <= want + 2 * DT, f"t*={got}, exact {want}"


def test_m_star_is_the_mass_at_t_star():
    t = _grid()
    row = launch._char_time_onset(_two_mode(t), DT, tau=1e-3)
    assert math.isclose(row["m_star"], _two_mode(row["t_star"]), rel_tol=1e-12)
    assert row["total_mass_at_t1"] == row["m_star"]
    assert row["fit_window_t1"] == row["t_star"]
    assert math.isclose(row["fit_window_t2"], T_END, abs_tol=DT)


def test_fit_over_the_steady_segment_recovers_the_slow_rate():
    """The slope is the asymptotic decay rate of log10 M, i.e. -l1 / ln 10."""
    t = _grid()
    row = launch._char_time_onset(_two_mode(t), DT, tau=1e-4)
    assert math.isclose(row["fit_slope"], -L1 / math.log(10), rel_tol=1e-3), row


def test_pure_exponential_is_straight_from_the_start():
    t = _grid()
    row = launch._char_time_onset(np.exp(-3.0 * t), DT, tau=1e-3)
    # Never above tau, so t* is the first judged sample, one step H in.
    assert row["t_star"] <= launch.CHAR_TIME_DIFF_SPACING + DT, row
    assert math.isclose(row["fit_slope"], -3.0 / math.log(10), rel_tol=1e-9)


def test_smaller_tau_is_stricter():
    series = _two_mode(_grid())
    onsets = [launch._char_time_onset(series, DT, tau=tau)["t_star"]
              for tau in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert all(a < b for a, b in zip(onsets, onsets[1:])), onsets


def test_independent_of_sample_spacing_and_of_scale():
    """t* is a property of the curve, not of how finely it was sampled.

    Sample spacing is what changes with the grid resolution, so this is the
    property that keeps t* comparable across grids. Scaling M shifts ln M and
    must not move t* by more than a sample.
    """
    tau = 1e-3
    want = _exact_last_exceedance(_two_mode_d2, tau)
    for dt in (2.5e-6, 4e-5, 1.5e-4):
        got = launch._char_time_onset(_two_mode(_grid(dt)), dt, tau=tau)["t_star"]
        assert want < got <= want + 2 * dt, f"dt={dt}: {got} vs exact {want}"
    t = _grid()
    base = launch._char_time_onset(_two_mode(t), DT, tau=tau)["t_star"]
    scaled = launch._char_time_onset(7.0 * _two_mode(t), DT, tau=tau)["t_star"]
    assert abs(scaled - base) <= DT


def test_still_above_tau_at_the_end_is_nan():
    """A run that ends before |y''| settles has no t*: degenerate, not wrong."""
    t_end = 0.8 * _exact_last_exceedance(_two_mode_d2, 1e-4)
    row = launch._char_time_onset(_two_mode(_grid(t_end=t_end)), DT, tau=1e-4)
    assert all(math.isnan(v) for v in row.values()), row


def test_bad_samples_end_the_usable_record():
    """Non-positive or non-finite mass truncates the series instead of crashing."""
    series = _two_mode(_grid())
    cut = int(0.8 / DT)
    dirty = series.copy()
    dirty[cut] = 0.0
    dirty[cut + 1:] = np.nan
    got = launch._char_time_onset(dirty, DT, tau=1e-3)
    want = launch._char_time_onset(series[:cut], DT, tau=1e-3)
    assert got == want, (got, want)


def test_preflight_refuses_what_cannot_succeed():
    cases = [
        (dict(tau=0.0), "tau must be"),
        (dict(tau=-1e-3), "tau must be"),
        (dict(tau=float("nan")), "tau must be"),
        (dict(relative_k=5), "too short"),
    ]
    for overrides, expected in cases:
        kwargs = dict(T_param=1.0, MA_collection_factor=5, relative_k=1000,
                      tau=1e-3)
        kwargs.update(overrides)
        try:
            launch._char_time_preflight(**kwargs)
        except ValueError as e:
            assert expected in str(e), f"{overrides}: wrong error {e}"
        else:
            raise AssertionError(f"{overrides} was accepted")
    launch._char_time_preflight(1.0, 5, 1000, 1e-3)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
