"""The steady-decay criterion that defines t*, checked against analytic curves.

``launch._char_time_onset`` is pure numpy -- a mass timeseries in, t* and the
fit over the steady segment out -- so it is tested here on curves whose answer
is known in closed form, with no solver involved. The solver-backed properties
(both kernels agree, workers do not change the numbers, rows are labelled
correctly) are in tests/test_ab_grid_char_time.py.

The reference curve is a two-mode decay,

    M(t) = (1 - c) exp(-l1 t) + c exp(-l2 t),     l2 > l1,

which is what a mass timeseries looks like once the higher modes have died:
the slow mode is the asymptotic straight line in ln M, and the fast one is the
transient the criterion has to wait out. Its normalised curvature
kappa = |y''| / y'^2 of y = ln M has a closed form, so the exact t* -- the last
time kappa exceeds tau -- is found by bisection and compared.

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

# Rates chosen to resemble the solver's: the 16x16 test grids decay at 2.5-6
# per unit time asymptotically, and their curvature falls about a decade per
# 0.05, i.e. a spectral gap near 45.
L1, L2, C = 4.0, 50.0, 0.3
DT = 1e-5
T_END = 1.0


def _two_mode(t, l1=L1, l2=L2, c=C):
    return (1 - c) * np.exp(-l1 * t) + c * np.exp(-l2 * t)


def _exact_kappa(t, l1=L1, l2=L2, c=C):
    e1, e2 = (1 - c) * np.exp(-l1 * t), c * np.exp(-l2 * t)
    m = e1 + e2
    dm = -l1 * e1 - l2 * e2
    d2m = l1 ** 2 * e1 + l2 ** 2 * e2
    dy = dm / m
    d2y = d2m / m - dy ** 2
    return abs(d2y) / dy ** 2


def _exact_onset(tau, lo=0.0, hi=T_END, **kw):
    """Where kappa falls through tau. kappa is monotone decreasing here."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _exact_kappa(mid, **kw) > tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _series(dt=DT, t_end=T_END, **kw):
    t = np.arange(int(round(t_end / dt)) + 1) * dt
    return _two_mode(t, **kw)


def test_onset_matches_the_analytic_crossing():
    """t* is where kappa last crosses tau, to well inside the spacing."""
    for tau in (1e-2, 1e-3, 1e-4):
        got = launch._char_time_onset(_series(), DT, tau=tau)["t_star"]
        want = _exact_onset(tau)
        # The decimated spacing is 1e-3; interpolating the crossing puts t*
        # far inside it.
        assert abs(got - want) < 1e-4, f"tau={tau}: t*={got}, exact {want}"


def test_m_star_is_the_mass_at_t_star():
    row = launch._char_time_onset(_series(), DT, tau=1e-3)
    want = _two_mode(row["t_star"])
    assert math.isclose(row["m_star"], want, rel_tol=1e-8), \
        f"m*={row['m_star']}, M(t*)={want}"
    assert row["total_mass_at_t1"] == row["m_star"]
    assert row["fit_window_t1"] == row["t_star"]


def test_fit_over_the_steady_segment_recovers_the_slow_rate():
    """The slope is the asymptotic decay rate of log10 M, i.e. -l1 / ln 10."""
    row = launch._char_time_onset(_series(), DT, tau=1e-4)
    want = -L1 / math.log(10)
    assert math.isclose(row["fit_slope"], want, rel_tol=1e-3), \
        f"slope {row['fit_slope']}, expected {want}"
    assert math.isclose(row["fit_window_t2"], T_END, abs_tol=2e-3)


def test_pure_exponential_is_straight_from_the_start():
    t = np.arange(100001) * DT
    row = launch._char_time_onset(np.exp(-3.0 * t), DT, tau=1e-3)
    # Straight everywhere, so t* is the first sample the criterion judges.
    assert row["t_star"] < 5e-3, row
    assert math.isclose(row["fit_slope"], -3.0 / math.log(10), rel_tol=1e-9)


def test_smaller_tau_is_stricter():
    series = _series()
    onsets = [launch._char_time_onset(series, DT, tau=tau)["t_star"]
              for tau in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert all(a < b for a, b in zip(onsets, onsets[1:])), onsets


def test_independent_of_sample_spacing_and_of_scale():
    """t* is a property of the curve, not of how finely it was sampled.

    Sample spacing is what changes with the grid resolution, so this is the
    property that keeps t* comparable across grids. Scaling M is a shift in
    ln M and must not move t* beyond roundoff in the differences.
    """
    tau = 1e-3
    base = launch._char_time_onset(_series(dt=DT), DT, tau=tau)["t_star"]
    for dt in (2.5e-6, 4e-5, 1.5e-4):
        other = launch._char_time_onset(_series(dt=dt), dt, tau=tau)["t_star"]
        assert abs(other - base) < 2e-4, f"dt={dt}: {other} vs {base}"
    scaled = launch._char_time_onset(7.0 * _series(), DT, tau=tau)["t_star"]
    assert abs(scaled - base) < 1e-9


def test_log_base_and_rate_do_not_change_what_tau_means():
    """Doubling both rates halves t*: kappa is dimensionless in the rates.

    A raw threshold on |y''| would not scale like this; it is why the criterion
    normalises by y'^2.
    """
    tau = 1e-3
    slow = _exact_onset(tau)
    fast = launch._char_time_onset(
        _series(l1=2 * L1, l2=2 * L2), DT, tau=tau)["t_star"]
    assert abs(fast - slow / 2) < 1e-4, f"{fast} vs {slow / 2}"


def test_no_onset_within_the_record_is_nan():
    """A record that ends before the decay settles is degenerate, not wrong."""
    short = _series(t_end=_exact_onset(1e-4) + 0.5 * launch.CHAR_TIME_MIN_HOLD)
    row = launch._char_time_onset(short, DT, tau=1e-4)
    assert all(math.isnan(v) for v in row.values()), row


def test_bad_samples_end_the_usable_record():
    """Non-positive or non-finite mass truncates the series instead of crashing.

    Beyond the truncation point nothing is judged, so the result equals that
    of the clean prefix.
    """
    series = _series()
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
        (dict(T_param=0.05), "too short"),
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
