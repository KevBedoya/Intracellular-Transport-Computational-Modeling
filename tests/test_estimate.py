"""Check the API's cost model against the solver and against reality.

server/estimate.py reimplements the timestep count so the API process does not
have to import numba to answer an estimate request. That duplication is only
safe if it is asserted, so the first test compares the two directly.

The rest measure the model against jobs that actually ran, which is the only
thing that makes an estimate worth showing.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "server"),
           os.path.join(_ROOT, "src", "intracellular_transport"),
           os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate  # noqa: E402

CT = "Characteristic Time (mass vs v)"


def test_steps_matches_the_solver():
    """estimate.steps must equal numerical_tools.compute_K exactly."""
    from computational_tools import numerical_tools as num
    cases = [(48, 48, 1.0), (96, 96, 1.0), (112, 112, 1.0), (160, 160, 1.0),
             (176, 176, 1.0), (192, 192, 1.0), (224, 224, 1.0),
             (64, 96, 0.5), (96, 64, 2.0), (32, 32, 0.8)]
    bad = []
    for rg, ry, T in cases:
        mine = estimate.steps(rg, ry, T)
        theirs = int(num.compute_K(rg, ry, T))
        if mine != theirs:
            bad.append(f"{rg}x{ry} T={T}: estimate={mine:,} solver={theirs:,}")
    assert not bad, "estimate.steps disagrees with compute_K:\n  " + "\n  ".join(bad)


def _hours(rg, ry, device, T=1.0, v_list=(10000,)):
    p = {"rg_param": rg, "ry_param": ry, "T_param": T,
         "v_LIST": list(v_list), "device": device}
    return estimate.seconds(CT, p) / 3600.0


def test_gpu_estimate_matches_measured_runs():
    """The 176x176 GPU job is the one real end-to-end GPU data point."""
    # It ran 22.5 hr while sharing the card with two other GPU jobs; solo it is
    # the ~6.6 hr this model should produce.
    h = _hours(176, 176, "gpu")
    assert 5.5 <= h <= 8.0, f"176^2 GPU estimate {h:.2f} hr outside 5.5-8.0"


def test_cpu_estimate_matches_measured_runs():
    """Completed CPU jobs, with their actual wall times.

    48^2 and 64^2 are here deliberately. An affine-in-patches fit reproduced the
    large grids well and underestimated 48^2 fivefold, which no large-grid case
    would have caught -- and small grids are what people run while exploring.
    """
    measured = {48: 0.0639, 64: 0.3222, 80: 1.2292, 96: 3.68, 112: 9.32,
                128: 21.31, 144: 43.68, 160: 82.36}
    bad = []
    for g, actual in sorted(measured.items()):
        h = _hours(g, g, "cpu")
        ratio = h / actual
        if not (0.55 <= ratio <= 1.8):
            bad.append(f"{g}^2: estimate {h:.3f} hr vs actual {actual:.3f} hr "
                       f"(ratio {ratio:.2f})")
    assert not bad, "CPU estimate off:\n  " + "\n  ".join(bad)


def test_gpu_is_faster_and_the_gap_widens():
    """The whole point of the change: device must alter the answer."""
    prev = 0.0
    for g in (96, 128, 160, 192):
        ratio = _hours(g, g, "cpu") / _hours(g, g, "gpu")
        assert ratio > 1.0, f"{g}^2: GPU not faster ({ratio:.2f})"
        assert ratio > prev, f"{g}^2: speedup {ratio:.2f} did not grow past {prev:.2f}"
        prev = ratio


def test_auto_is_costed_as_gpu():
    assert _hours(160, 160, "auto") == _hours(160, 160, "gpu")


def test_velocity_sweep_scales():
    """Three velocities cost three solves -- but only one numba compile.

    So the ratio is 3x on the solve, slightly under 3x on the total. Comparing
    totals directly would bake in the startup constant and break whenever it is
    retuned, so the fixed cost is removed from both sides first.
    """
    base = {"rg_param": 96, "ry_param": 96, "T_param": 1, "device": "cpu"}
    one = estimate.seconds(CT, dict(base, v_LIST=[1e4]))
    three = estimate.seconds(CT, dict(base, v_LIST=[1e3, 1e4, 1e5]))
    s1 = one - estimate.STARTUP_SECONDS
    s3 = three - estimate.STARTUP_SECONDS
    assert abs(s3 / s1 - 3.0) < 1e-9, f"expected 3x on the solve, got {s3/s1:.4f}"
    assert three < 3 * one, "startup should be paid once, not per velocity"


def test_unestimable_returns_none():
    assert estimate.seconds(CT, {"T_param": 1}) is None          # no grid
    assert estimate.seconds(CT, {"rg_param": 1, "ry_param": 1}) is None
    assert estimate.seconds("Analytic Benchmark",
                            {"rg_param": 96, "ry_param": 96}) is None


def test_share_factor():
    assert estimate.share_factor("cpu", 3) == 1.0
    assert estimate.share_factor("gpu", 1) == 1.0
    assert estimate.share_factor("gpu", 3) == 3.0
    assert estimate.share_factor("auto", 2) == 2.0
    assert estimate.share_factor("gpu", 0) == 1.0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns)-failed} passed, {failed} failed")

    print("\nModel output (hours), for the record:")
    print(f"  {'grid':>6} {'CPU':>9} {'GPU':>9} {'speedup':>8}")
    for g in (48, 96, 128, 160, 176, 192, 208, 224):
        c, gp = _hours(g, g, "cpu"), _hours(g, g, "gpu")
        print(f"  {g:>4}^2 {c:>8.2f}h {gp:>8.2f}h {c/gp:>7.1f}x")
    sys.exit(1 if failed else 0)
