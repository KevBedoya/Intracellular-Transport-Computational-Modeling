"""Correctness of the (a, b) grid characteristic-time kernel.

The kernel's whole claim is that it is the existing characteristic-time solve
run once per switch-rate pair, several at a time. So there are only three things
that can go wrong, and each has a test here:

  * the *criterion* could differ from the velocity sweep's, which would make a t* from
    this kernel not comparable with one already published;
  * running the points concurrently could change them, which would mean the
    parallelism is not the transparent optimisation it is documented to be;
  * the grid could not cover the cartesian product it claims to, or could label
    a row with the wrong (a, b).

Grids are deliberately tiny -- 16x16, four microtubules -- because none of those
three properties depends on resolution, and at this size a full solve is a
fraction of a second. Runtime here is dominated by numba compiling the stencil,
once per process.

Run directly (``python tests/test_ab_grid_char_time.py``) or under pytest.
"""
import glob
import math
import os
import sys
import tempfile

# Must precede any package import: system_configuration.file_paths reads this at
# import time, and writing test artefacts into the real data_output tree would
# leave the recovery scripts (which glob it) reading them as results.
# setdefault, not assignment: a spawned worker re-imports this module and has to
# agree with its parent about where the tree is.
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
import pandas as pd                                            # noqa: E402

from gpu import persistent_kernel as pk                        # noqa: E402
from launch_functions import launch                            # noqa: E402
from system_configuration import file_paths as fp              # noqa: E402

RG = RY = 16
# Long enough for every point tested here (v = 100, switch rates >= 1) to
# settle below the default tau before the run ends. Chosen so the tests compare
# real numbers rather than agreeing on NaN.
T = 1.0
N_LIST = np.array([0, 4, 8, 12], dtype=np.int64)
V = 100.0

A_LIST = [1.0, 10.0]
B_LIST = [1.0, 10.0]

# Columns that must agree exactly between two CPU runs of the same point. Every
# one is a float the solve produced, not a parameter echoed back, so an exact
# comparison is the right one: the CPU path is deterministic.
RESULT_COLUMNS = ("t_star", "m_star", "fit_slope", "fit_intercept",
                  "fit_window_t1", "fit_window_t2",
                  "total_mass_at_t1", "total_mass_at_t2")

# CPU-vs-GPU tolerance. The two disagree because the mass and centre reductions
# are sequential on one and parallel tree reductions on the other, and
# floating-point addition is not associative -- see docs/GPU_PLAN.md. This is the
# same criterion tests/test_gpu_agreement.py applies to the underlying solve.
GPU_TOLERANCE = 1e-9

# Looser for the columns located by the steady-decay criterion (t* and the
# masses read at it). t* is a sample time -- the first sample after the last
# |y''| > tau -- and y'' is a second difference at H ~ 1e-4, which turns a
# ~1e-14 disagreement in M into ~1e-6 in y''. Should a sample sit that close to
# tau, the two devices put t* one sample apart: 1.5e-4 here, i.e. up to ~1.5e-3
# relative for t* >= 0.1, and a mass that differs by lambda * 1.5e-4 ~ 1e-3.
# Usually they agree exactly; this bounds the case where they do not.
GPU_ONSET_TOLERANCE = 2e-3
ONSET_COLUMNS = {"t_star", "m_star", "fit_window_t1", "total_mass_at_t1"}


def _rows(result):
    """The CSV a grid run wrote, as a DataFrame."""
    return pd.read_csv(result["csv"])


def _newest_velocity_sweep_csv():
    """The CSV most recently written by collect_char_time_mass.

    That kernel returns None -- it reports by writing files -- so the only way
    to read its answer back is to find what it just wrote.
    """
    pattern = os.path.join(str(fp.char_time_analysis_output), "*",
                           "char_t_analysis_data.csv")
    found = glob.glob(pattern)
    assert found, f"collect_char_time_mass wrote no CSV under {pattern}"
    return max(found, key=os.path.getmtime)


def test_single_point_reproduces_the_velocity_sweep():
    """A 1x1 grid at (w, w) must equal the velocity sweep at that w.

    This is the load-bearing test. Both kernels go through _char_time_point, so
    what it really asserts is that the (a, b) kernel wires its parameters into
    that shared code the same way -- in particular that a_list feeds
    switch_param_a and b_list feeds switch_param_b, the two slots the velocity
    sweep fills with a single w. Swapping them would pass every other test in
    this file and silently transpose the physics.
    """
    w = 10.0

    launch.collect_char_time_mass(RG, RY, [V], w, T, N_LIST, show_plt=False)
    reference = pd.read_csv(_newest_velocity_sweep_csv())

    grid = _rows(launch.collect_ab_grid_char_time(
        RG, RY, [w], [w], V, T, N_LIST, show_plt=False, workers=1))

    assert len(grid) == 1, f"a 1x1 grid produced {len(grid)} rows"
    assert len(reference) == 1

    mismatched = []
    for column in RESULT_COLUMNS:
        mine, theirs = grid[column][0], reference[column][0]
        if not (mine == theirs or (math.isnan(mine) and math.isnan(theirs))):
            mismatched.append(f"{column}: grid={mine!r} sweep={theirs!r}")
    assert not mismatched, ("the (a, b) grid disagrees with the velocity "
                            "sweep at the same point:\n  "
                            + "\n  ".join(mismatched))


def test_parallel_matches_sequential():
    """Concurrency must not touch the numbers.

    Each point is solved in its own process, so nothing is shared and the two
    should agree bit for bit rather than approximately. They would not if a
    worker picked up state from the parent -- a resumed checkpoint, a reused
    layer buffer -- which is exactly the failure this catches.
    """
    one = _rows(launch.collect_ab_grid_char_time(
        RG, RY, A_LIST, B_LIST, V, T, N_LIST, show_plt=False, workers=1))
    many = _rows(launch.collect_ab_grid_char_time(
        RG, RY, A_LIST, B_LIST, V, T, N_LIST, show_plt=False, workers=4))

    assert list(one["a"]) == list(many["a"]), "row order changed under workers"
    assert list(one["b"]) == list(many["b"]), "row order changed under workers"
    for column in RESULT_COLUMNS:
        np.testing.assert_array_equal(
            one[column].to_numpy(), many[column].to_numpy(),
            err_msg=f"{column} differs between sequential and parallel runs")


def test_grid_covers_the_cartesian_product():
    """Every (a, b) pair appears exactly once, a-major, correctly labelled."""
    a_list = [1.0, 5.0, 20.0]
    b_list = [2.0, 8.0]
    frame = _rows(launch.collect_ab_grid_char_time(
        RG, RY, a_list, b_list, V, T, N_LIST, show_plt=False, workers=1))

    assert len(frame) == len(a_list) * len(b_list)
    assert list(zip(frame["a"], frame["b"])) == [(a, b) for a in a_list
                                                 for b in b_list]
    assert frame["t_star"].notna().all(), "a well-posed grid produced no t*"
    assert frame["m_star"].notna().all(), "a well-posed grid produced no m*"


def test_switch_rates_actually_reach_the_solver():
    """Different (a, b) must give different answers.

    Without this, a kernel that dropped a_list and b_list on the floor and
    solved the same point every time would pass every other test here: the rows
    would be labelled correctly and agree across devices and worker counts.
    """
    frame = _rows(launch.collect_ab_grid_char_time(
        RG, RY, [1.0, 100.0], [1.0, 100.0], V, T, N_LIST,
        show_plt=False, workers=1))
    assert frame["t_star"].nunique() == len(frame), (
        "every (a, b) point returned the same t*; the switch rates are not "
        f"reaching the solver:\n{frame[['a', 'b', 't_star']]}")


def test_writes_the_csv_and_all_three_figures():
    result = launch.collect_ab_grid_char_time(
        RG, RY, [1.0, 10.0], [1.0, 10.0], V, T, N_LIST, show_plt=False,
        workers=1)

    assert os.path.isfile(result["csv"])
    assert set(result["plots"]) == {"m_star_vs_a", "m_star_vs_b",
                                    "m_star_intensity"}
    for name, path in result["plots"].items():
        assert os.path.isfile(path), f"{name} was not written"
        assert os.path.getsize(path) > 0, f"{name} is empty"
    assert result["points"] == 4
    assert result["device"] == "cpu"

    # Every file names the parameters held fixed across the run, so one lifted
    # out of its timestamped directory still says what produced it.
    stem = f"v{V:g}_N{len(N_LIST)}_{RG}x{RY}"
    for path in [result["csv"], *result["plots"].values()]:
        assert stem in os.path.basename(path), \
            f"{os.path.basename(path)} does not carry the run parameters"


def test_csv_leads_with_the_four_result_columns():
    """a, b, t_star, m_star are the results and come first, in that order.

    The fit columns follow rather than being dropped: the slope over the steady
    segment is the asymptotic decay rate, and neither it nor the segment can be recovered
    afterwards -- the mass timeseries is not retained, and a GPU run logs
    nothing per step.
    """
    frame = _rows(launch.collect_ab_grid_char_time(
        RG, RY, [1.0], [1.0], V, T, N_LIST, show_plt=False, workers=1))
    assert list(frame.columns[:4]) == ["a", "b", "t_star", "m_star"], \
        f"unexpected leading columns: {list(frame.columns)}"
    assert "fit_slope" in frame.columns
    assert (frame["tau"] == launch.CHAR_TIME_TAU).all()
    # v, N and the grid do not vary within a run, so they belong in the
    # filename, not in a column repeated on every row.
    assert "v" not in frame.columns


def test_t_star_is_the_onset_of_steady_decay():
    """On a real solve, t* is where ln(total mass) turns into a straight line.

    Checked independently of _char_time_onset's own arithmetic: the decay rate
    measured over the first half of the steady segment must match the one over
    the second half, which is what "steady" means, and a stricter tau must give
    a later onset.
    """
    loose = _rows(launch.collect_ab_grid_char_time(
        RG, RY, [10.0], [10.0], V, T, N_LIST, show_plt=False, workers=1,
        tau=1e-2)).iloc[0]
    strict = _rows(launch.collect_ab_grid_char_time(
        RG, RY, [10.0], [10.0], V, T, N_LIST, show_plt=False, workers=1,
        tau=1e-4)).iloc[0]

    assert 0 < loose["t_star"] < strict["t_star"] < T, (loose, strict)
    assert strict["tau"] == 1e-4
    t1, t2 = strict["fit_window_t1"], strict["fit_window_t2"]
    assert t1 == strict["t_star"] and math.isclose(t2, T, abs_tol=2e-3)
    # The line through the segment's two ends agrees with the least-squares
    # slope over it: the segment really is straight.
    chord = (math.log10(strict["total_mass_at_t2"])
             - math.log10(strict["total_mass_at_t1"])) / (t2 - t1)
    assert math.isclose(chord, strict["fit_slope"], rel_tol=1e-3),         (chord, strict["fit_slope"])
    assert strict["m_star"] == strict["total_mass_at_t1"]
    assert 0 < strict["m_star"] < loose["m_star"] < 1


def test_router_result_reaches_the_gui_and_the_api():
    """Both computations, run the way the desktop app and the job worker run them.

    The router turns each kernel's return value into
    {output_dirs, csv, tau, t_star: [...]}: output_dirs is what the GUI lists
    and previews, t_star what it prints and what an API job stores. The
    velocity sweep used to return None, which crashed the GUI's result handler
    (``"MFPT" in None``); the job queue did not know either computation at all.
    """
    import json
    from multiprocessing_tools import computation_router as router
    from gui_components import aux_gui_funcs, controller

    # The queue path is the router now, not a drifting copy of it.
    assert controller.run_selected_computation is router.run_selected_computation
    assert "Characteristic Time (mass vs v)" in controller.COMPUTATION_FUNCTIONS
    assert "Characteristic Time (a,b grid)" in controller.COMPUTATION_FUNCTIONS

    # Text values, as the GUI's form fields supply them.
    common = dict(rg_param=str(RG), ry_param=str(RY), T_param=str(T),
                  N_LIST=str(N_LIST.tolist()), show_plt="False")
    sweep = router.run_selected_computation(
        "Characteristic Time (mass vs v)",
        dict(common, v_LIST=f"[{V}]", w_param="10.0"))
    grid = router.run_selected_computation(
        "Characteristic Time (a,b grid)",
        dict(common, a_list="[1.0, 10.0]", b_list="[10.0]", v_param=str(V),
             workers="1", tau="0.01"))

    for result, keys, n in ((sweep, ("v",), 1), (grid, ("a", "b"), 2)):
        json.dumps(result)                       # stored as JSON by the worker
        assert os.path.isdir(result["output_dirs"][0])
        csvs, pngs = aux_gui_funcs.extract_csv_and_png_paths(result["output_dirs"])
        assert csvs and pngs, "the GUI would find nothing to list or preview"
        assert len(result["t_star"]) == n
        for row in result["t_star"]:
            assert all(k in row for k in keys)
            assert row["t_star"] is None or 0 < row["t_star"] < T
        lines = aux_gui_funcs.char_time_lines(result)
        assert len(lines) == n + 1 and "t*" in lines[1], lines
    assert grid["tau"] == 0.01 and sweep["tau"] == launch.CHAR_TIME_TAU
    assert [(r["a"], r["b"]) for r in grid["t_star"]] == [(1.0, 10.0), (10.0, 10.0)]


def test_invalid_input_is_refused():
    """Bad parameters must raise before any solving, not produce nonsense."""
    cases = [
        (dict(a_list=[], b_list=[1.0]), "at least one"),
        (dict(a_list=[1.0], b_list=[]), "at least one"),
        # Too few samples to take a second difference over: 1e-3 is fewer than
        # eight samples at 16x16, refused before any solving.
        (dict(a_list=[1.0], b_list=[1.0], T_param=1e-3), "too short"),
        (dict(a_list=[1.0], b_list=[1.0], tau=0.0), "tau must be"),
        (dict(a_list=[1.0], b_list=[1.0], device="cuda:0"), "unknown device"),
    ]
    for overrides, expected in cases:
        kwargs = dict(rg_param=RG, ry_param=RY, v_param=V, T_param=T,
                      N_LIST=N_LIST, show_plt=False, workers=1)
        kwargs.update(overrides)
        try:
            launch.collect_ab_grid_char_time(**kwargs)
        except ValueError as e:
            assert expected in str(e), \
                f"{overrides} raised the wrong error: {e}"
        else:
            raise AssertionError(f"{overrides} was accepted but should not be")


def test_gpu_matches_cpu_within_tolerance():
    """The GPU grid must agree with the CPU grid to the stated tolerance.

    Skips cleanly with no CUDA device. Bit-identity is not on offer here for the
    reason given at GPU_TOLERANCE.
    """
    if not pk.is_available():
        print("no CUDA device; skipping")
        return

    cpu = _rows(launch.collect_ab_grid_char_time(
        RG, RY, A_LIST, B_LIST, V, T, N_LIST, show_plt=False, workers=1,
        device="cpu"))
    result = launch.collect_ab_grid_char_time(
        RG, RY, A_LIST, B_LIST, V, T, N_LIST, show_plt=False, device="gpu")
    gpu = _rows(result)

    assert result["device"] == "gpu", "device=gpu did not run on the GPU"
    assert list(cpu["a"]) == list(gpu["a"])
    assert list(cpu["b"]) == list(gpu["b"])

    worst = []
    for column in RESULT_COLUMNS:
        c = cpu[column].to_numpy(dtype=np.float64)
        g = gpu[column].to_numpy(dtype=np.float64)
        error = np.max(np.abs(c - g) / np.maximum(np.abs(c), 1e-300))
        bound = (GPU_ONSET_TOLERANCE if column in ONSET_COLUMNS
                 else GPU_TOLERANCE)
        if error > bound:
            worst.append(f"{column}: relative error {error:.3e} exceeds "
                         f"{bound:.0e}")
    assert not worst, "GPU grid disagrees with CPU grid:\n  " + "\n  ".join(worst)


def test_gpu_refuses_unsupported_configurations():
    """device='gpu' must fail loudly rather than quietly using the CPU."""
    if not pk.is_available():
        print("no CUDA device; skipping")
        return
    from gpu.driver import GpuUnsupported

    try:
        launch.collect_ab_grid_char_time(
            RG, RY, [1.0], [1.0], V, T, N_LIST, show_plt=False,
            device="gpu", d_tube=0.05)
    except GpuUnsupported as e:
        assert "d_tube" in str(e), f"unexpected refusal message: {e}"
    else:
        raise AssertionError("d_tube != 0 was accepted on the GPU")


def test_auto_falls_back_when_the_gpu_cannot_run_it():
    """device='auto' with an unsupported option must still produce results."""
    result = launch.collect_ab_grid_char_time(
        RG, RY, [1.0], [1.0], V, T, N_LIST, show_plt=False,
        device="auto", d_tube=0.05)
    assert result["device"] == "cpu", \
        "auto should fall back to the CPU when d_tube != 0"
    assert os.path.isfile(result["csv"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---", flush=True)
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
