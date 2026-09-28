"""
Application entry point for the Biophysics GUI.

Version 2.2

This module has two run modes:

* **GUI mode** (default) -- launches the PyQt5 desktop application.
* **Worker mode** (``--worker``) -- runs a single headless computation and
  writes its result to disk. This path exists for *frozen* (PyInstaller)
  builds, where ``sys.executable`` is the bundled app binary rather than a
  Python interpreter, so the usual ``python -m multiprocessing_tools.
  subprocess_launcher`` invocation is unavailable. The GUI re-invokes itself
  with ``--worker`` to run heavy computations in a child process. In source
  runs the original ``-m`` launcher is still used (see
  ``multiprocessing_tools.subprocess_launcher.build_worker_args``).
"""

import sys
import os
import multiprocessing
import time
import json


def package_on_path():
    """Put the package and its parent directory on ``sys.path``.

    The project uses a src layout, so the package lives at
    ``src/intracellular_transport``. Two entries are required because the
    codebase mixes import styles:

      * flat   -- ``from computational_tools import ...`` resolves against the
                  package directory itself
      * dotted -- ``from intracellular_transport.x import ...`` resolves
                  against ``src/``

    Returns the package directory.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    pkg = os.path.join(src, "intracellular_transport")
    for entry in (pkg, src):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return pkg


def _run_worker(argv):
    """Run a single computation in worker mode and exit.

    ``argv`` is ``sys.argv``; positions 2, 3 and 4 hold the computation name,
    the JSON-encoded parameter dict and the job id, matching the contract used
    by ``multiprocessing_tools.subprocess_launcher``.
    """

    from multiprocessing_tools.compute_worker import compute_and_send

    computation_name = argv[2]
    inputs = json.loads(argv[3])
    job_id = argv[4]
    compute_and_send(computation_name, inputs, job_id)


def run_main():
    """Launch the desktop GUI."""
    from intracellular_transport.gui_components import main_gui as gui
    gui.run_app()


# ---------------------------------------------------------------------------
# Deterministic analytic-vs-numerical verification examples (run with --verify).
# Each entry is a fully-specified parameter set for
# computational_tools.analytic_benchmark.deterministic_verification, which writes
# its figures under data_output/analytic_verification/<label>/.  Edit / extend
# these for reproducible future runs.
# ---------------------------------------------------------------------------
VERIFICATION_EXAMPLES = [
    # (label, kwargs) -- 32x32 grid, 4 evenly-spaced microtubules, R=D=1.
    dict(label="strong_coupling_v1_ab100",
         rings=32, rays=32, microtubule_list=[0, 8, 16, 24],
         a=100, b=100, v=1, timestamps=[0.05, 0.1, 0.2, 0.3, 0.4], T=0.4,
         show_mse=True),
    dict(label="strong_advection_v100_ab1",
         rings=32, rays=32, microtubule_list=[0, 8, 16, 24],
         a=1, b=1, v=100, timestamps=[0.05, 0.1, 0.2, 0.3, 0.4], T=0.4,
         show_mse=True),
    dict(label="balanced_v10_ab10",
         rings=32, rays=32, microtubule_list=[0, 8, 16, 24],
         a=10, b=10, v=10, timestamps=[0.05, 0.1, 0.2, 0.3, 0.4], T=0.4,
         show_mse=True),
]


def run_verification_examples():
    """Run the deterministic verification presets in VERIFICATION_EXAMPLES."""
    package_on_path()
    from computational_tools import analytic_benchmark as ab
    for cfg in VERIFICATION_EXAMPLES:
        print(f"\n=== verification example: {cfg['label']} ===")
        ab.deterministic_verification(**cfg)


# ---------------------------------------------------------------------------
# Concurrent characteristic-time sweep (96x96)
# ---------------------------------------------------------------------------
# WHY THESE JOBS RUN AS SEPARATE CONCURRENT PROCESSES
#
# The four microtubule configurations (24 / 16 / 8 / 4 tubes) are completely
# independent of one another: they share no state, and each writes its results
# into its own timestamp-stamped directory under
# ``data_output/char_time_analysis/``.  Running them concurrently is therefore
# safe and is a straight ~4x wall-clock win, for three measured reasons:
#
# 1. The solver is single-threaded.  ``comp_mass_analysis_respect_to_time`` is
#    an ``@njit`` function with a serial time-stepping loop; the running process
#    was measured at 0.99 cores busy (322 CPU-sec / 324 wall-sec).  This machine
#    has 12 logical cores, so a sequential sweep leaves 11 of them idle.
#
# 2. The jobs are near-identical in cost, so they finish together.  ``K`` (the
#    number of timesteps) depends only on the grid and ``T``, not on the number
#    of microtubules -- so all four jobs run the same 43,028,399 steps.  Tube
#    count only affects the advective inner work, measured as a ~4% effect
#    (slowest/fastest = 1.04x).  There is no long-tail straggler.
#
# 3. Separate processes fix a figure-reuse bug.  ``collect_char_time_mass``
#    calls ``plt.scatter`` but never ``plt.clf()``/``plt.close()``.  Run
#    sequentially in ONE process, each job's PNG accumulates every previous
#    job's scatter points (job 4's plot would show 4 points instead of 1).  The
#    CSVs are unaffected because they are rebuilt per job.  Giving each job its
#    own process gives it its own matplotlib figure, so each PNG contains only
#    its own data.
#
# Measured cost: ~10.7 hr per job, so ~43 hr sequential vs ~10.7 hr concurrent.
# Each child JIT-compiles independently (~30 s), which overlaps across workers.
# ---------------------------------------------------------------------------

# One source of truth for the sweep, shared by the workers and the docs above.
CHAR_TIME_SWEEP = dict(
    rings=96,
    rays=96,
    v_list=[10 ** 4],
    w=100,          # switch rate (a = b = w)
    T=1,            # dimensionless solution duration
)

# Microtubule counts to sweep, one concurrent process each.  Ordered
# heaviest-first so the longest job starts earliest.
CHAR_TIME_TUBE_COUNTS = (24, 16, 8, 4)


def _char_time_pkg_on_path():
    """Put ``intracellular_transport`` on ``sys.path`` (needed in each child).

    Workers are started with the "spawn" method, so every child re-imports this
    module from scratch and must set its own import path up again.
    """
    return package_on_path()


def _run_one_char_time_job(job):
    """Run a single ``collect_char_time_mass`` job.

    ``job`` is ``(n_tubes, rings, rays)``.  The grid is passed *through the
    argument* rather than read from the CHAR_TIME_SWEEP global on purpose: the
    "spawn" start method re-imports this module in every child, so a grid that
    the parent overrode at runtime would not reach the workers -- they would
    silently fall back to the module-level default.

    Executed in its own process (see CHAR_TIME_SWEEP notes above).  Must stay a
    module-level function so it is picklable by "spawn".

    Returns ``(n_tubes, elapsed_seconds)``; the caller prints the summary so the
    four workers do not interleave their reports mid-line.
    """
    import time as _time

    import numpy as np

    n_tubes, rings, rays = job

    _char_time_pkg_on_path()
    from launch_functions import launch

    cfg = CHAR_TIME_SWEEP

    # Evenly-spaced microtubule angular positions, e.g. for 96 rays and 24
    # tubes: linspace(0, 96 - 96//24, 24) -> every 4th ray.
    N_LIST = np.linspace(0, rings - (rings // n_tubes), n_tubes, dtype=int)

    print(f"[{n_tubes:>2} tubes] starting ({rings}x{rays}, "
          f"v={cfg['v_list']}, a=b={cfg['w']}, T={cfg['T']})", flush=True)

    start = _time.perf_counter()
    launch.collect_char_time_mass(rings, rays, cfg["v_list"], cfg["w"],
                                  cfg["T"], N_LIST, show_plt=False)
    elapsed = _time.perf_counter() - start

    print(f"[{n_tubes:>2} tubes] done in {elapsed/3600:.2f} hr "
          f"({elapsed/60:.1f} min)", flush=True)
    return n_tubes, elapsed


def run_char_time_sweep_concurrent(rings=None, rays=None):
    """Run the four microtubule configurations as concurrent processes.

    Uses one worker per configuration so all four proceed in parallel on
    separate cores.  Wall time for the whole sweep is therefore roughly the
    cost of a single job rather than the sum of all four.

    ``rings`` / ``rays`` override the grid in CHAR_TIME_SWEEP for this run, so
    the same sweep can be repeated at another resolution without editing the
    module.  The override is forwarded to each worker as part of its job tuple
    (see _run_one_char_time_job on why it cannot go through the global).

    Cost scales steeply with the grid: dT = 0.1*dThe^2*dRad^2/(2D) means
    K ~ G^4 and total work ~ G^6 for a GxG grid, so halving the grid from 96 to
    48 cuts each job by roughly 64x (measured: 43.0M -> 2.69M timesteps).
    """
    import time as _time

    cfg = CHAR_TIME_SWEEP
    rings = cfg["rings"] if rings is None else rings
    rays = cfg["rays"] if rays is None else rays
    n_jobs = len(CHAR_TIME_TUBE_COUNTS)

    print(f"=== concurrent characteristic-time sweep ===")
    print(f"grid          : {rings}x{rays}")
    print(f"v list        : {cfg['v_list']}")
    print(f"a = b         : {cfg['w']}")
    print(f"T             : {cfg['T']}")
    print(f"tube counts   : {list(CHAR_TIME_TUBE_COUNTS)}")
    print(f"processes     : {n_jobs} (one per tube count, running concurrently)")
    print(flush=True)

    sweep_start = _time.perf_counter()

    # "spawn" is the only start method on Windows and keeps each worker's
    # numba/matplotlib state fully isolated.
    jobs = [(n, rings, rays) for n in CHAR_TIME_TUBE_COUNTS]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        results = pool.map(_run_one_char_time_job, jobs)

    sweep_elapsed = _time.perf_counter() - sweep_start

    print()
    print(f"=== sweep complete ({rings}x{rays}) ===")
    print(f"{'tubes':>6} {'wall time':>12}")
    serial_total = 0.0
    for n_tubes, elapsed in results:
        serial_total += elapsed
        print(f"{n_tubes:>6} {elapsed/3600:>9.2f} hr ({elapsed/60:.1f} min)")
    print()
    print(f"sum of job times (what a sequential run would cost): "
          f"{serial_total/3600:.2f} hr")
    print(f"actual wall time for the concurrent sweep          : "
          f"{sweep_elapsed/3600:.2f} hr")
    if sweep_elapsed > 0:
        print(f"effective speedup from concurrency                 : "
              f"{serial_total/sweep_elapsed:.2f}x")


def run_super_comp_off_center():
    """Run the super computation (launch_super_comp_I) under an OFF-CENTERED
    initial condition.

    Instead of seeding the unit initial mass in the central patch, the whole
    unit mass is placed at the discrete patch (m_init, n_init): the m_init-th
    ring and the n_init-th ray. Set ``center_init_cond=True`` (or just remove
    the off-center kwargs) to fall back to the default centered scheme.

    Results (CSVs / PNGs / heatmaps) are written under
    intracellular_transport/data_output, exactly as in a normal super run.
    """
    import numpy as np
    package_on_path()
    from launch_functions import launch
    from data_visualization import plot_functions
    from computational_tools import numerical_tools

    # --- domain / physics parameters ---
    rg_param = 96                       # rings  (M)
    ry_param = 96                       # rays   (N)
    v_param = .1                       # advective velocity on microtubules
    w_param = 100                        # switch rate (a = b = w)
    T_param = 1                       # dimensionless solution duration
    N_LIST = np.array([0, 4, 8, 12])   # microtubule angular positions
    d_tube = 0.0                        # microtubule extraction width
    Timestamp_List = [1, 1.5, 2, 2.5]   # snapshot times (each <= T_param)

    # --- off-centered initial condition ---
    # Unit mass seeded at patch (m_init, n_init).
    #   m_init must be in [0, rg_param - 1]   (ring index)
    #   n_init must be in [0, ry_param - 1]   (ray index)
    center_init_cond = True
    m_init = 16
    n_init = 5

    # launch.collect_mass_analysis(rg_param, ry_param, v_param, w_param, w_param, T_param, N_LIST)
    # (the two switch rates are separate now: a onto DL, b onto AL; a = b = w recovers the old call)

    # Task for 8/5/2026
    # candidate V list : [0.1, 1, 10, 100, 1000, 10**4], for 96x96, N=4,8,16,24, a=b=10, 100

    # --- Timed runs with 96x96 grid (SUPERSEDED) ----------------------------
    # These four blocks ran sequentially in a single process, which cost ~43 hr
    # and also let each job's PNG inherit the previous jobs' scatter points
    # (see the CHAR_TIME_SWEEP notes above).  They are kept here only for
    # reference; use run_char_time_sweep_concurrent() instead, which runs the
    # same four configurations as concurrent processes in ~10.7 hr.
    #
    # print("\n=== Starting timed runs with 96x96 grid ===")
    #
    # for n_tubes in (24, 16, 8, 4):
    #     print(f"\nBlock: {n_tubes} microtubules")
    #     start_time = time.time()
    #     launch.collect_char_time_mass(
    #         rg_param, ry_param, [10 ** 4], w_param, T_param,
    #         np.linspace(0, 96 - (96 // n_tubes), n_tubes, dtype=int),
    #         show_plt=False)
    #     elapsed = time.time() - start_time
    #     print(f"Wall time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    #
    # print("\n=== All runs completed ===")


if __name__ == "__main__":
    # Required for frozen builds and the "spawn" start method so child
    # processes do not re-run the top-level application logic.
    # multiprocessing.freeze_support()

    # --- Concurrent characteristic-time sweep -------------------------------
    # Runs the 24 / 16 / 8 / 4 microtubule configurations as four concurrent
    # processes.  See the CHAR_TIME_SWEEP block above for why this is safe and
    # what it costs.  multiprocessing.freeze_support() is required here because
    # the "spawn" start method re-imports this module in every child.
    #
    # Optional square-grid override, for repeating the sweep at another
    # resolution without editing the module:
    #     python main.py            -> CHAR_TIME_SWEEP default (96x96)
    #     python main.py 48         -> 48x48
    multiprocessing.freeze_support()
    _grid = None
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        _grid = int(sys.argv[1])
    run_char_time_sweep_concurrent(rings=_grid, rays=_grid)

    # --- Off-centered super-computation run ---------------------------------
    # run_super_comp_off_center()

    # --- Previous setup (GUI / worker / verification dispatch) --------------
    # Re-enable this block (and comment out the call above) to restore the
    # normal desktop-app entry point.
    #
    # if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
    #     _run_worker(sys.argv)
    # elif len(sys.argv) >= 2 and sys.argv[1] == "--verify":
    #     run_verification_examples()
    # else:
    #     multiprocessing.set_start_method("spawn", force=True)
    #     run_main()
