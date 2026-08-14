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
import multiprocessing


def _run_worker(argv):
    """Run a single computation in worker mode and exit.

    ``argv`` is ``sys.argv``; positions 2 and 3 hold the computation name and
    the JSON-encoded parameter dict, matching the contract used by
    ``multiprocessing_tools.subprocess_launcher``.
    """

    from multiprocessing_tools.compute_worker import compute_and_send

    computation_name = argv[2]
    inputs = json.loads(argv[3])
    compute_and_send(computation_name, inputs)


def run_main():
    """Launch the desktop GUI."""
    from project_src_package_2025.gui_components import main_gui as gui
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
    import os
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "project_src_package_2025")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from computational_tools import analytic_benchmark as ab
    for cfg in VERIFICATION_EXAMPLES:
        print(f"\n=== verification example: {cfg['label']} ===")
        ab.deterministic_verification(**cfg)


def run_super_comp_off_center():
    """Run the super computation (launch_super_comp_I) under an OFF-CENTERED
    initial condition.

    Instead of seeding the unit initial mass in the central patch, the whole
    unit mass is placed at the discrete patch (m_init, n_init): the m_init-th
    ring and the n_init-th ray. Set ``center_init_cond=True`` (or just remove
    the off-center kwargs) to fall back to the default centered scheme.

    Results (CSVs / PNGs / heatmaps) are written under
    project_src_package_2025/data_output, exactly as in a normal super run.
    """
    import os
    import numpy as np
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "project_src_package_2025")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from launch_functions import launch
    from data_visualization import plot_functions
    from computational_tools import numerical_tools

    # --- domain / physics parameters ---
    rg_param = 48                       # rings  (M)
    ry_param = 48                       # rays   (N)
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

    # launch.collect_mass_analysis(rg_param, ry_param, v_param, w_param, T_param, N_LIST)

    # Task for 8/5/2026
    # candidate V list : [0.1, 1, 10, 100, 1000, 10**4], for 48x48, N=4,8,16, a=b=10, 100

    # launch.collect_char_time_mass(rg_param, ry_param, [0.1, 1, 10, 100, 1000, 10**4], w_param, T_param, np.linspace(0, 48 - (48//4), 4, dtype=int))
    # launch.collect_char_time_mass(rg_param, ry_param, [0.1, 1, 10, 100, 1000, 10 ** 4], w_param, T_param, np.linspace(0, 48 - (48//8), 8, dtype=int))
    # launch.collect_char_time_mass(rg_param, ry_param, [0.1, 1, 10, 100, 1000, 10 ** 4], w_param, T_param, np.linspace(0, 48 - (48//16), 8, dtype=int), show_plt=False)

    launch.collect_char_time_mass(rg_param, ry_param, [10 ** 4], w_param, T_param, np.linspace(0, 48 - (48//24), 24, dtype=int), show_plt=False)


if __name__ == "__main__":
    # Required for frozen builds and the "spawn" start method so child
    # processes do not re-run the top-level application logic.
    # multiprocessing.freeze_support()

    run_super_comp_off_center()
    # --- Off-centered super-computation run ---------------------------------
    # run_super_comp_off_center()

    # --- Previous setup (GUI / worker / verification dispatch) --------------
    # Re-enable this block (and comment out the call above) to restore the
    # normal desktop-app entry point.
    #
    # if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
    #     _run_worker(sys.argv)
    # elif len(sys.argv) >= 2 and sys.argv[1] == "--verify":
    #     run_verification_examples()
    # else:
    #     multiprocessing.set_start_method("spawn", force=True)
    #     run_main()
