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
    import json
    from multiprocessing_tools.compute_worker import compute_and_send

    computation_name = argv[2]
    inputs = json.loads(argv[3])
    compute_and_send(computation_name, inputs)


def run_main():
    """Launch the desktop GUI."""
    from project_src_package_2025.gui_components import main_gui as gui
    gui.run_app()


if __name__ == "__main__":
    # Required for frozen builds and the "spawn" start method so child
    # processes do not re-run the top-level application logic.
    multiprocessing.freeze_support()

    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _run_worker(sys.argv)
    else:
        multiprocessing.set_start_method("spawn", force=True)
        run_main()
