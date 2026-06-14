import os
import sys
import subprocess
from pathlib import Path


def build_worker_args(computation_name, inputs_json):
    """Return the argv that launches a single computation in a child process.

    The invocation differs between source and frozen (PyInstaller) runs:

    * **Source**: ``sys.executable`` is a real Python interpreter, so we run
      this module as ``python -m multiprocessing_tools.subprocess_launcher``.
    * **Frozen**: ``sys.executable`` is the bundled app binary, which does not
      understand ``-m``. The binary instead re-dispatches to worker mode via
      the ``--worker`` sentinel handled in ``main.py``.

    ``inputs_json`` must already be a JSON string (the parameter dict).
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", computation_name, inputs_json]
    return [
        sys.executable,
        "-m",
        "multiprocessing_tools.subprocess_launcher",
        computation_name,
        inputs_json,
    ]


def launch_subprocess(args):

    # project_root = "/Users/kbedoya88/Desktop/QC25-Summer/Research/Computational-Biophysics/Comp-Bio-Summer/June-2025"
    # project_root = "/Users/kbedoya88"
    # project_root = ""
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()

    return subprocess.Popen(
        args,
        cwd=project_root,
        env=env
    )


if __name__ == "__main__":
    from multiprocessing_tools.compute_worker import compute_and_send
    import sys
    import json

    computation_name = sys.argv[1]
    inputs = json.loads(sys.argv[2])
    compute_and_send(computation_name, inputs)
