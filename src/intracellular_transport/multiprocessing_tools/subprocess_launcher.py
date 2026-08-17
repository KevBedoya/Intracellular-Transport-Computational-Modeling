import os
import sys
import subprocess
from pathlib import Path


def build_worker_args(computation_name, inputs_json, job_id):
    """Return the argv that launches a single computation in a child process.

    The invocation differs between source and frozen (PyInstaller) runs:

    * **Source**: ``sys.executable`` is a real Python interpreter, so we run
      this module as ``python -m multiprocessing_tools.subprocess_launcher``.
    * **Frozen**: ``sys.executable`` is the bundled app binary, which does not
      understand ``-m``. The binary instead re-dispatches to worker mode via
      the ``--worker`` sentinel handled in ``main.py``.

    ``inputs_json`` must already be a JSON string (the parameter dict).
    ``job_id`` scopes the result file so concurrent jobs cannot overwrite one
    another; mint one with ``compute_worker.new_job_id()``.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", computation_name, inputs_json, job_id]
    return [
        sys.executable,
        "-m",
        "multiprocessing_tools.subprocess_launcher",
        computation_name,
        inputs_json,
        job_id,
    ]


def launch_subprocess(args, extra_env=None):
    """Start a worker child process.

    ``extra_env`` is merged into the child's environment; the job worker uses
    it to point ITCM_OUTPUT_ROOT at a per-job output directory.

    The child is run with ``cwd`` at the project root, but ``python -m
    multiprocessing_tools.subprocess_launcher`` resolves against the *package*
    directory one level down -- so the import path has to be supplied
    explicitly. Relying on ``cwd`` alone made the child fail with
    ``ModuleNotFoundError: No module named 'multiprocessing_tools'``, since the
    package is not directly under the project root. Setting PYTHONPATH keeps
    the working directory (which computations use for their relative output
    paths) independent of where the package is importable from.
    """
    here = Path(__file__).resolve()
    package_dir = here.parents[1]    # src/intracellular_transport
    src_dir = here.parents[2]        # src
    project_root = here.parents[3]   # repo root

    # Three entries, each for a different resolution style the codebase uses:
    #   package_dir  -- flat imports, e.g. `from computational_tools import ...`
    #   src_dir      -- dotted imports, `from intracellular_transport.x import ...`
    #   project_root -- top-level modules that live beside the package
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(package_dir), str(src_dir), str(project_root),
                    env.get("PYTHONPATH", "")) if p
    )
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

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
    job_id = sys.argv[3]
    compute_and_send(computation_name, inputs, job_id)
