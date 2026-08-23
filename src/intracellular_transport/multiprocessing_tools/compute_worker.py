"""Run a single computation in a child process and write its result to disk.

Results are written per job.  Every worker used to write to one fixed
``result.json``, which is fine for a single desktop user running one
computation at a time, but silently corrupts results as soon as two
computations overlap -- the second overwrites the first, and whichever poller
reads first consumes a result that may belong to the other job.  That makes it
unusable as a server, where concurrent jobs are the normal case.

Each run now writes ``result_<job_id>.json``.  Callers mint an id with
``new_job_id()`` and read it back with ``result_path(job_id)``.
"""

import json
import os
import uuid


def new_job_id():
    """Return a fresh job identifier (hex, filename-safe)."""
    return uuid.uuid4().hex


def result_path(job_id, output_root=None):
    """Absolute path of the result file for ``job_id``.

    Kept here so producers and consumers cannot drift apart on the naming.

    ``output_root`` must be supplied when the reader and the writer disagree
    about the output tree. The job worker runs each child with its own
    ITCM_OUTPUT_ROOT, so the child resolves this path under that root while
    the worker itself does not -- the worker passes the job's root explicitly
    to look in the same place the child wrote.
    """
    if output_root is not None:
        base = os.path.join(str(output_root), "json_output")
    else:
        from system_configuration import file_paths as fp

        base = str(fp.json_output)
    return os.path.join(base, f"result_{job_id}.json")


def compute_and_send(computation_name, param_dict, job_id):
    """Run ``computation_name`` and write its outcome to that job's result file.

    Always writes the file -- on success with ``status="ok"`` and the result
    payload, on failure with ``status="error"`` and the message -- so a poller
    can distinguish "still running" (no file) from "finished and failed"
    (file with an error) rather than waiting forever on a crashed job.
    """
    from multiprocessing_tools import computation_router
    from system_configuration import file_paths as fp

    os.makedirs(str(fp.json_output), exist_ok=True)
    result_file = result_path(job_id)

    try:
        result = computation_router.run_selected_computation(computation_name, param_dict)
        output = {"status": "ok", "result": result}
    except Exception as e:
        output = {"status": "error", "message": str(e)}

    # Re-create the directory immediately before writing, not just at entry
    # above. For a multi-day solve those two moments are separated by the whole
    # run, and anything that removes the tree in between turns a completed
    # computation into a job the worker reports as a hard crash -- which is
    # exactly what happened to a 160x160 run on 2026-08-22: 82 hours of solving
    # finished, the CSV and plot were written, and only this write failed.
    #
    # The result is also worth more than the directory it lives in, so fall back
    # to the shared tree rather than losing it. A result in the wrong place can
    # be found; a result never written cannot.
    try:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        target = result_file
    except OSError as e:
        from system_configuration import file_paths as fp2

        target = os.path.join(str(fp2.json_output), f"result_{job_id}.json")
        print(f"WARNING: could not create {os.path.dirname(result_file)} ({e}); "
              f"writing the result to {target} instead")
        os.makedirs(os.path.dirname(target), exist_ok=True)

    # Write to a temporary file and replace, so a poller never observes a
    # half-written result.
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, default=str)
    os.replace(tmp, target)
