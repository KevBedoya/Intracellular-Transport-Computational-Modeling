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

    # Write to a temporary file and replace, so a poller never observes a
    # half-written result.
    tmp = result_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, default=str)
    os.replace(tmp, result_file)
