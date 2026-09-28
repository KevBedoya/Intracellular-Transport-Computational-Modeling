"""HTTP API over the job queue.

A thin layer: it validates a request, writes a row, and reads rows back. All
computation happens in ``server/worker.py``, so no endpoint here ever blocks on
a solver -- submitting a job that will run for hours returns immediately.

Flask rather than FastAPI because Flask is already present in the environment
and nothing here needs async: the API does no heavy work, and SQLite reads are
microseconds. The tradeoff is no auto-generated OpenAPI, so ``GET /`` returns a
hand-written endpoint listing.

Endpoints
    GET    /                      endpoint listing
    GET    /health                liveness + queue counts
    GET    /computations          runnable computation names
    GET    /computations/<name>   parameter schema (drives a UI form)
    POST   /helpers/d_tube        is a d_tube valid for this grid and N_LIST?
    POST   /jobs                  queue a job -> 201 {job_id}
    GET    /jobs                  list jobs (?status=&limit=)
    GET    /jobs/<id>             one job (accepts an id prefix)
    DELETE /jobs/<id>             cancel a queued job
    GET    /jobs/<id>/outputs     files this job produced
    GET    /jobs/<id>/outputs/<p> download one output file

Run:
    python server/api.py --host 127.0.0.1 --port 8000

Bind to 127.0.0.1 and reach it over Tailscale, or bind to the Tailscale
address directly. There is no authentication here on purpose -- see the note
at the bottom of this docstring.

Auth: none. This is intended to sit on a private Tailscale network shared by a
handful of known people, where the network is the trust boundary. Do not
expose it to the public internet: any caller could queue unbounded work.
"""

import argparse
import mimetypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
_PKG = os.path.join(_SRC, "intracellular_transport")
for _p in (_PKG, _SRC, _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Flask, jsonify, request, send_file  # noqa: E402

import jobstore  # noqa: E402

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Re-read templates when they change on disk. Off by default outside debug mode,
# which means an edit to ui.html needs a full restart to appear -- and restarting
# is awkward here, because the scheduled task's python child has to be killed
# separately (see serve.ps1 -Stop). ui.js is already live, being a static file.
# The cost is a stat() per render, which is nothing at this traffic.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Set by main(); None means jobstore's default location.
DB_PATH = None


def _conn():
    """A short-lived connection per request.

    SQLite connections are not safe to share across threads, and Flask may
    serve requests on several. Opening per request costs microseconds and
    avoids the whole problem.
    """
    return jobstore.connect(DB_PATH)


def _computation_names():
    from multiprocessing_tools.computation_router import COMPUTATION_FUNCTIONS

    return sorted(COMPUTATION_FUNCTIONS)


def _schema(name):
    """Schema for one computation, including each field's expected type.

    The type comes from the same table the validator uses, so a form built from
    this response coerces inputs exactly the way submission will check them.
    """
    from gui_components.params_config import PARAMETER_SCHEMAS, PARAMETER_HINTS
    from multiprocessing_tools.computation_router import _expected_type

    schema = PARAMETER_SCHEMAS.get(name)
    if schema is None:
        return None

    def kind(param, default=None, has_default=False):
        t = _expected_type(param, default, has_default)
        return t.__name__ if t else "unknown"

    return {
        "computation": name,
        "required": [
            {"name": k, "type": kind(k), "hint": PARAMETER_HINTS.get(k)}
            for k, _ in schema.get("required", [])
        ],
        "optional": [
            {"name": k, "default": v, "type": kind(k, v, True),
             "hint": PARAMETER_HINTS.get(k)}
            for k, v in schema.get("default", [])
        ],
        "approach": schema.get("approach"),
    }


def _job_output_root(job_id):
    """Mirror of worker._job_output_root, for serving results back."""
    from system_configuration import file_paths as fp

    return os.path.join(str(fp.general_output), "jobs", job_id)


# --------------------------------------------------------------------- routes
@app.get("/")
def index():
    return jsonify({
        "service": "intracellular-transport job API",
        "endpoints": {
            "GET /health": "liveness and queue counts",
            "GET /computations": "runnable computation names",
            "GET /computations/<name>": "parameter schema",
            "POST /helpers/d_tube": "check d_tube against a grid and N_LIST: {params}",
            "POST /jobs": "queue a job: {computation, params, submitted_by?}",
            "GET /jobs": "list jobs (?status=&limit=)",
            "GET /jobs/<id>": "one job (id or unambiguous prefix)",
            "DELETE /jobs/<id>": "cancel a queued job",
            "GET /jobs/<id>/outputs": "list files produced",
            "GET /jobs/<id>/outputs/<path>": "download one file",
        },
    })


@app.get("/ui")
def ui():
    """Browser front end.

    Served from this same app so it is same-origin with the API: no CORS
    configuration, no separate deployment, and no build step. The page holds no
    parameter knowledge of its own -- it builds every form from
    /computations/<name>, so registering a new computation gives it a working UI
    with no change here.
    """
    from flask import render_template

    return render_template("ui.html")


@app.get("/health")
def health():
    conn = _conn()
    try:
        return jsonify({"status": "ok", "queue": jobstore.counts_by_status(conn)})
    finally:
        conn.close()


@app.get("/computations")
def computations():
    return jsonify({"computations": _computation_names()})


@app.get("/computations/<path:name>")
def computation_schema(name):
    schema = _schema(name)
    if schema is None:
        return jsonify({"error": f"unknown computation {name!r}",
                        "computations": _computation_names()}), 404
    return jsonify(schema)


@app.get("/helpers/n_list")
def helper_n_list():
    """Evenly spaced microtubule ray indices for ?rays=<N>&tubes=<n>.

    Served rather than reimplemented in the browser on purpose: these indices
    place the microtubules, so a client-side approximation would silently change
    the physics. numpy.linspace truncates, and rounding diverges whenever the
    spacing is not exact -- for rays=50, tubes=16 the two differ in 6 of 16
    positions. This computes it with the same call the solver scripts use.
    """
    import numpy as np

    try:
        rays = int(request.args["rays"])
        tubes = int(request.args["tubes"])
    except (KeyError, ValueError):
        return jsonify({"error": "rays and tubes must both be integers"}), 400
    if tubes < 2 or rays < 2:
        return jsonify({"error": "rays and tubes must each be at least 2"}), 400
    if tubes > rays:
        return jsonify({"error": f"tubes ({tubes}) cannot exceed rays ({rays})"}), 400

    positions = np.linspace(0, rays - (rays // tubes), tubes, dtype=int).tolist()
    distinct = len(set(positions)) == len(positions)
    return jsonify({"rays": rays, "tubes": tubes, "N_LIST": positions,
                    "distinct": distinct})


@app.post("/helpers/d_tube")
def helper_d_tube():
    """Is this d_tube valid for this grid and N_LIST? {params} -> verdict.

    Answers ``{"status": "valid" | "invalid" | "incomplete", "message",
    "max_d_tube"?}``. "incomplete" means the grid or N_LIST is not yet usable,
    so there is nothing to judge against -- the form says so rather than
    guessing. Served for the same reason as /helpers/n_list: the bound is the
    solver's own arithmetic (computation_router.d_tube_limit), and a copy in
    the browser would drift from it. POST /jobs applies the same check, so the
    form cannot be bypassed.
    """
    body = request.get_json(silent=True)
    params = (body or {}).get("params") if isinstance(body, dict) else None
    if not isinstance(params, dict):
        return jsonify({"error": "body must be {\"params\": {...}}"}), 400

    def number(key):
        v = params.get(key)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    rg, ry, d_tube = number("rg_param"), number("ry_param"), number("d_tube")
    nlist = params.get("N_LIST")
    if d_tube is None:
        return jsonify({"status": "incomplete",
                        "message": "enter a number to check it"})
    ok_list = (isinstance(nlist, list) and nlist
               and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and float(x) == int(x) for x in nlist))
    if (rg is None or ry is None or rg != int(rg) or ry != int(ry)
            or rg < 2 or ry < 2 or not ok_list
            or len({int(x) for x in nlist}) != len(nlist)
            or min(nlist) < 0 or max(nlist) > ry - 1):
        return jsonify({"status": "incomplete",
                        "message": "set a valid rg_param, ry_param and N_LIST "
                                   "to check d_tube against them"})

    from multiprocessing_tools.computation_router import check_d_tube

    verdict = check_d_tube(int(rg), int(ry), [int(x) for x in nlist], d_tube)
    return jsonify({"status": "valid" if verdict["valid"] else "invalid",
                    "message": verdict["message"],
                    "max_d_tube": verdict["max_d_tube"]})


@app.post("/jobs")
def submit_job():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400

    computation = body.get("computation")
    params = body.get("params")
    if not computation:
        return jsonify({"error": "missing 'computation'"}), 400
    if params is None:
        params = {}

    # Validate before touching the store so the response can name the offending
    # fields individually -- a form needs to know *which* input to highlight,
    # not just that something was wrong.
    from multiprocessing_tools.computation_router import validate_params

    problems = validate_params(computation, params)
    if problems:
        return jsonify({"error": "invalid parameters", "fields": problems}), 400

    conn = _conn()
    try:
        job_id = jobstore.submit(conn, computation, params,
                                 submitted_by=body.get("submitted_by"))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    return jsonify({"job_id": job_id, "status": jobstore.STATUS_QUEUED}), 201


def _is_gpu(job):
    return str((job.get("params") or {}).get("device", "")).lower() in (
        "gpu", "cuda", "auto")


def _with_effective_estimate(jobs, running):
    """Annotate jobs with the estimate that accounts for a shared GPU.

    ``est_seconds`` is the solo figure stored at submission -- what the job would
    take with the card to itself. ``est_seconds_effective`` stretches it by the
    number of GPU jobs currently sharing, which is the number worth showing in a
    Remaining column: three concurrent jobs turned a 6.6 hr solo estimate into a
    22.5 hr actual, so the unadjusted figure is not a small error.

    Backfills ``est_seconds`` for rows submitted before the column existed, so
    the queue does not show a column of dashes for the jobs already in flight.
    """
    import estimate as est_mod

    gpu_running = sum(1 for j in running if _is_gpu(j))
    out = []
    for j in jobs:
        j = dict(j)
        if j.get("est_seconds") is None:
            try:
                j["est_seconds"] = est_mod.seconds(j.get("computation"),
                                                   j.get("params") or {})
            except Exception:
                j["est_seconds"] = None
        base = j.get("est_seconds")
        if base is None:
            j["est_seconds_effective"] = None
        elif j.get("status") == jobstore.STATUS_RUNNING and _is_gpu(j):
            j["est_seconds_effective"] = base * est_mod.share_factor(
                "gpu", gpu_running)
        else:
            j["est_seconds_effective"] = base
        j["gpu_running"] = gpu_running
        out.append(j)
    return out


@app.post("/helpers/estimate")
def estimate_job():
    """Cost a parameter set without submitting it.

    Server-side so the browser form and the queue agree. The model used to live
    in the page and knew only about the CPU, which made a GPU job's quoted wall
    time wrong by more than an order of magnitude.
    """
    import estimate as est_mod

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    params = body.get("params") or {}
    if not isinstance(params, dict):
        return jsonify({"error": "'params' must be an object"}), 400

    info = est_mod.describe(body.get("computation"), params,
                            device=body.get("device"))

    # How many GPU jobs are already running, so the caller can show what the
    # card being shared will actually cost.
    conn = _conn()
    try:
        running = jobstore.list_jobs(conn, status=jobstore.STATUS_RUNNING,
                                     limit=1000)
    finally:
        conn.close()
    gpu_running = sum(
        1 for j in running
        if str((j.get("params") or {}).get("device", "")).lower()
        in ("gpu", "cuda", "auto"))

    info["gpu_running"] = gpu_running
    if info["seconds"] is not None and info["device"] == "gpu":
        # The submitted job would be one more sharer than are running now.
        info["seconds_shared"] = info["seconds"] * est_mod.share_factor(
            "gpu", gpu_running + 1)
    else:
        info["seconds_shared"] = info["seconds"]
    return jsonify(info)


@app.get("/jobs")
def list_jobs():
    try:
        limit = min(int(request.args.get("limit", 100)), 1000)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    status = request.args.get("status")
    if status and status not in (
        jobstore.STATUS_QUEUED, jobstore.STATUS_RUNNING,
        jobstore.STATUS_SUCCEEDED, jobstore.STATUS_FAILED,
        jobstore.STATUS_CANCELLED,
    ):
        return jsonify({"error": f"unknown status {status!r}"}), 400

    conn = _conn()
    try:
        jobs = jobstore.list_jobs(conn, status=status, limit=limit)
        # Sharing depends on what else is running, so it cannot be stored on the
        # row at submission -- it has to be computed against the live queue.
        running = (jobs if status == jobstore.STATUS_RUNNING
                   else jobstore.list_jobs(conn, status=jobstore.STATUS_RUNNING,
                                           limit=1000))
    finally:
        conn.close()
    return jsonify({"jobs": _with_effective_estimate(jobs, running)})


@app.get("/jobs/<job_id>")
def get_job(job_id):
    conn = _conn()
    try:
        job = jobstore.get(conn, job_id)
        running = jobstore.list_jobs(conn, status=jobstore.STATUS_RUNNING,
                                     limit=1000) if job else []
    except ValueError as e:            # ambiguous prefix
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()
    if job is None:
        return jsonify({"error": f"no job {job_id}"}), 404
    return jsonify(_with_effective_estimate([job], running)[0])


@app.delete("/jobs/<job_id>")
def cancel_job(job_id):
    conn = _conn()
    try:
        job = jobstore.get(conn, job_id)
        if job is None:
            return jsonify({"error": f"no job {job_id}"}), 404
        if jobstore.cancel(conn, job["id"]):
            return jsonify({"job_id": job["id"],
                            "status": jobstore.STATUS_CANCELLED})
        # Running jobs own a subprocess and possibly hours of partial output;
        # stopping one is the worker's business, not a store update.
        return jsonify({
            "error": "only queued jobs can be cancelled",
            "job_id": job["id"], "status": job["status"],
        }), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()


@app.get("/jobs/<job_id>/outputs")
def job_outputs(job_id):
    conn = _conn()
    try:
        job = jobstore.get(conn, job_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()
    if job is None:
        return jsonify({"error": f"no job {job_id}"}), 404

    root = _job_output_root(job["id"])
    files = []
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.startswith("."):
                    continue
                full = os.path.join(dirpath, fn)
                files.append({
                    "path": os.path.relpath(full, root).replace(os.sep, "/"),
                    "bytes": os.path.getsize(full),
                })
    return jsonify({"job_id": job["id"], "status": job["status"],
                    "files": sorted(files, key=lambda f: f["path"])})


@app.get("/jobs/<job_id>/outputs/<path:rel>")
def job_output_file(job_id, rel):
    conn = _conn()
    try:
        job = jobstore.get(conn, job_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()
    if job is None:
        return jsonify({"error": f"no job {job_id}"}), 404

    root = os.path.realpath(_job_output_root(job["id"]))
    target = os.path.realpath(os.path.join(root, rel))
    # Contain the download inside the job's own directory: rel comes from the
    # caller, so without this check "../../.." would read arbitrary files.
    if target != root and not target.startswith(root + os.sep):
        return jsonify({"error": "path outside job output directory"}), 403
    if not os.path.isfile(target):
        return jsonify({"error": f"no such output {rel!r}"}), 404

    mime = mimetypes.guess_type(target)[0] or "application/octet-stream"
    return send_file(target, mimetype=mime,
                     as_attachment=not mime.startswith("image/"))


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "internal server error"}), 500


def main(argv=None):
    global DB_PATH
    p = argparse.ArgumentParser(description="Job queue HTTP API.")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1; reach it over "
                        "Tailscale rather than exposing it)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default=None, help="path to the job database")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args(argv)

    DB_PATH = a.db
    # Touch the store once at startup so a bad path fails now, not on the
    # first request.
    jobstore.connect(DB_PATH).close()
    print(f"db: {a.db or jobstore.default_db_path()}", flush=True)
    print(f"listening on http://{a.host}:{a.port}", flush=True)
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)


if __name__ == "__main__":
    main()
