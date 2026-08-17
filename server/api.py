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
    from gui_components.params_config import PARAMETER_SCHEMAS, PARAMETER_HINTS

    schema = PARAMETER_SCHEMAS.get(name)
    if schema is None:
        return None
    return {
        "computation": name,
        "required": [
            {"name": k, "hint": PARAMETER_HINTS.get(k)}
            for k, _ in schema.get("required", [])
        ],
        "optional": [
            {"name": k, "default": v, "hint": PARAMETER_HINTS.get(k)}
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
            "POST /jobs": "queue a job: {computation, params, submitted_by?}",
            "GET /jobs": "list jobs (?status=&limit=)",
            "GET /jobs/<id>": "one job (id or unambiguous prefix)",
            "DELETE /jobs/<id>": "cancel a queued job",
            "GET /jobs/<id>/outputs": "list files produced",
            "GET /jobs/<id>/outputs/<path>": "download one file",
        },
    })


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

    conn = _conn()
    try:
        # jobstore.submit validates the computation name and params type, so a
        # typo is rejected now rather than hours later at the front of the
        # queue.
        job_id = jobstore.submit(conn, computation, params,
                                 submitted_by=body.get("submitted_by"))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    return jsonify({"job_id": job_id, "status": jobstore.STATUS_QUEUED}), 201


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
        return jsonify({"jobs": jobstore.list_jobs(conn, status=status,
                                                   limit=limit)})
    finally:
        conn.close()


@app.get("/jobs/<job_id>")
def get_job(job_id):
    conn = _conn()
    try:
        job = jobstore.get(conn, job_id)
    except ValueError as e:            # ambiguous prefix
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()
    if job is None:
        return jsonify({"error": f"no job {job_id}"}), 404
    return jsonify(job)


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
