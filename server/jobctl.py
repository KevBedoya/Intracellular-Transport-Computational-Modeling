"""Command-line front end to the job store.

Enough to drive the queue without a web UI; the HTTP API will call the same
jobstore functions.

    python server/jobctl.py computations
    python server/jobctl.py schema "Characteristic Time (mass vs v)"
    python server/jobctl.py submit "Characteristic Time (mass vs v)" params.json
    python server/jobctl.py list
    python server/jobctl.py show <job_id>
    python server/jobctl.py cancel <job_id>
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
_PKG = os.path.join(_SRC, "intracellular_transport")
for _p in (_PKG, _SRC, _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jobstore  # noqa: E402


def _computations():
    from multiprocessing_tools.computation_router import COMPUTATION_FUNCTIONS
    return sorted(COMPUTATION_FUNCTIONS)


def cmd_computations(_a):
    for name in _computations():
        print(name)


def cmd_schema(a):
    from gui_components.params_config import PARAMETER_SCHEMAS, PARAMETER_HINTS

    schema = PARAMETER_SCHEMAS.get(a.computation)
    if schema is None:
        sys.exit(f"no schema for {a.computation!r}")
    out = {
        "computation": a.computation,
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
    print(json.dumps(out, indent=2, default=str))


def cmd_submit(a):
    if a.params == "-":
        params = json.load(sys.stdin)
    elif os.path.exists(a.params):
        with open(a.params) as f:
            params = json.load(f)
    else:
        params = json.loads(a.params)

    conn = jobstore.connect(a.db)
    try:
        job_id = jobstore.submit(conn, a.computation, params,
                                 submitted_by=a.by)
    except (ValueError, TypeError) as e:
        sys.exit(str(e))
    print(job_id)
    conn.close()


def _fmt_row(j):
    dur = ""
    if j.get("started_at") and j.get("finished_at"):
        from datetime import datetime
        try:
            d = (datetime.fromisoformat(j["finished_at"])
                 - datetime.fromisoformat(j["started_at"]))
            dur = f"{d.total_seconds()/60:.1f}m"
        except ValueError:
            pass
    return (f"{j['id'][:8]}  {j['status']:<10} {dur:>7}  "
            f"{j['computation'][:44]}")


def cmd_list(a):
    conn = jobstore.connect(a.db)
    jobs = jobstore.list_jobs(conn, status=a.status, limit=a.limit)
    if not jobs:
        print("(no jobs)")
    else:
        print(f"{'id':<8}  {'status':<10} {'dur':>7}  computation")
        for j in jobs:
            print(_fmt_row(j))
        counts = jobstore.counts_by_status(conn)
        print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    conn.close()


def cmd_show(a):
    conn = jobstore.connect(a.db)
    job = jobstore.get(conn, a.job_id)
    if job is None:
        sys.exit(f"no job {a.job_id}")
    print(json.dumps(job, indent=2, default=str))
    conn.close()


def cmd_cancel(a):
    conn = jobstore.connect(a.db)
    ok = jobstore.cancel(conn, a.job_id)
    print("cancelled" if ok else
          "not cancelled (already running or finished)")
    conn.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Inspect and drive the job queue.")
    p.add_argument("--db", default=None, help="path to the job database")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("computations", help="list runnable computations"
                   ).set_defaults(fn=cmd_computations)

    s = sub.add_parser("schema", help="show a computation's parameters")
    s.add_argument("computation")
    s.set_defaults(fn=cmd_schema)

    s = sub.add_parser("submit", help="queue a computation")
    s.add_argument("computation")
    s.add_argument("params", help="JSON file, inline JSON, or - for stdin")
    s.add_argument("--by", default=None, help="who submitted it")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("list", help="list jobs")
    s.add_argument("--status", default=None)
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("show", help="show one job in full")
    s.add_argument("job_id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("cancel", help="cancel a queued job")
    s.add_argument("job_id")
    s.set_defaults(fn=cmd_cancel)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
