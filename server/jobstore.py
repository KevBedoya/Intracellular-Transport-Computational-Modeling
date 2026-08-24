"""SQLite-backed job store for queued computations.

Holds *metadata only*: which computation, its parameters, status, timings and
where the outputs landed.  Bulk numerical output stays on disk -- a single
characteristic-time run allocates five 8.6M-element float64 timeseries
(~344 MB), which has no business inside a database row.  The store records
paths; the filesystem holds the arrays and figures.

SQLite is deliberate at this scale (a handful of users, at most a few
concurrent solver processes): no server to install or supervise, and the write
volume is a few rows per hour.  WAL mode lets readers (an API process, the
GUI) run while the worker writes.  The schema is plain enough to move to
Postgres later if concurrency ever outgrows it.

Status lifecycle:

    queued -> running -> succeeded
                      -> failed
    queued -> cancelled          (only before a worker claims it)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

TERMINAL_STATUSES = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    computation   TEXT NOT NULL,
    params        TEXT NOT NULL,          -- JSON object
    status        TEXT NOT NULL,
    submitted_at  TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    result        TEXT,                   -- JSON, on success
    error         TEXT,                   -- message, on failure
    output_dirs   TEXT,                   -- JSON list of paths produced
    git_sha       TEXT,
    submitted_by  TEXT,
    est_seconds   REAL                    -- solo wall-time estimate at submission
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, submitted_at);
"""

# Columns added after the table first shipped. SQLite has no "ADD COLUMN IF NOT
# EXISTS", and the CREATE TABLE above is a no-op on an existing database, so new
# columns have to be applied separately or every deployment would need its DB
# rebuilt -- losing the job history.
_MIGRATIONS = (
    ("est_seconds", "ALTER TABLE jobs ADD COLUMN est_seconds REAL"),
)


def _migrate(conn):
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    for column, ddl in _MIGRATIONS:
        if column not in have:
            conn.execute(ddl)


def default_db_path():
    """Location of the job database, alongside the other generated output."""
    from system_configuration import file_paths as fp

    return os.path.join(str(fp.general_output), "jobs.sqlite3")


def _utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path=None):
    """Open the store, creating it if needed."""
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # isolation_level=None -> explicit transaction control, needed for the
    # BEGIN IMMEDIATE in claim_next().
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    for key in ("params", "result", "output_dirs"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (TypeError, ValueError):
                pass
    return d


def submit(conn, computation, params, submitted_by=None, git_sha=None):
    """Queue a computation.  Returns the new job id.

    The computation name is validated here rather than in the worker, so a
    typo is rejected at submission instead of failing hours later when the job
    finally reaches the front of the queue.
    """
    from multiprocessing_tools.computation_router import (
        COMPUTATION_FUNCTIONS, validate_params)

    if computation not in COMPUTATION_FUNCTIONS:
        raise ValueError(
            f"unknown computation {computation!r}; expected one of: "
            + ", ".join(sorted(COMPUTATION_FUNCTIONS))
        )
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")

    # Reject a bad parameter set here rather than letting it reach a worker: an
    # invalid job would otherwise claim a slot and fail mid-solve, which on a
    # long queue can be hours after submission.
    problems = validate_params(computation, params)
    if problems:
        raise ValueError("invalid parameters: " + "; ".join(
            f"{k}: {v}" for k, v in sorted(problems.items())))

    # Cost the job once, at submission, and store it. Recomputing it later from
    # the params would give the same answer, but storing it means the number the
    # user was shown before launching is the same number the queue reports while
    # it runs -- and it survives a change to the cost model.
    try:
        import estimate
        est_seconds = estimate.seconds(computation, params)
    except Exception:
        est_seconds = None      # never block a submission over an estimate

    job_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO jobs (id, computation, params, status, submitted_at,"
        " git_sha, submitted_by, est_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, computation, json.dumps(params), STATUS_QUEUED, _utcnow(),
         git_sha, submitted_by, est_seconds),
    )
    return job_id


def claim_next(conn):
    """Atomically take the oldest queued job and mark it running.

    Returns the claimed job as a dict, or None if the queue is empty.

    BEGIN IMMEDIATE takes the write lock up front so two workers cannot select
    the same row before either has updated it.  Without it, concurrent workers
    would race and run the same job twice.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY submitted_at LIMIT 1",
            (STATUS_QUEUED,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        started = _utcnow()
        conn.execute(
            "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
            (STATUS_RUNNING, started, row["id"]),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    claimed = _row_to_dict(row)
    claimed["status"] = STATUS_RUNNING
    claimed["started_at"] = started
    return claimed


def finish(conn, job_id, status, result=None, error=None, output_dirs=None):
    """Record a terminal outcome for a job."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"{status!r} is not a terminal status")
    conn.execute(
        "UPDATE jobs SET status = ?, finished_at = ?, result = ?, error = ?,"
        " output_dirs = ? WHERE id = ?",
        (status, _utcnow(),
         json.dumps(result, default=str) if result is not None else None,
         error,
         json.dumps(output_dirs) if output_dirs is not None else None,
         job_id),
    )


def cancel(conn, job_id):
    """Cancel a job that has not started.  True if it was cancelled.

    Running jobs are left alone: they own a subprocess and possibly hours of
    partial output, so stopping them is the worker's business, not the store's.
    """
    job = get(conn, job_id)          # resolves an id prefix
    if job is None:
        return False
    cur = conn.execute(
        "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ? AND status = ?",
        (STATUS_CANCELLED, _utcnow(), job["id"], STATUS_QUEUED),
    )
    return cur.rowcount > 0


def get(conn, job_id):
    """Fetch one job by full id, or by an unambiguous id prefix.

    Listings abbreviate ids to 8 characters, so accepting a prefix means the
    id you can see is the id you can use. An ambiguous prefix is an error
    rather than a silent pick.
    """
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is not None:
        return _row_to_dict(row)

    rows = conn.execute(
        "SELECT * FROM jobs WHERE id LIKE ? LIMIT 2", (job_id + "%",)
    ).fetchall()
    if len(rows) > 1:
        raise ValueError(f"job id prefix {job_id!r} is ambiguous")
    return _row_to_dict(rows[0]) if rows else None


def list_jobs(conn, status=None, limit=100):
    if status:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY submitted_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def counts_by_status(conn):
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
    ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def requeue_stale_running(conn):
    """Return jobs stuck in 'running' to the queue.

    A worker killed mid-job (power loss, reboot) leaves its rows claimed
    forever, so nothing will ever pick them up again.  Call this once at
    worker startup, when no job can legitimately be running yet.  Returns the
    number of jobs requeued.
    """
    cur = conn.execute(
        "UPDATE jobs SET status = ?, started_at = NULL WHERE status = ?",
        (STATUS_QUEUED, STATUS_RUNNING),
    )
    return cur.rowcount
