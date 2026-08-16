"""Polling worker: claims queued jobs and runs them as child processes.

Each job runs in its own process via the existing subprocess_launcher, which
is what the desktop GUI already does -- the worker just drives it from the job
table instead of a button. That keeps one execution path for both front ends.

Concurrency defaults to 4. The solver is single-threaded (measured at 0.99
cores busy) and four concurrent jobs were measured at 3.90x throughput with
only ~3.4% per-process slowdown, so four is close to ideal on a 12-core
machine while leaving headroom for an API process and the OS.

Run it directly:

    python server/worker.py                 # 4 slots, poll every 5s
    python server/worker.py --slots 2 --poll 10
    python server/worker.py --once          # drain the queue, then exit
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG = os.path.join(_ROOT, "project_src_package_2025")
for _p in (_ROOT, _PKG, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jobstore  # noqa: E402
from multiprocessing_tools import compute_worker, subprocess_launcher  # noqa: E402


def _log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _job_output_root(job_id):
    """Per-job output tree, passed to the child as ITCM_OUTPUT_ROOT.

    Several kernels write files and return None (collect_char_time_mass among
    them), so a job's return value does not say what it produced. Giving each
    job its own output root makes that unambiguous: everything under this
    directory belongs to this job.

    Diffing a shared tree before/after was tried first and is wrong under
    concurrency -- overlapping jobs observe each other's directories, so the
    same output gets attributed to several jobs at once.
    """
    from system_configuration import file_paths as fp

    # fp.general_output already reflects any ITCM_OUTPUT_ROOT set for the
    # worker itself, so nesting stays consistent if the whole tree is moved.
    default_root = os.path.join(str(fp.general_output), "jobs")
    return os.path.join(default_root, job_id)


def _produced_dirs(root):
    """Directories holding this job's actual results.

    Skips two kinds of noise so the recorded list is the scientific output and
    nothing else:
      * ``json_output`` -- the control-plane result file, not a result.
      * directories whose only entries are dotfiles; the output helper seeds
        each parent with a .gitkeep, which would otherwise look like content.
    """
    out = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) == "json_output":
            continue
        if any(not f.startswith(".") for f in filenames):
            out.append(dirpath)
    return sorted(out)


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class _Running:
    """A job in flight: its row, child process, and output root."""

    def __init__(self, job, proc, out_root):
        self.job = job
        self.proc = proc
        self.out_root = out_root
        self.started = time.time()


def _launch(job):
    """Start a claimed job in a child process keyed to its own job id."""
    args = subprocess_launcher.build_worker_args(
        job["computation"], json.dumps(job["params"]), job["id"]
    )
    out_root = _job_output_root(job["id"])
    os.makedirs(out_root, exist_ok=True)

    # Agg keeps a headless child from trying to open a GUI backend; the child
    # inherits it unless the operator has chosen otherwise.
    extra_env = {"ITCM_OUTPUT_ROOT": out_root}
    extra_env.setdefault("MPLBACKEND", os.environ.get("MPLBACKEND", "Agg"))

    _log(f"start   {job['id'][:8]} {job['computation']}")
    proc = subprocess_launcher.launch_subprocess(args, extra_env=extra_env)
    return proc, out_root


def _collect(conn, run):
    """Turn a finished child process into a terminal job row."""
    job_id = run.job["id"]
    short = job_id[:8]
    elapsed = time.time() - run.started
    # Resolved against the job's own output root: the child wrote it there,
    # not under this worker's default tree.
    result_file = compute_worker.result_path(job_id, output_root=run.out_root)

    produced = _produced_dirs(run.out_root)

    payload = None
    if os.path.exists(result_file):
        try:
            with open(result_file) as f:
                payload = json.load(f)
            os.remove(result_file)
        except (ValueError, OSError) as e:
            payload = {"status": "error",
                       "message": f"unreadable result file: {e}"}

    if payload is None:
        # The child exited without writing a result at all -- a hard crash
        # (segfault, OOM kill, power loss). Distinguish it from a computation
        # that ran and raised, which does write a result with status=error.
        jobstore.finish(
            conn, job_id, jobstore.STATUS_FAILED,
            error=f"worker exited with code {run.proc.returncode} "
                  f"without writing a result",
            output_dirs=produced,
        )
        _log(f"FAILED  {short} no result file (exit {run.proc.returncode}) "
             f"after {elapsed/60:.1f} min")
        return

    if payload.get("status") == "ok":
        jobstore.finish(conn, job_id, jobstore.STATUS_SUCCEEDED,
                        result=payload.get("result"), output_dirs=produced)
        _log(f"done    {short} in {elapsed/60:.1f} min"
             + (f", {len(produced)} output dir(s)" if produced else ""))
    else:
        jobstore.finish(conn, job_id, jobstore.STATUS_FAILED,
                        error=payload.get("message"), output_dirs=produced)
        _log(f"FAILED  {short} {payload.get('message')!r} "
             f"after {elapsed/60:.1f} min")


def run(slots=4, poll=5.0, once=False, db_path=None):
    conn = jobstore.connect(db_path)

    requeued = jobstore.requeue_stale_running(conn)
    if requeued:
        _log(f"requeued {requeued} job(s) left running by a previous worker")

    _log(f"worker up: {slots} slot(s), polling every {poll:g}s")
    _log(f"db: {db_path or jobstore.default_db_path()}")

    running = []
    try:
        while True:
            # Reap finished children first so their slots free up immediately.
            for r in [r for r in running if r.proc.poll() is not None]:
                _collect(conn, r)
                running.remove(r)

            while len(running) < slots:
                job = jobstore.claim_next(conn)
                if job is None:
                    break
                try:
                    proc, out_root = _launch(job)
                except Exception as e:
                    jobstore.finish(conn, job["id"], jobstore.STATUS_FAILED,
                                    error=f"failed to launch: {e}")
                    _log(f"FAILED  {job['id'][:8]} launch error: {e}")
                    continue
                running.append(_Running(job, proc, out_root))

            if once and not running:
                queued = jobstore.counts_by_status(conn).get(
                    jobstore.STATUS_QUEUED, 0)
                if queued == 0:
                    _log("queue drained; exiting (--once)")
                    break

            time.sleep(poll)
    except KeyboardInterrupt:
        _log(f"interrupted; {len(running)} job(s) still running as child "
             f"processes and will be requeued at next startup")
    finally:
        conn.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Run queued computations.")
    p.add_argument("--slots", type=int, default=4,
                   help="max concurrent jobs (default 4)")
    p.add_argument("--poll", type=float, default=5.0,
                   help="seconds between queue polls (default 5)")
    p.add_argument("--once", action="store_true",
                   help="exit once the queue is empty instead of waiting")
    p.add_argument("--db", default=None, help="path to the job database")
    a = p.parse_args(argv)
    run(slots=a.slots, poll=a.poll, once=a.once, db_path=a.db)


if __name__ == "__main__":
    main()
