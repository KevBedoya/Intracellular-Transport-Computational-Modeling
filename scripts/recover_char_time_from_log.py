"""Reconstruct characteristic-time results from the worker log.

Three grids in the N=16 study lost their result CSVs:

  * 128x128 (job ba8606ef) and 144x144 (job bd42ce92) succeeded, but their
    output directories were removed from disk afterwards. The job store keeps
    ``result = None`` for this computation because ``collect_char_time_mass``
    writes files and returns nothing, so the values were not recoverable from
    the database either.
  * 160x160 (job 1583ec96) completed all 332,009,254 steps and wrote its CSV,
    then crashed writing its result JSON, so the job is recorded as failed. Its
    CSV does survive, and is preferred over reconstruction where present.

What does survive for all three is the solver's own progress log, which prints
the diffusive and advective mass every ``mass_checkpoint`` steps. Total mass is
their sum, so the characteristic-time fit can be redone from it:

    t* from a log-linear fit of total mass between t = 0.4 and t = 0.5
    m* = total mass at t = 10 t*

matching launch.collect_char_time_mass. The log samples every 1e6 steps rather
than every ``MA_collection_factor`` steps, so values at 0.4, 0.5 and 10 t* come
from interpolating log10(total mass) linearly in t -- exact for a pure
exponential and very close for this nearly-exponential decay.

**The method is validated, not assumed.** Every grid whose CSV still exists is
reconstructed too and compared against it. Agreement is currently within 0.006%
on all six, and reproduces 160x160 to eight decimals. If a future run disagrees
by more than TOLERANCE_PCT the script fails rather than emitting a number.

    python scripts/recover_char_time_from_log.py            # report + write
    python scripts/recover_char_time_from_log.py --check    # validate only
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "2026-08-18_char-time-grid-study_N16")
RECOVERED = os.path.join(OUT_DIR, "recovered_rows.json")

sys.path.insert(0, os.path.join(ROOT, "src", "intracellular_transport"))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Fit window, fixed in collect_char_time_mass.
X1, X2 = 0.4, 0.5

# Largest disagreement tolerated against a surviving CSV, in percent. Set well
# above the observed 0.006% but far below anything that would matter
# scientifically -- the point is to catch a broken parse, not to police
# round-off.
TOLERANCE_PCT = 0.05

GRIDS = (48, 64, 80, 96, 112, 128, 144, 160, 176)

# Grids to recover, with the job they came from and that job's wall time. Wall
# hours are read from the job store rather than the log, which carries no
# timestamps.
TARGETS = {
    128: {"job_id": "ba8606eff3244902b828fdaab127c011", "status": "succeeded"},
    144: {"job_id": "bd42ce92df6b4ae181b83ff2a104b2a2", "status": "succeeded"},
    160: {"job_id": "1583ec968bbf4361845a31791fb9ae58", "status": "failed"},
}

LINE = re.compile(
    r"timestep:\s+(\d+)\s+"
    r"Current simulation time:\s+([\d.eE+-]+)\s+"
    r"Current DL mass:\s+([\d.eE+-]+)\s+"
    r"Current AL mass:\s+([\d.eE+-]+)")


def newest_worker_log():
    logs = sorted(glob.glob(os.path.join(ROOT, "server", "logs", "worker-*.log")))
    if not logs:
        sys.exit("no worker log found under server/logs/")
    return logs[-1]


def step_counts():
    from computational_tools import numerical_tools as num
    return {g: num.compute_K(g, g, 1.0) for g in GRIDS}


def parse_log(path, Ks):
    """Group progress samples by grid.

    The log interleaves concurrent jobs with nothing identifying which line
    belongs to which, so each sample is attributed by its implied step count,
    ``timestep / simulation_time``, which equals that grid's K(T=1). The grids
    differ by far more than the 2% window used here, so attribution is
    unambiguous.
    """
    series = {}
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        step, t = int(m.group(1)), float(m.group(2))
        if t <= 0:
            continue
        implied = step / t
        g = min(Ks, key=lambda G: abs(Ks[G] - implied))
        if abs(Ks[g] - implied) / Ks[g] > 0.02:
            continue
        total = float(m.group(3)) + float(m.group(4))
        series.setdefault(g, []).append((t, total))
    for g in series:
        series[g].sort()
    return series


def interp_log10(points, x):
    """Total mass at time ``x``, interpolating log10(mass) linearly in t."""
    for i in range(len(points) - 1):
        t0, m0 = points[i]
        t1, m1 = points[i + 1]
        if t0 <= x <= t1:
            if m0 <= 0 or m1 <= 0:
                return None
            if t1 == t0:
                return m0
            w = (x - t0) / (t1 - t0)
            return 10 ** (math.log10(m0) * (1 - w) + math.log10(m1) * w)
    return None


def reconstruct(points):
    """(t_star, m_star) from a [(t, total_mass)] series, or (None, None)."""
    y_a = interp_log10(points, X1)
    y_b = interp_log10(points, X2)
    if not y_a or not y_b:
        return None, None
    y1, y2 = math.log10(y_a), math.log10(y_b)
    slope = (y2 - y1) / (X2 - X1)
    if slope == 0:
        return None, None
    intercept = y2 - slope * X2
    t_star = -intercept / slope
    if not (0 < t_star < 1):
        return t_star, None
    return t_star, interp_log10(points, 10 * t_star)


def known_values():
    """Published values for grids whose CSV survives, for validation."""
    known = {}
    csv_path = os.path.join(OUT_DIR, "grid_study_results.csv")
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            try:
                known[int(r["grid"])] = (float(r["t_star"]), float(r["m_star"]))
            except (ValueError, KeyError):
                continue
    # The 160x160 CSV survives on disk even though its job is marked failed.
    for g, meta in TARGETS.items():
        d = os.path.join(ROOT, "data_output", "jobs", meta["job_id"])
        for p in glob.glob(os.path.join(d, "char_time_analysis", "*",
                                        "char_t_analysis_data.csv")):
            for r in csv.DictReader(open(p)):
                try:
                    known[g] = (float(r["t_star"]), float(r["m_star"]))
                    meta["source_csv"] = os.path.relpath(p, ROOT)
                except (ValueError, KeyError):
                    pass
    return known


def wall_hours(job_id):
    """Elapsed hours from the job store, or None."""
    import sqlite3
    from datetime import datetime
    db = os.path.join(ROOT, "data_output", "jobs.sqlite3")
    if not os.path.exists(db):
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT started_at, finished_at FROM jobs WHERE id = ?",
            (job_id,)).fetchone()
    except sqlite3.Error:
        return None
    if not row or not row[0] or not row[1]:
        return None
    try:
        return (datetime.fromisoformat(row[1])
                - datetime.fromisoformat(row[0])).total_seconds() / 3600.0
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="worker log (default: newest)")
    ap.add_argument("--check", action="store_true",
                    help="validate only; do not write recovered_rows.json")
    a = ap.parse_args()

    log = a.log or newest_worker_log()
    Ks = step_counts()
    series = parse_log(log, Ks)
    known = known_values()

    print(f"log: {os.path.relpath(log, ROOT)}")
    print(f"grids with progress samples: "
          f"{', '.join(str(g) for g in sorted(series))}\n")

    print(f"{'grid':>5} {'samples':>8} {'recon t*':>13} {'recon m*':>13} "
          f"{'vs CSV t*':>11} {'vs CSV m*':>11}")
    failures, recovered = [], {}
    for g in sorted(series):
        pts = series[g]
        t_star, m_star = reconstruct(pts)
        if t_star is None:
            print(f"{g:>5} {len(pts):>8} {'--':>13} {'--':>13} "
                  f"{'(t=0.5 not reached)':>25}")
            continue
        cells = [f"{g:>5}", f"{len(pts):>8}", f"{t_star:>13.9f}",
                 f"{m_star:>13.9f}" if m_star else f"{'nan':>13}"]
        if g in known:
            kt, km = known[g]
            et = abs(t_star - kt) / kt * 100
            em = abs(m_star - km) / km * 100 if m_star else float("nan")
            cells += [f"{et:>10.4f}%", f"{em:>10.4f}%"]
            if et > TOLERANCE_PCT or (m_star and em > TOLERANCE_PCT):
                failures.append(f"{g}x{g}: t* off by {et:.4f}%, "
                                f"m* off by {em:.4f}% (limit {TOLERANCE_PCT}%)")
        else:
            cells += [f"{'--':>11}", f"{'--':>11}"]
        print(" ".join(cells))

        if g in TARGETS:
            meta = TARGETS[g]
            # Prefer the surviving CSV where there is one; reconstruction is the
            # fallback, and the row records which was used.
            if g in known and "source_csv" in meta:
                t_out, m_out, prov = known[g][0], known[g][1], "csv"
            else:
                t_out, m_out, prov = t_star, m_star, "reconstructed-from-log"
            recovered[str(g)] = {
                "grid": g, "v": 10000.0,
                "t_star": t_out, "m_star": m_out,
                "wall_hours": wall_hours(meta["job_id"]),
                "job_id": meta["job_id"],
                "job_status": meta["status"],
                "provenance": prov,
                "source": meta.get("source_csv",
                                   os.path.relpath(log, ROOT).replace("\\", "/")),
            }

    if failures:
        print("\nVALIDATION FAILED -- not writing recovered values:")
        for f in failures:
            print("  " + f)
        sys.exit(1)

    n_val = sum(1 for g in series if g in known)
    print(f"\nvalidated against {n_val} surviving CSV(s), all within "
          f"{TOLERANCE_PCT}%")

    # The fit slope, which older runs did not record.
    #
    # collect_char_time_mass computed the slope of log10(total mass) across the
    # window, used it for t*, and dropped it. It is recoverable wherever the mass
    # log survives -- either directly, as (y2 - y1) / (x2 - x1), or from the
    # recorded t* as y2 / (x2 - t*), which is the same number and serves as a
    # cross-check. Runs that went through the GPU path log nothing per step, so
    # for those it is not recoverable at all; they are written as blank rather
    # than estimated.
    slope_rows = []
    for g in sorted(series):
        y_a = interp_log10(series[g], X1)
        y_b = interp_log10(series[g], X2)
        if not y_a or not y_b:
            continue
        y1, y2 = math.log10(y_a), math.log10(y_b)
        slope = (y2 - y1) / (X2 - X1)
        row = {"grid": g, "fit_slope": slope,
               "total_mass_at_t1": y_a, "total_mass_at_t2": y_b,
               "cross_check_via_t_star": "", "agreement_pct": ""}
        if g in known:
            t_star = known[g][0]
            if X2 - t_star != 0:
                via = y2 / (X2 - t_star)
                row["cross_check_via_t_star"] = via
                row["agreement_pct"] = abs(slope - via) / abs(slope) * 100
        slope_rows.append(row)

    if a.check:
        print("--check given; nothing written")
        return

    if slope_rows:
        path = os.path.join(OUT_DIR, "fit_slopes.csv")
        with open(path, "w", newline="\n") as f:
            w = csv.DictWriter(f, fieldnames=list(slope_rows[0]))
            w.writeheader()
            for r in slope_rows:
                w.writerow(r)
        print(f"wrote {os.path.relpath(path, ROOT)} ({len(slope_rows)} grid(s))")
    with open(RECOVERED, "w", newline="\n") as f:
        json.dump(recovered, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(RECOVERED, ROOT)} "
          f"({len(recovered)} row(s))")


if __name__ == "__main__":
    main()
