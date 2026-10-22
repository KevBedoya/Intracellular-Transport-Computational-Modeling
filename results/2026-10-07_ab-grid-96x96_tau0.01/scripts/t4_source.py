"""Which total-mass series each (a, b) pair of the 96x96 study uses.

Job 0b8d4022 solved all sixteen pairs to T = 1. Job 54cdf1f3 re-solves the
eight b <= 1 pairs to T = 4; a pair counts as complete once its final state
has been saved (final_state.json) by the current run, and from then on the
T = 4 series replaces the T = 1 one. The job restarted from the first pair on
2026-10-17 (API reset), so a final_state.json saved before that is left over
from the earlier run while the solver may be rewriting that pair's series in
place -- only states saved after RUN_STARTED count.
Everything else about the series is identical: same grid,
rates, velocity, tubes and sampling (sample i at t = i * 5 * dT).
"""
import json
import os

import numpy as np

R = "N:/work/Intracellular-Transport-Computational-Modeling"
T4_JOB = "54cdf1f33b454147b0e56700860c2d10"
T1_JOB = "0b8d4022"
T4_DIR = os.path.join(R, "data_output", "jobs", T4_JOB, "checkpoints")
RUN_STARTED = "2026-10-17T22:33:56+00:00"  # the job's started_at


def load_series(a, b, t1_dir):
    """(M, T, source label) for pair (a, b)."""
    d = os.path.join(T4_DIR, f"ab_grid_char_time_a{a:g}_b{b:g}")
    meta_path = os.path.join(d, "final_state.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    if os.path.exists(meta_path) and meta["saved_utc"] >= RUN_STARTED:
        rk = int(meta["relative_k"])
        m = np.array(np.memmap(os.path.join(d, "timeseries_4.dat"), dtype=np.float64,
                               mode="r", shape=(rk,)))
        return m, float(meta["T_requested"]), f"job {T4_JOB[:8]}, T = {meta['T_requested']:g}"
    m = np.fromfile(os.path.join(t1_dir, f"tm_a{a:g}_b{b:g}.dat"), dtype=np.float64)
    return m, 1.0, f"job {T1_JOB}, T = 1"
