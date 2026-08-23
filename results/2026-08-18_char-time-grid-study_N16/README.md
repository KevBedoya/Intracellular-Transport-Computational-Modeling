# Characteristic-time grid study — N = 16

Grid-convergence study of the characteristic time at a fixed microtubule count.
Everything except the grid is held constant, so the only varying quantity is
spatial resolution.

## Parameters

| | |
|---|---|
| Microtubules | N = 16, evenly spaced (ray indices regenerated per grid) |
| Advective velocity | v = 10⁴ |
| Switch rates | a = b = w = 100 |
| Solution time | T = 1 |
| Domain radius | R = 1 |
| Diffusion coefficient | D = 1 |
| Extraction width | d_tube = 0 |
| Initial condition | centred |
| Grids | 48², 64², 80², 96², 112², 128², 144², 160² |

## Contents (after the run completes)

| File | |
|---|---|
| `jobs.json` | manifest: job id and estimated cost per grid |
| `grid_study_results.csv` | numerical results — `grid, v, t_star, m_star, wall_hours, job_id, provenance` |
| `recovered_rows.json` | values for grids whose result file no longer exists on disk |
| `mass_vs_grid_size.png` | m* against grid size |
| `grid_study_report.tex` | parameters, wall time per grid, and the figure |
| `grid_study_report.pdf` | the same report as a PDF |

## Reproducing

```bash
python scripts/grid_study_char_time.py submit    # dispatch
python scripts/grid_study_char_time.py status    # progress
python scripts/grid_study_char_time.py report    # plot + .tex + .pdf
```

Jobs are submitted heaviest grid first. The worker runs 4 concurrently and
claims the oldest queued job, so ordering this way lets 112² — which dominates
the wall time — start immediately instead of queueing behind the cheap grids.

## Expected cost

| Grid | Timesteps (T=1) | Est. wall | Timeseries memory |
|---|---|---|---|
| 48² | 2,689,274 | ~4 min | 22 MB |
| 64² | 8,499,436 | ~17 min | 68 MB |
| 80² | 20,750,578 | ~1.1 hr | 166 MB |
| 96² | 43,028,399 | ~3.3 hr | 344 MB |
| 112² | 79,715,422 | ~8.3 hr | 638 MB |

Run concurrently, so **wall time ≈ 8.3 hr**, set by 112² alone; sequentially it
would be ~13 hr. Peak memory ~1.2 GB.

Estimates scale the measured 96² figure by `G⁶`, which follows from the
scheme's stability limit `dT ∝ 1/(M²N²)`: `K ∝ G⁴` timesteps over `G²` patches.

## Results

| Grid | t* | m* | Wall |
|---|---|---|---|
| 48² | 0.052357 | 0.670035 | 3.8 min |
| 64² | 0.055390 | 0.546092 | 19.3 min |
| 80² | 0.057689 | 0.450505 | 1.23 hr |
| 96² | 0.059530 | 0.377297 | 3.68 hr |
| 112² | 0.061058 | 0.320804 | 9.32 hr |
| 128² | 0.062355 | 0.276652 | 21.31 hr |
| 144² | 0.063476 | 0.241655 | 43.68 hr |
| 160² | 0.064458 | 0.213518 | 82.36 hr |

`t*` rises monotonically with refinement and `m*` falls monotonically, with the
local exponent of `m*` steepening from -0.71 (48->64) to -1.14 (144->160).

## Provenance of the stored values

Output files live on disk under each job's own output root and are not archived
anywhere else, so the API can report a job as `succeeded` while its CSV is gone.
That happened to this study: no job's original CSV survives except 160x160's.
The `provenance` column records how each row was obtained.

* `csv` — read from the job's result CSV, or carried forward from the value
  published in this file when that CSV was still present.
* `reconstructed-from-log` — 128x128 and 144x144. Their solves succeeded and
  their CSVs were later removed. The values were rebuilt from the solver's
  progress log, which records diffusive and advective mass every 10^6 steps, by
  redoing the same log-linear fit. `scripts/recover_char_time_from_log.py`
  performs the reconstruction and validates it against every grid whose value is
  independently known, agreeing to within 0.006% on six of them and reproducing
  160x160 to eight decimal places. It refuses to emit numbers if that check
  fails.

Regenerating the report is non-destructive: `report` prefers a live CSV, then a
recovered value, then the value already published here, so a rerun cannot shrink
the table.

## Notes

**160x160 is recorded as `failed`.** Its solve ran all 332,009,254 steps and
wrote its CSV and plot; only the result-JSON write crashed, on a missing parent
directory. Fixed in `compute_worker.compute_and_send`, which now creates the
directory immediately before writing and falls back to the shared tree rather
than raising.

**Checkpointing exists but was not active for these runs.** The worker process
predates it, and restarting the worker requeues running jobs. Runs from 128x128
onward were therefore unprotected; 160x160 came within one write of losing 82
hours.
