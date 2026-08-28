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
| Grids | 48² through 224² in steps of 16 (12 grids) |

## Contents (after the run completes)

| File | |
|---|---|
| `jobs.json` | manifest: job id and estimated cost per grid |
| `grid_study_results.csv` | numerical results — `grid, v, t_star, m_star, wall_hours, device, job_id, provenance` |
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

| Grid | t* | m* | Device | Wall |
|---|---|---|---|---|
| 48² | 0.052357 | 0.670035 | CPU | 3.8 min |
| 64² | 0.055390 | 0.546092 | CPU | 19.3 min |
| 80² | 0.057689 | 0.450505 | CPU | 1.23 hr |
| 96² | 0.059530 | 0.377297 | CPU | 3.68 hr |
| 112² | 0.061058 | 0.320804 | CPU | 9.32 hr |
| 128² | 0.062355 | 0.276652 | CPU | 21.31 hr |
| 144² | 0.063476 | 0.241655 | CPU | 43.68 hr |
| 160² | 0.064458 | 0.213518 | CPU | 82.36 hr |
| 176² | 0.065327 | 0.190594 | GPU | 22.48 hr |
| 192² | 0.066104 | 0.171680 | GPU | 30.32 hr |
| 208² | 0.066804 | 0.155899 | GPU | 35.26 hr |
| 224² | 0.067439 | 0.142591 | GPU | 54.01 hr |

**Wall times are not comparable across the device boundary.** The four GPU jobs
ran concurrently on one card, which time-slices between them, so each took
roughly three times its solo cost. Solo, 224² is about 27 hr on the GPU against
an extrapolated 24 days on the CPU.

## m* does not converge under refinement

Worth stating plainly, because the raw ratios invite the opposite reading. The
successive ratios `m*(G+16)/m*(G)` do rise toward 1 — 0.815, 0.825, ..., 0.893,
0.901, 0.908, 0.915 — but that is mostly because the *grid* ratio does, falling
from 1.333 to 1.077 as the fixed +16 step lands on a larger base.

Normalising for that, `m* ~ G^-p`, and `p` rises and then settles:

| G | 64 | 96 | 128 | 160 | 176 | 192 | 208 | 224 |
|---|---|---|---|---|---|---|---|---|
| p | 0.711 | 0.973 | 1.109 | 1.175 | 1.192 | 1.201 | 1.205 | 1.204 |

`p` has plateaued at **≈ 1.204** — flat over the last three refinements and
slightly turned over, with Aitken extrapolation of the sequence giving 1.2041.
A pure power law fits the tail to an rms residual of 4e-4 in log space.

So `m*` is **not** tending to a nonzero grid-independent value; it decays as
roughly `G^-1.2` and tends to zero. Fitting `m* = m_inf + A·G^-q` returns a
negative `m_inf` on every window, tightening toward zero as the window narrows,
which is what a sequence heading to zero looks like under that model.

Note that `p ≈ 1.204`, not 1. A plausible mechanism for a vanishing `m*` is that
the microtubules occupy single rays (`d_tube = 0`), so their angular extent
`dθ = 2π/ry` shrinks as the grid refines — but that alone would give `p → 1`.
The measured exponent is meaningfully above 1, so if the tube width is the cause
it is not the whole of it. Testing it needs a sweep at fixed *physical* `d_tube`
rather than fixed cell count; that is cheap at 48²–96² and has not been run.

`t*` behaves differently: it rises monotonically and its increments shrink
steadily (0.00303, 0.00230, 0.00184, ..., 0.00070, 0.00063), so unlike `m*` it is
plausibly heading somewhere finite. But it is **not** converged here. Aitken
extrapolation of the `t*` sequence still drifts upward with each grid added —
0.07267 through 192², 0.07311 through 208², 0.07360 through 224² — so the limit
is somewhere near 0.073-0.074 rather than the 0.0674 measured at 224², and the
sequence has not settled enough to pin it down. Extending the study would tighten
`t*`; it will not rescue `m*`.

## The fit slope m does not converge

`m` is the slope of `log10(total mass)` across the fit window `t in [0.4, 0.5]`,
the quantity `t*` is derived from. It is in `fit_slopes.csv` for all twelve
grids, and in the report table.

| G | 48 | 64 | 80 | 96 | 112 | 128 | 144 | 160 | 176 | 192 | 208 | 224 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m | -0.369 | -0.527 | -0.667 | -0.790 | -0.899 | -0.994 | -1.080 | -1.156 | -1.224 | -1.286 | -1.342 | -1.394 |

Unlike the `m*` exponent, this one is **not** settling. The successive
differences shrink, but by a ratio that is itself *rising* toward 1 -- 0.886,
0.880, 0.881, 0.884, 0.889, 0.894, 0.899, 0.903, 0.908, 0.912 -- so each
refinement buys proportionally less than the last. Aitken's estimate of the
limit recedes with every grid added: -1.833 through 176 squared, then -1.865,
-1.895, and -1.922 through 224 squared.

So at 224 squared, `m = -1.394` against an apparent limit somewhere past -1.9
that has not stopped moving. Extending the grid further will not close that gap
at any tractable size.

Three quantities, three behaviours, worth keeping straight:

* the `m*` exponent `p` **converged**, plateauing at 1.204;
* `t*` is **plausibly convergent but unresolved**, with Aitken drifting up
  through 0.0736 and no plateau yet;
* `m` is **not usefully convergent** on this range at all.

### Recovering m for older runs

`collect_char_time_mass` did not record the slope until 27 August 2026; it was a
local used for `t*` and discarded. For the nine CPU grids it was rebuilt from the
worker log, which preserves total mass every 1e6 steps as a side effect of
progress printing, and cross-checked two independent ways (directly, and as
`y2 / (x2 - t*)`) agreeing to within 0.0007%.

The GPU path prints nothing per step, so for 192, 208 and 224 squared no such
fallback existed and the slope was unrecoverable from stored artifacts. Those
three were re-solved at `T = 0.52` -- the least that still reaches the fit window
-- which cost about half a full run and returned `t*` identical to the original
`T = 1` jobs, confirming the shortcut is exact. `fit_slopes.csv` records which
source each value came from.

## Provenance of the stored values

Output files live on disk under each job's own output root and are not archived
anywhere else, so the API can report a job as `succeeded` while its CSV is gone.
That happened to this study: no job's original CSV survives except 160x160's.
The `provenance` column records how each row was obtained.

* `csv` — read from the job's result CSV, or carried forward from the value
  published in this file when that CSV was still present. All four GPU rows
  (176²–224²) were read from live CSVs.
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
