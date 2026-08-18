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
| Grids | 48², 64², 80², 96², 112² |

## Contents (after the run completes)

| File | |
|---|---|
| `jobs.json` | manifest: job id and estimated cost per grid |
| `grid_study_results.csv` | numerical results — `grid, v, t_star, m_star, wall_hours, job_id` |
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

## Caveats

**No checkpointing.** A reboot or power cut loses the run; nothing resumes. The
112² job is the exposure — 8 hours with no intermediate state.

**`m*` can come back empty.** It is only defined when the fitted `t* < 0.1`; the
solver records `NaN` with a warning rather than failing, so `t*` survives and the
run is not wasted. Earlier N=16 runs gave `t* = 0.052` (48²) and `t* = 0.060`
(96²), both comfortably inside the window, and `t*` drifts upward with
refinement — so 112² is the one to check if a value is missing.
