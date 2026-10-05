# Second derivative of ln M(t) over the (a, b) grid — 2026-10-05

Exploratory study for choosing a characteristic-time criterion. It covers this
folder and three siblings:

| folder | grid | T | (a, b) |
|---|---|---|---|
| `2026-10-05_ab-grid-16x16_second-derivative/` (this) | 16×16 | 1 | all 16 pairs of {0.1, 1, 10, 100}² |
| `2026-10-05_ab-grid-32x32_second-derivative/` | 32×32 | 1 | all 16 pairs |
| `2026-10-05_ab-pairs-16x16_T2/` | 16×16 | 2 | (10,0.1) (10,1) (1,0.1) (1,1) (0.1,0.1) (0.1,1) |
| `2026-10-05_ab-pairs-16x16_T4/` | 16×16 | 4 | same six pairs |

Common setup, matching job `0b8d4022` (the 96×96 grid whose b ≤ 1 rows came
back NaN): v = 1, four evenly spaced tubes (`N_LIST = [0, G/4, G/2, 3G/4]`),
centred initial condition, d_tube = 0, `MA_collection_factor = 5`, CPU solver.
M is the total mass, y = ln M (natural log).

## What is plotted

- **ln M(t)**, with vertical lines:
  - **red** — first root of y″: the first genuine sign change of the second
    difference (|y″| > 1e-4 on both sides; below that, rounding in M is
    visible), linearly interpolated.
  - **blue** — t\* by the committed criterion (`launch._char_time_onset`,
    τ = 1e-3): κ = |y″|/y′² drops below τ and stays there to the end of the
    run, with at least 0.1 of run after it. Shown where it exists.
- **d²(ln M)/dt²**, second-order central difference
  (y[i+s] − 2y[i] + y[i−s]) / H², with H ≈ 1e-4 (H = 1.5e-4 at 16×16,
  1.04e-4 at 32×32), on linear axes; `_zoom` copies use y ∈ [−1, max] with a
  5% margin.
- **`chebyshev_full_interval/`** (16×16, T = 1 only): `numpy.polynomial.Chebyshev`
  least-squares fit of ln M on [0, 1], degree 120–160, residual ≈ 1e-14 to
  1e-13 for t ≥ 0.02 and up to 8e-13 on [0, 0.02]; its f″ overlaid on the
  central difference, plus residual and coefficient-decay plots.

`curves_*.csv` hold t, ln M and the second difference for every pair (32×32 is
stored at the differencing spacing); `first_root_t_star_*.csv` /
`t_star_16x16_T*.csv` hold both t\* values per pair.

## Findings

- The Chebyshev f″ and the central difference agree to a median of ~1e-7;
  first roots agree to ~1e-6. The derivative values are not a numerical
  artefact.
- **b = 100:** y″ approaches 0 from below and never changes sign → no first
  root; κ settles cleanly (t\* ≈ 0.37–0.51).
- **b ≤ 1:** y″ oscillates about 0 with shrinking amplitude (a damped wobble
  in the decay rate). κ does not settle by T = 1 or 2 (NaN); at T = 4 it does,
  t\* = 2.2–3.6. First roots are unchanged by T (0.08–0.27).
- **b = 10:** both definitions give a value but disagree (e.g. a = 1:
  0.26 vs 0.77).
- 32×32 first roots are 9–16% later than 16×16; κ t\* within a few percent.
- The pre-2026-09-28 definition (intercept of the line through log₁₀ M at
  t = 0.4, 0.5) gives 0.07–0.09 here, and negative values for a ≥ 10, b ≤ 1;
  it measures something different and does not match either.

## Reproducing

`scripts/` holds the generating scripts (paths are absolute to the original
machine; run with the project's Anaconda Python):

- `grid_plots.py <G>` — solve the 16-pair grid at G×G, T = 1; writes the
  curves, roots and the ln M / second-derivative grids.
- `zoom_d2.py <G>` — the `[-1, max]` zoomed derivative grid from those CSVs.
- `cheb_16x16.py` — Chebyshev fits and overlays for 16×16.
- `pairs_T.py <T>` — the six-pair 16×16 runs at T = 2 or 4.
