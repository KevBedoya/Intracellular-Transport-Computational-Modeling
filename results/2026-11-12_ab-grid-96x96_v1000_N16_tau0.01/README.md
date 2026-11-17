# 96x96, v = 1000, N = 16, tau = 0.01: the (a, b) grid

This is the v = 1000 study (`../2026-11-08_ab-grid-96x96_v1000_tau0.01/`, N = 4) rerun with N = 16. Everything else is unchanged:

- grid 96x96
- v = 1000
- N = 16 evenly spaced (`N_LIST = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]`; 11-08: N = 4, `[0, 24, 48, 72]`)
- tau = 0.01
- (a, b) in {0.1, 1, 10, 100}^2
- T = 4 for every pair

Every result in the 11-08 directory has a counterpart here, with `v1000` replaced by `v1000_N16` in the file names. The sister run at N = 8 is `../2026-11-12_ab-grid-96x96_v1000_N8_tau0.01/`.

## How it was computed

The same way as 11-08: `scripts/power_engine.py` (copied unchanged) takes exact matrix powers of the solver's one-step operator on the symmetry-reduced state, x_k = P^k x_0. The modal engine still cannot be used at v = 1000: here cond(V) is 2.3e22 to 1.3e25 (`validation/v1_power_vs_modal.csv`, column `cond_V_v1000`). See the 11-08 README for the full method; in brief:

| quantity | how |
|---|---|
| M(t) | every H = 4305 dT = 1.0005e-4 over [0, 4] |
| reference y'' | Richardson extrapolation of the central differences of ln M at 10H and 20H |
| t* | the last crossing of \|y''\| = tau on the reference y'', interpolated linearly between samples |
| `central_difference_t_star` | the 10-07 time-stepper rule (`launch._char_time_onset`) on the samples at spacing H |
| m*, M(5 t*), M(10 t*), M(t~) | by binary powers of P. m* from a cubic spline of the samples agrees with powers to 4.7e-13 |
| lambda_1, lambda_2 | eigenvalues of (P_s − I) / dT, with the excitation test of `launch._slowest_rates` |
| fit_slope | least squares of log10 M over [t*, T] |
| PSD columns | `y2_spectrum` of ln M at H over [0, T] |

With 16 tubes the symmetry-reduced state has 477 entries (1332 at N = 4), so the grid ran faster than at N = 4: 5 s for all 16 pairs.

The eigenvalues are well conditioned: kappa is 31–75 for lambda_1 and 38–53 for lambda_2. −fit_slope · ln 10 reproduces lambda_1 to 5e-8 for every pair.

## Validation

`scripts/validate_power.py` writes both checks to `validation/`.

**1. v = 1 against the modal engine** (`validation/v1_power_vs_modal.csv`, all 16 pairs). There is no 11-05 study at N = 16, so the modal engine was run live on the same `N_LIST` at v = 1, where it is usable (cond(V) 7.8e4 to 7.9e7). Its t*, m* and fit_slope come from `launch._char_time_onset_modal` with the exact modal y''.

| quantity | largest difference |
|---|---|
| lambda_1, lambda_2, \|Im lambda_2\| | none: identical |
| M(t~), all three t~ (same t) | 2.5e-9 relative |
| fit_slope | 1.0e-7 relative |
| t* | 1.3e-6 absolute, at (0.1, 1); 8.9e-8 or less at the other 15 pairs |
| m*, M(5 t*), M(10 t*) | 7.0e-6, 3.5e-5, 7.0e-5 relative: the t* difference times lambda_1 (and 5, 10) |
| reference y'' − exact modal y'', t > 0.05 | 6.3e-6 (the modal round-off bound there is 4.3e-6) |

**2. v = 1000 against the time-stepper** (`validation/v1000_power_vs_time_stepper.csv`). The time-stepper (`launch._solve_mass_analysis`, CPU) ran over T = 0.05 for (0.1, 100), (100, 0.1) and (10, 10): 2.15 million steps each, about 418 s per pair. The layer masses agree to 1.8e-10 relative in DL, AL and total mass. That is under one machine epsilon per step over 2.15 million steps. At N = 4 the gap was 3.9e-10, and there it was traced to the time-stepper's own round-off growing linearly with the step count; it was not traced step by step here.

## Files

Every figure is a 4x4 grid in the 10-07 layout: rows are a, columns are b, T = 4. Red is t*, and dotted lines are ±tau.

| file | content |
|---|---|
| `d2lnM_central_difference_ylim0.1.png`, `_ylim0.01.png` | y'' by central difference at H, y-limits ±0.1 and ±0.01 |
| `d2lnM_reference_over_central_difference.png` | the reference y'' (green) over the central difference (wide blue), ±0.1 |
| `lnM.png`, `lnM_t_star.png` | ln M(t), and the same with t* marked |
| `psd_y2.png` | \|F[y'']\|² on linear axes, each panel zoomed on its peak |
| `summary_96x96_v1000_N16.csv` | per pair: t*, m*, M(t*) by powers, 10 t*, M(10 t*), the 10-07-rule t*, lambda_1, lambda_2, \|Im lambda_2\|, their condition numbers, fit_slope, the PSD peak and FWHM, max \|reference − central difference\| for t > 0.05, and M(T) |
| `t_star_m_5t_star_96x96_v1000_N16.pdf` | table of t*, 5 t* and M(5 t*) per (a, b) |
| `heatmap_m_5t_star_96x96_v1000_N16.png`, `.pdf` | M(5 t*) over (a, b), with 5 t* in each cell |
| `m_5t_star_96x96_v1000_N16.csv` | per pair: t*, 5 t*, M(5 t*) |
| `heatmap_m_t_tilde_5_over_re_lambda2_96x96_v1000_N16.png`, `.pdf` | M(t~), t~ = 5/Re(lambda_2) |
| `heatmap_m_t_tilde_5_over_abs_lambda2_96x96_v1000_N16.png`, `.pdf` | M(t~), t~ = 5/\|lambda_2\|; lambda_2 is real for every pair, so this is the same map as 5/Re(lambda_2) |
| `heatmap_m_t_tilde_1_over_abs_lambda2_96x96_v1000_N16.png`, `.pdf` | M(t~), t~ = 1/\|lambda_2\| |
| `m_t_tilde_96x96_v1000_N16.csv` | per pair: lambda_2, \|Im lambda_2\|, \|lambda_2\|, the three t~ and M at each |
| `heatmap_lambda_1_96x96_v1000_N16.png`, `.pdf`, `heatmap_lambda_2_...` | lambda_1 and lambda_2 over (a, b) |
| `heatmap_lambda_1_2_N4_vs_N16_96x96_v1000.png`, `.pdf` | 2x2: the lambda_1 and lambda_2 tables of the N = 4 run (top, from 11-08) and of this run (bottom); each panel keeps its own colour scale (`scripts/heatmap_lambda_N4_vs_N16.py`) |
| `heatmap_m_5t_star_N4_vs_N16_96x96_v1000.png`, `.pdf` | the mass retained M(5 t*) (sweep-method t*), N = 4 (left, from 11-08) beside N = 16 (right), on one shared log10 colour scale (`scripts/heatmap_m_5t_star_N4_vs_N16.py`) |
| `heatmap_m_t_tilde_5_over_re_lambda2_N4_vs_N16_96x96_v1000.png`, `.pdf` | M(t~), t~ = 5/Re(lambda_2) (= 5/\|lambda_2\|, lambda_2 being real), N = 4 (left, from 11-08) beside N = 16 (right), on one shared linear colour scale (`scripts/heatmap_m_t_tilde_5_N4_vs_N16.py`) |
| `heatmap_m_5t_star_and_t_tilde_5_N4_vs_N16_96x96_v1000.png`, `.pdf` | the two figures above together, 2x2: M(5 t*) on top, M(t~ = 5/Re(lambda_2)) below, N = 4 left and N = 16 right; each row on one colour scale for both N (`scripts/heatmap_m_5t_star_t_tilde_5_N4_vs_N16.py`) |
| `validation/` | the two checks above |
| `scripts/` | the engine, the run, the validation, and the figure, table and heatmap scripts |

## Results

| a | t* | lambda_1 | M(5 t*) | 10 t* |
|---|---|---|---|---|
| 0.1 | 0.4246 | 5.817 | 6.9e-6 | 4.246 |
| 1 | 0.4190–0.4192 | 5.728–5.730 | 9.6e-6–9.7e-6 | 4.190–4.192 |
| 10 | 0.3733–0.3750 | 4.960–4.977 | 1.3e-4–1.4e-4 | 3.733–3.750 |
| 100 | 0.2201–0.2239 | 1.807–1.836 | 0.143–0.152 | 2.201–2.239 |

- **b matters little.** Along each row, t* changes by at most 0.0038 (a = 100). At N = 4 the largest row spread was 0.0016.
- **lambda_2** is real for every pair, 30.85 to 56.75 over the grid.
- **M(10 t\*)**: 10 t* is past T = 4 for a = 0.1, 1, and inside it for a = 10, 100.
- **M(t~)** is 0.614–0.946 for t~ = 5/lambda_2 and 0.999021–0.999999 for t~ = 1/|lambda_2|. Both are largest at (100, 0.1).

Across N at v = 1000 (a = 100 row, the one that moves most):

| N | t* | lambda_1 | lambda_2 | M(5 t*) |
|---|---|---|---|---|
| 4 | 0.348–0.350 | 4.32–4.34 | 35.6–35.7 | 7.0e-4–7.4e-4 |
| 8 | 0.286–0.289 | 3.16–3.19 | 42.3–42.9 | 0.0123–0.0134 |
| 16 | 0.220–0.224 | 1.81–1.84 | 55.1–56.8 | 0.143–0.152 |

At a = 0.1 the three N agree to within 0.0005 in t* and 0.008 in lambda_1.

## Notes

- **Noise in the plotted y''.** The central-difference y'' at H carries noise of up to 1.5e-4 (`max_abs_reference_minus_cd_after_0p05`), from round-off in M divided by H². The reference y'' is far less affected.
- **The 10-07 rule here.** `central_difference_t_star` lies 5.3e-7 to 9.8e-5 after t*, within one sample.
- **Raw series.** The per-pair series and reference y'' are cached in `data_output/ab-grid-96x96_power_v1000_N16/` (and `_v1_N16/` for the v = 1 check), which are not in git. `run_96_power.py` regenerates them in seconds.
- **To reproduce** (from the repository root, with `June-2026/.venv/bin/python`):
  1. `scripts/run_96_power.py 1000 results/2026-11-12_ab-grid-96x96_v1000_N16_tau0.01`
  2. `scripts/figures_96_power.py`, `scripts/table_pdf.py`, `scripts/heatmaps_96_power.py`
  3. `scripts/validate_power.py modal` (about 1 min) and `scripts/validate_power.py stepper` (about 7 min)
