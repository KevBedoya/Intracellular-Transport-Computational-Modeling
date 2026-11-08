# 96x96, v = 1000, tau = 0.01: the (a, b) grid

This is the 2026-11-05 study (`../2026-11-05_ab-grid-96x96_modal_tau0.01/`) rerun at v = 1000. Everything else is unchanged:

- grid 96x96
- v = 1000 (11-05: v = 1)
- N = 4 evenly spaced (`N_LIST = [0, 24, 48, 72]`)
- tau = 0.01
- (a, b) in {0.1, 1, 10, 100}^2
- T = 4 for every pair

Every result in the 11-05 directory has a counterpart here, with `modal` replaced by `v1000` in the file names.

## Why this is not the modal engine

At v = 1000 the modal engine fails at 96x96. The eigenvector matrix V of the symmetry-reduced one-step operator (1332 modes) has cond(V) between 1e22 and 1e25; at v = 1 it is about 1e8. Fast advection along the tubes makes the operator nearly defective, so the modal amplitudes are round-off. In a probe, ln M(4) came out NaN for (1, 0.1), (10, 1), (10, 10) and (100, 0.1), and was scattered for the other pairs.

The job API (100.83.174.69:8000) was also offline, so everything here ran locally.

## How it was computed

`scripts/power_engine.py` keeps the modal engine's operator: the same one-step map P, built from the solver's own step functions and restricted to the same symmetry-orbit basis (`mass_analysis_modal.build_reduced_operator`). It replaces the eigen-expansion with exact matrix powers, x_k = P^k x_0. P_s is entrywise nonnegative (checked on every build), so no product ever cancels. Every entry of P^k keeps its relative accuracy however ill-conditioned the eigenbasis is. This is the time-stepper's arithmetic, x_{k+1} = P x_k, regrouped. The whole grid takes 22 s; the time-stepper runs at about 6,000 steps/s here, about 7 h per pair.

| quantity | how |
|---|---|
| M(t) | every H = 4305 dT = 1.0005e-4 over [0, 4], the 11-05 Mass Analysis sampling, from A = P^4305 applied in blocks |
| reference y'' | Richardson extrapolation of the central differences of ln M at 10H and 20H, (4 D(10H) − D(20H)) / 3. It replaces the exact modal y''. |
| t* | the 11-05 char-time rule on the reference y'': the last crossing of \|y''\| = tau, interpolated linearly between samples |
| `central_difference_t_star` | the 10-07 time-stepper rule (`launch._char_time_onset`) on the samples at spacing H, as on 11-05 |
| m*, M(5 t*), M(10 t*), M(t~) | by binary powers of P at any t. m* is also read off the samples by a cubic spline of ln M; the two agree to 2e-13 |
| lambda_1, lambda_2 | eigenvalues of L_s = (P_s − I) / dT, with the excitation test of `launch._slowest_rates`. Each mode's amplitude comes from its own left and right eigenvectors, not from solving V c = x_0 |
| fit_slope | least squares of log10 M over [t*, T], as the modal t* path |
| PSD columns | `y2_spectrum` of ln M at H over [0, T], as the char-time job |

The eigenvalues are reliable even though the full eigenbasis is not. Their condition numbers are 16–21 for lambda_1 and 21–27 for lambda_2 (`kappa_lambda_1`, `kappa_lambda_2`). Independently, −fit_slope · ln 10 reproduces lambda_1 to 5e-8 for every pair.

## Validation

`scripts/validate_power.py` writes both checks to `validation/`.

**1. v = 1 against the 11-05 modal study** (`validation/v1_power_vs_modal_11-05.csv`, all 16 pairs). At v = 1 the modal engine is well conditioned, so the whole pipeline can be checked against it.

| quantity | largest difference |
|---|---|
| lambda_1, lambda_2 | 4e-9 relative; \|Im lambda_2\| 6e-9 |
| M(t~), all three t~ | 5e-9 relative |
| PSD peak, FWHM | 6e-9 relative |
| fit_slope | 1e-7 relative |
| t* | 4.8e-7 absolute |
| m*, M(5 t*), M(10 t*) | 2.4e-6, 1.2e-5, 2.4e-5 relative: the t* difference times lambda_1 (and 5, 10) |
| reference y'' − exact modal y'', t > 0.05 | 5.3e-6 (the modal round-off bound there is 8.8e-6) |

The 10-07 rule's t* lands on the same sample in 13 of the 16 pairs. It is one sample (1e-4) off at (1, 1) and (10, 1). At (0.1, 1) it gives 1.5453 where 11-05 had 1.5469. That is the near-tangent pair the 11-05 README flags, where noise in its samples moved the crossing. 1.5453 is closer to the exact 1.5452.

**2. v = 1000 against the time-stepper** (`validation/v1000_power_vs_time_stepper.csv`). The time-stepper itself (`launch._solve_mass_analysis`, CPU) ran over T = 0.05 for (0.1, 100), (100, 0.1) and (10, 10). That is 2.15 million steps each, with one sample every 4305 steps; the time-stepper stores the masses of x_{j·4305 − 1} in slot j. Each run took about 400 s.

The layer masses agree to 3.9e-10 relative in DL, AL and total mass, for all three pairs. The gap grows linearly with the step count, at about 1.4 eps per step; this was checked sample by sample over the first 160,000 steps of (100, 0.1), and 1.4 × 2.15e6 × 1.1e-16 = 3.3e-10. It is the time-stepper's own round-off, added at every step. The power engine reaches each sample in a few dozen products and accumulates far less. Over a full T = 4 run the time-stepper would drift by about 3e-8.

## Files

Every figure is a 4x4 grid in the 10-07 layout: rows are a, columns are b, one pair per panel, T = 4 throughout. Red is t*, and dotted lines are ±tau.

| file | content |
|---|---|
| `d2lnM_central_difference_ylim0.1.png` | y'' = d² ln M/dt² by central difference at H, y-limits ±0.1 |
| `d2lnM_central_difference_ylim0.01.png` | the same at ±0.01 |
| `d2lnM_reference_over_central_difference.png` | the reference y'' (green) over the central difference (wide blue), ±0.1. This replaces 11-05's exact-over-central-difference figure |
| `lnM.png`, `lnM_t_star.png` | ln M(t), and the same with t* marked |
| `psd_y2.png` | \|F[y'']\|² on linear axes, each panel zoomed on its peak: red = peak, shaded = FWHM, dashed = left half-maximum crossing |
| `summary_96x96_v1000.csv` | per pair: t*, m*, M(t*) by powers, 10 t*, M(10 t*), the 10-07-rule t*, lambda_1, lambda_2, \|Im lambda_2\|, their condition numbers, fit_slope, the PSD peak and FWHM, max \|reference − central difference\| for t > 0.05, and M(T) |
| `t_star_m_10t_star_96x96_v1000.pdf` | table of t*, 10 t* and M(10 t*) per (a, b) |
| `heatmap_m_5t_star_96x96_v1000.png`, `.pdf` | M(5 t*) over (a, b), with 5 t* in each cell |
| `m_5t_star_96x96_v1000.csv` | per pair: t*, 5 t*, M(5 t*) |
| `heatmap_m_t_tilde_5_over_re_lambda2_96x96_v1000.png`, `.pdf` | M(t~), t~ = 5/Re(lambda_2) |
| `heatmap_m_t_tilde_5_over_abs_lambda2_96x96_v1000.png`, `.pdf` | M(t~), t~ = 5/\|lambda_2\|. lambda_2 is real for every pair here, so this is the same map as 5/Re(lambda_2) |
| `heatmap_m_t_tilde_1_over_abs_lambda2_96x96_v1000.png`, `.pdf` | M(t~), t~ = 1/\|lambda_2\| |
| `m_t_tilde_96x96_v1000.csv` | per pair: lambda_2, \|Im lambda_2\|, \|lambda_2\|, the three t~ and M at each |
| `heatmap_lambda_1_96x96_v1000.png`, `.pdf` | lambda_1 over (a, b) |
| `heatmap_lambda_2_96x96_v1000.png`, `.pdf` | lambda_2 over (a, b), coloured by Re(lambda_2) |
| `validation/` | the two checks above |
| `scripts/` | the engine, the run, the validation, and the figure, table and heatmap scripts |

A heatmap is coloured by log10 M when M spans a decade or more across the grid, and by M itself otherwise. Here that means log10 for M(5 t*) and linear for the three M(t~) maps.

## Results

- **b barely matters.** Along each row, t* changes by at most 0.0016 (a = 100) and lambda_1 by about 0.02. At v = 1 the same rows spanned a factor of 4.6 to 8 in t*.
- **t\*** is 0.4251 for a = 0.1, 0.4237 for a = 1, 0.4107–0.4112 for a = 10 and 0.3481–0.3497 for a = 100: 0.348–0.425 over the grid, against 0.405–3.43 at v = 1.
- **lambda_1** is 5.824, 5.802, 5.592–5.597 and 4.320–4.340 for a = 0.1, 1, 10, 100.
- **lambda_2** is real for every pair, 30.82 (a = 0.1) to 35.74 (a = 100). At v = 1 it was complex except at b = 100.
- **M(5 t\*)** is 6.7e-6 to 1.6e-5 for a ≤ 10 and 7.0e-4 to 7.4e-4 for a = 100.
- **M(10 t\*)**: 10 t* is past T = 4 for a ≤ 10 and inside it (3.48–3.50) for a = 100.
- **M(t~)** is 0.614–0.744 for t~ = 5/lambda_2 and 0.99901–0.99973 for t~ = 1/|lambda_2|. Both are largest at (100, 0.1).

## Notes

- **Noise in the plotted y''.** The central-difference y'' at H carries noise of about 1.4e-4 (`max_abs_reference_minus_cd_after_0p05`), as on 11-05 (1.7e-4). It comes from round-off in M of about 1e-12 relative divided by H². The reference y'' differences at 10H and 20H, so the same round-off is roughly 70 times smaller there.
- **The 10-07 rule here.** `central_difference_t_star` lies 9e-7 to 9e-5 after t*, within one sample.
- **Raw series.** The per-pair series and reference y'' are cached in `data_output/ab-grid-96x96_power_v1000/`, which is not in git. `run_96_power.py` regenerates them in about 20 s.
- **To reproduce** (from the repository root, with `June-2026/.venv/bin/python`):
  1. `scripts/run_96_power.py 1000 results/2026-11-08_ab-grid-96x96_v1000_tau0.01`
  2. `scripts/figures_96_power.py`, `scripts/table_pdf.py`, `scripts/heatmaps_96_power.py`
  3. `scripts/validate_power.py modal` (about 1 min) and `scripts/validate_power.py stepper` (about 7 min)
