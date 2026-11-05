# 96x96, tau = 0.01: the (a, b) grid on the modal engine

This is the 2026-10-07 study (`../2026-10-07_ab-grid-96x96_tau0.01/`) recomputed with the
eigen-modal expansion (`engine = "modal"`) in place of the time-stepper.

The fixed parameters are the same as on 10-07:

- grid 96x96
- v = 1
- N = 4 evenly spaced (`N_LIST = [0, 24, 48, 72]`)
- tau = 0.01
- (a, b) in {0.1, 1, 10, 100}^2
- T = 4 for every pair. On 10-07, the b >= 10 pairs used T = 1.

## Jobs

All of these ran on the API on 2026-11-05. Job ids are in `scripts/jobs_96_modal.json`.

| what | jobs | params | wall time |
|---|---|---|---|
| Mass Analysis (+ PSD of y'') | 16, one per (a, b) | `engine=modal`, `compute_psd=true`, `MA_collection_factor=4305` (one sample every H = 1.0005e-4, the 10-07 differencing step) | ~55 s each, 4 at a time |
| Characteristic Time (a,b grid) | 1 (`ff0ab37a`) | `engine=modal`, `tau=0.01`, `compute_psd=true`, `char_time_mult=1`, `MA_collection_factor=5` | 2 min |

The whole grid took about 6 min. The 10-07 time-stepper runs took days on the GPU.

## Files

Every figure is a 4x4 grid in the 10-07 layout: rows are a, columns are b, one pair per panel, T = 4 throughout. Red is t*, and dotted lines are ±tau.

| file | content |
|---|---|
| `d2lnM_central_difference_ylim0.1.png` | y'' = d² ln M/dt² by central difference at H, y-limits ±0.1 |
| `d2lnM_central_difference_ylim0.01.png` | the same at ±0.01. The ±tau lines sit on the plot edges. |
| `d2lnM_exact_over_central_difference.png` | the only superimposed figure: exact modal y'' (green) over the central difference (wide blue), ±0.1 |
| `lnM.png`, `lnM_t_star.png` | ln M(t), and the same with t* marked |
| `psd_y2.png` | \|F[y'']\|² on linear axes, each panel zoomed on its peak: red = peak, shaded = FWHM, dashed = left half-maximum crossing |
| `summary_96x96_modal.csv` | per pair: t*, m* = M(t*), M(10 t*) (`m_at_10_t_star`, from the modal series, since 10 t* is past T = 4 for every pair), the central-difference-rule t*, lambda_1, lambda_2, the PSD peak and FWHM, and max \|exact − central difference\| for t > 0.05 |
| `t_star_m_10t_star_96x96_modal.pdf` | table of t*, 10 t* and M(10 t*) per (a, b) (`scripts/table_pdf.py`) |
| `heatmap_m_5t_star_96x96_modal.png`, `.pdf` | heatmap of M(5 t*) over (a, b), rows a, columns b, colour log10 M(5 t*). Each cell shows M(5 t*) and 5 t* (`scripts/heatmap_m_5t_star.py`) |
| `heatmap_m_t_tilde_5_over_re_lambda2_96x96_modal.png`, `.pdf` | the same layout for M(t~), t~ = 5/Re(lambda_2) (`scripts/heatmap_m_t_tilde.py`) |
| `heatmap_m_t_tilde_5_over_abs_lambda2_96x96_modal.png`, `.pdf` | the same for t~ = 5/\|lambda_2\|. For b = 100 lambda_2 is real, so these cells match the Re(lambda_2) figure |
| `heatmap_m_t_tilde_1_over_abs_lambda2_96x96_modal.png`, `.pdf` | the same for t~ = 1/\|lambda_2\|. Here M(t~) only spans 0.74–1.00, so the colour is M itself on a linear scale, not log10 M |
| `heatmap_lambda_1_96x96_modal.png`, `.pdf` | lambda_1 (the slowest decay rate of M) over (a, b), from the char-time job (`scripts/heatmap_lambda.py`) |
| `heatmap_lambda_2_96x96_modal.png`, `.pdf` | lambda_2 over (a, b), coloured by Re(lambda_2). Each cell shows Re ± i\|Im\| and \|lambda_2\|; it is real for b = 100 |
| `m_t_tilde_96x96_modal.csv` | per pair: Re and \|Im\| of lambda_2 (from the summary), \|lambda_2\|, the three t~ and M at each, and lambda_2 recomputed from the local solve (agrees to ≤ 6e-9) |
| `m_5t_star_96x96_modal.csv` | per pair: t*, 5 t*, M(5 t*), and M at t* and 10 t* recomputed against the summary's m* and M(10 t*) (agree to ≤ 1.2e-7 relative; see Notes) |
| `scripts/` | submit, wait and figure scripts, plus the job ids |

## Agreement with the 10-07 time-stepper

The modal t* matches the 10-07 values to 5–7 digits for every pair. The relative difference ranges from 4e-8 to 7e-6.

The modal m* matches to 4–7 digits.

On 10-07 the (100, 0.1) and (100, 1) pairs had t* only from the T = 4 log of job 54cdf1f3 (2.33743 and 1.92414). The modal values are 2.337432 and 1.924143.

This agreement is closer than in the a = b = 100, tau = 1e-3 grid-convergence comparison, which reached about 4 digits. That gap came from noise in the time-stepper's y'' (round-off divided by H²) relative to tau. At tau = 0.01 the same noise moves the crossing about 10x less.

## Notes

- **Noise in the plotted y''.** The central-difference y'' carries noise of about 1.7e-4 (`max_abs_exact_minus_cd_after_0p05`). It shows as fuzz in the ±0.01 plot, comes from round-off in M of about 1e-12 relative divided by H², and stays well below tau.
- **Where t\* comes from.** t* in this directory is taken from the exact modal y'' (char-time job).
- **The 10-07 rule on these samples.** `central_difference_t_star` applies the 10-07 right-sweep rule to the sampled M. It agrees with the modal t* to about 1e-4 everywhere except (0.1, 1), where it gives 1.5469 against 1.5452. There y'' approaches −tau almost tangentially (its minimum is about −0.011), so the noise shifts the crossing.
- **FWHM.** It is 2(omega_peak − omega_left). For pairs whose PSD stays above half-maximum down to omega = 0, it is read on the mirrored side (`psd_fwhm_side = mirrored`). The PSD peaks at omega = 0 for (0.1, 10), (1, 10) and all four b = 100 pairs; (10, 10) and (100, 10) peak at 5.3 and 11.4.
- **M(5 t\*).** It comes from the spectral (eigen-modal) expansion of the 96x96 one-step operator (`mass_analysis_modal.solve_modes`, `ln_mass`), re-solved locally for each pair. 5 t\* runs from 2.03 to 17.17. It is past T = 4 for every pair except (0.1, 10), (0.1, 100), (1, 10), (1, 100), (10, 100), (100, 10) and (100, 100). The local re-solve reproduces the summary's m\* and M(10 t\*) to 1e-12–1.2e-7 relative, against 3e-15 on the original machine. The difference comes from a different LAPACK under cond(V) ~ 1e8. The largest gap is (100, 0.1).
