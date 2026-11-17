# 96x96, tau = 0.01: the (a, b) grid at v = 1 and 1000, N = 4 and 16

This study puts the four (v, N) combinations side by side:

| v | N (evenly spaced) | `N_LIST` | engine |
|---|---|---|---|
| 1 | 4 | 0, 24, 48, 72 | modal expansion |
| 1000 | 4 | 0, 24, 48, 72 | power engine |
| 1 | 16 | 0, 6, 12, .., 90 | modal expansion |
| 1000 | 16 | 0, 6, 12, .., 90 | power engine |

The other parameters are fixed: grid 96x96, tau = 0.01, (a, b) in {0.1, 1, 10, 100}^2 (the values in `../2026-11-12_ab-grid-96x96_v1000_N16_tau0.01/heatmap_m_t_tilde_1_over_abs_lambda2_96x96_v1000_N16.png`), and T = 4.

The request listed (v = 1, N = 4) twice and did not list (v = 1, N = 16). The second entry was taken to be (v = 1, N = 16), which completes the 2x2.

## Method

Each pair uses the characteristic-time method of the modal char-time job:

- **t\*** is the last crossing of |y''| = tau, where y = ln M, found by the right sweep.
- **lambda_1, lambda_2** are the slowest decay rate and the next excited one. They come from the eigenvalues of the symmetry-reduced one-step operator, by the rule in `launch._slowest_rates`.
- **t^ = 1/|lambda_2|.** The heatmaps show M(5 t^), that is M at 5/|lambda_2|.

**v = 1** uses the modal expansion itself (`mass_analysis_modal.solve_modes`):

- t*, lambda_1 and lambda_2 come from `launch._char_time_onset_modal`, applied to the exact modal y''.
- M comes from the modal sum (`mam.ln_mass`).
- cond(V) is 7.8e4 to 1.5e8.

**v = 1000** cannot use the modal sum. cond(V) is 1.1e22 to 1.5e25 (column `cond_V`), so the modal amplitudes are round-off; see the 11-08 and 11-12 READMEs. It uses `scripts/power_engine.py` instead, copied unchanged from 11-12. That engine takes exact powers of the same one-step operator:

- t* comes from the same right sweep, applied to the reference y'' (Richardson extrapolation on central differences at 10H and 20H).
- lambda_1 and lambda_2 come from the eigenvalues of the same operator.
- M comes from binary powers of the operator.

M(t) is sampled every H = 4305 dT = 1.0005e-4 over [0, 4].

The whole job ran locally in 53 s on 8 workers: 64 pairs, plus the v = 1 cross-check on the power engine. The log is `scripts/run_vN.log`.

## Files

| path | content |
|---|---|
| `m_5t_star/heatmap_m_5t_star_96x96_<v>_<N>.png`, `.pdf` | M(5 t*) over (a, b) for one (v, N), with 5 t* in each cell. The layout and colour rule follow the reference heatmap: log10 M when M spans a decade or more, M itself otherwise. |
| `m_5t_star/heatmap_m_5t_star_96x96_all4.png`, `.pdf` | the four together, 2x2: rows N = 4, 16; columns v = 1, 1000. All four panels share one log10 colour scale. |
| `m_5t_hat/heatmap_m_5t_hat_96x96_<v>_<N>.png`, `.pdf` | M(5 t^), t^ = 1/\|lambda_2\|, with 5 t^ in each cell |
| `m_5t_hat/heatmap_m_5t_hat_96x96_all4.png`, `.pdf` | the four together, laid out as above, on one shared log10 scale |
| `mass_vs_time/M_vs_t_96x96_<v>_<N>.png` | M(t) on [0, 4], one panel per (a, b), with t* (red), 5 t* (red dashed) and 5 t^ (purple dash-dot) marked where they fall inside [0, 4]. Their values are printed in each panel. |
| `mass_vs_time/M_vs_t_logy_96x96_<v>_<N>.png` | the same on a log y-axis |
| `eigenvalues/heatmap_lambda_1_96x96_<v>_<N>.png`, `.pdf` | lambda_1 over (a, b) |
| `eigenvalues/heatmap_lambda_2_96x96_<v>_<N>.png`, `.pdf` | lambda_2 over (a, b), coloured by Re(lambda_2). Cells show Re ± i\|Im\| and \|lambda_2\|, or "real". |
| `d2lnM/d2lnM_central_difference_ylim0.1_96x96_<v>_<N>.png`, `_ylim0.01_` | y'' by central difference at H, with t* in red and ±tau dotted. The y-limits are ±0.1 and ±0.01, as in the 10-07, 11-05 and 11-12 figures. |
| `data/summary_96x96_<v>_<N>.csv`, `data/summary_96x96_all.csv` | per pair: t*, 5 t*, M(5 t*), m* = M(t*), lambda_1, lambda_2, \|Im lambda_2\|, \|lambda_2\|, t^, 5 t^, M(5 t^), the 10-07-rule t* on the samples, M(T), cond(V) of the modal eigenvectors, and kappa of lambda_1 and lambda_2 |
| `validation/v1_modal_vs_power.csv` | v = 1: the modal expansion against the power engine, pair by pair |
| `validation/vs_earlier_runs.csv` | each (v, N) against the run already in `results/` |
| `scripts/` | `run_vN.py` (the job), `figures_vN.py` (every figure), `power_engine.py`, and the run log |

`<v>_<N>` is one of `v1_N4`, `v1000_N4`, `v1_N16`, `v1000_N16`.

## Results

| v, N | t* | 5 t* | M(5 t*) | 5 t^ | M(5 t^) | lambda_1 | \|lambda_2\| |
|---|---|---|---|---|---|---|---|
| 1, 4 | 0.405–3.435 | 2.03–17.17 | 2.6e-30 – 1.2e-5 | 0.162–0.655 | 0.058–0.613 | 1.20–5.83 | 7.63–30.81 |
| 1000, 4 | 0.348–0.425 | 1.74–2.13 | 6.7e-6 – 7.4e-4 | 0.140–0.162 | 0.614–0.744 | 4.32–5.82 | 30.82–35.74 |
| 1, 16 | 0.425–3.646 | 2.13–18.23 | 1.2e-30 – 0.198 | 0.162–0.663 | 0.063–0.821 | 0.26–5.83 | 7.54–30.81 |
| 1000, 16 | 0.220–0.425 | 1.10–2.12 | 6.9e-6 – 0.152 | 0.088–0.162 | 0.614–0.946 | 1.81–5.82 | 30.85–56.75 |

- **At v = 1, lambda_2 is complex for every b ≤ 10** (12 of 16 pairs, at both N). That is why 5 t^ = 5/\|lambda_2\| differs there from 5/Re(lambda_2). At v = 1000, lambda_2 is real for every pair.
- **At v = 1, b controls the result.** For b ≤ 1, t* is 1.2 to 3.6 and M(5 t*) falls to 1e-19 to 1e-30. For b = 100, every a gives t* of about 0.43 and M(5 t*) of about 7e-6.
- **At v = 1000, b barely matters.** Along each row, t* changes by less than 0.004, and a sets the result. N matters only at large a: at a = 100, M(5 t*) goes from 7e-4 at N = 4 to 0.15 at N = 16.
- **The b = 100 column at v = 1 approaches the v = 1000 values** for a ≤ 10. For example, (0.1, 100) at v = 1, N = 4 has t* = 0.4253, against 0.4251 at v = 1000.

## Validation

**1. v = 1, modal expansion against power engine** (`validation/v1_modal_vs_power.csv`, all 32 pairs):

| quantity | largest difference |
|---|---|
| lambda_1, lambda_2, \|Im lambda_2\| | none (identical) |
| t* | 1.3e-6 absolute |
| M(5 t^) | 1.3e-9 relative |
| m*, M(5 t*) | 7.0e-6 and 3.5e-5 relative. This is the t* difference times lambda_1 (and times 5). |

**2. Against the earlier runs** (`validation/vs_earlier_runs.csv`):

| (v, N) | earlier run | t* | M(5 t*), M(5 t^), lambda |
|---|---|---|---|
| (1, 4) | 11-05 modal study | ≤ 1.5e-9 | ≤ 4.2e-8 relative |
| (1000, 4) | 11-08 | ≤ 6e-17 | ≤ 1e-13 relative |
| (1000, 16) | 11-12 | ≤ 8e-17 | ≤ 7e-13 relative |
| (1, 16) | the modal t* of the 11-12 v = 1 validation | ≤ 6e-17 | – |

(1, 16) has no study of its own to compare against.

The 10-07 rule on the sampled M (`central_difference_t_star`) lands within 1e-4 of t* for every pair.

## Notes

- **Figure titles don't name the engine.** The modal and power engines agree to 7–10 digits at v = 1 (see Validation), so the titles describe the method only. Which engine produced each (v, N) is recorded in the `engine` column of `data/` and in the table at the top of this README.
- **5 t\* is past T = 4 for 9 pairs at v = 1, at each N.** These are the eight b ≤ 1 pairs, plus (10, 10) at N = 4 or (1, 10) at N = 16. M(5 t*) there is still exact, because both engines evaluate M at any t. The M(t) figures stop at T = 4, so they do not draw the 5 t* line for those pairs, but its value is printed in each panel.
- **Shared colour scale in the `_all4` figures.** Both composites colour log10 M on one scale across all four panels, so every colour means the same mass everywhere. M(5 t*) spans 1e-30 to 0.2, so the v = 1000 panels look nearly uniform; the numbers in the cells carry the detail. The single-(v, N) heatmaps each use their own scale.
- **t\* = 0.49999986 at (v = 1, N = 16, a = b = 100)** is a coincidence, not an artifact. The modal and power engines agree on it to 3e-8, and there was no sweep warning.
- **Raw series.** The per-pair series (t, M, and the power engine's reference y'') are cached in `data_output/ab-grid-96x96_vN_<v>_<N>/`, which is not in git.
- **To reproduce**, run these from the repository root with `June-2026/.venv/bin/python`:
  1. `scripts/run_vN.py` (about 1 min)
  2. `scripts/figures_vN.py` (about 15 s)
