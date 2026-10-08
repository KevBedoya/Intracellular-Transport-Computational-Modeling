# t* on the 96×96 (a, b) grid, τ = 0.01 — 2026-10-07

Source: job `0b8d4022` (96×96, v = 1, N_LIST = [0, 24, 48, 72], T = 1, GPU),
all sixteen pairs (a, b) ∈ {0.1, 1, 10, 100}². Its checkpoints keep each
pair's total-mass series (`timeseries_4.dat`, M itself, float64, one sample per
5 steps); everything here is computed from those, with no re-solve.

Three criteria, one folder each. All use y = ln M with central differences
across s = 861 samples (H ≈ 1.0e-4), and the same right sweep: t* is the sample
just after the last one where the criterion's function exceeds τ = 0.01.

## Right-sweep |y″| ≤ τ (this folder; the committed criterion, `launch._char_time_onset`)

| (a, b) with a t* | t* | | (a, b) without |
|---|---|---|---|
| (0.1, 10), (0.1, 100), (1, 100), (10, 100) | 0.405–0.427 | | all six with b ≤ 1 at a ≤ 10 |
| (100, 100) | 0.447 | | (100, 0.1), (100, 1) |
| (100, 10), (1, 10), (10, 10) | 0.729, 0.750, 0.876 | | |

The eight pairs without t* still have |y″| well above τ at T = 1 (oscillating
for b ≤ 1, a ≤ 10; still bending at −0.2 to −0.7 for a = 100, b ≤ 1). At 16×16
the b ≤ 1 pairs needed T ≈ 4; (100, 0.1) and (100, 1) are also strongly
grid-dependent (late decay rate ~0.4 at 16×16, ~0.6 at 32×32, ~1.2 at 96×96).

- `t_star_96x96_tau0.01.csv`, `t_star_table_96x96_tau0.01.png` — t*,
  m* = M(t*), M(T = 1) (CSV also has the fitted slope and τ)
- `lnM_96x96_tau0.01.png` — ln M with t* in red; `_zoom` windows each panel to
  t* ± 0.15 and ln M(t*) ± 0.5 (`_zoom_clipped` stops at [0, T])
- `second_derivative_96x96_tau0.01.png` — y″, full range; `_ylim0.3/0.2/0.1`
  — y in ±0.3 / ±0.2 / ±0.1 with ±τ dotted

## `kappa_criterion/` — κ = |y″| / y′² ≤ τ

The criterion used 2026-09-28 to 2026-10-05, with τ = 0.01 and the derivatives
above (y′ = (y[i+s] − y[i−s]) / 2H). κ is the rate at which the local e-folding
time 1/λ drifts (λ = −y′). The earlier version also required ≥ 0.1 of run
after t*; here that is a column, not applied.

- `kappa_96x96_tau0.01_ylim0.1.png`, `_ylim0.03.png` — κ(t), the function
  that decides the criterion, with τ dotted and t* in red
- `lnM_kappa_96x96_tau0.01.png` and `_zoom.png` — ln M with t*
- `t_star_kappa_96x96_tau0.01.csv`, `t_star_table_kappa_96x96_tau0.01.png` —
  t*, m*, M(T), κ(T) and whether ≥ 0.1 of run follows t*

κ gives a t* for ten pairs: 0.274–0.301 for b ≥ 10 (and (0.1, 1)), 0.494 for
(10, 10), 0.574 for (100, 10), and 0.997 for (0.1, 0.1) — the last only
because κ dips just under τ in the final samples (flagged "no" in the 0.1-run
column). Dividing by y′² makes κ smaller than |y″| wherever the decay is fast
(|y′| ≈ 5), so κ t* is earlier than |y″| t* for b ≥ 10.

## `line_intercept_criterion/` — the original definition

A line through log10 M at t = 0.4 and t = 0.5, extended to M = 1
(t* = −intercept/slope); old m* = M(10 t*). t* ≈ 0.068–0.083 for most pairs,
0.038 for (100, 10), and negative for (10, 0.1), (10, 1), (100, 0.1), (100, 1).
`lnM_line_intercept_96x96.png` (with the line and its two points), `_zoom.png`
(t* ± 0.15, ln M in [−0.5, 0.5]), and `t_star_line_intercept_96x96.csv`.

## Reproducing

`scripts/tau_mask_96.py [tau]`, `scripts/kappa_96.py [tau]` and
`scripts/line_intercept_96.py` expect the sixteen `tm_a{a}_b{b}.dat` files
(download from
`/jobs/0b8d4022/outputs/checkpoints/ab_grid_char_time_a{a}_b{b}/timeseries_4.dat`)
next to the script. All three cover the sixteen pairs; `tau_mask_96.py` and
`line_intercept_96.py` also take `main` / `additional` to reproduce the earlier
ten / six split.
