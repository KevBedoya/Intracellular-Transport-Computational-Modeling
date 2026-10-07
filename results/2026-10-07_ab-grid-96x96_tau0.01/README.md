# t* on the 96×96 (a, b) grid, τ = 0.01 — 2026-10-07

Source: job `0b8d4022` (96×96, v = 1, N_LIST = [0, 24, 48, 72], T = 1, GPU).
Its checkpoints keep each pair's total-mass series (`timeseries_4.dat`, M
itself, float64, one sample per 5 steps); everything here is computed from
those, with no re-solve. The six pairs studied to T = 4 at 16×16
((10,0.1), (10,1), (1,0.1), (1,1), (0.1,0.1), (0.1,1)) are excluded and shown
as blank panels.

## Right-sweep mask (the committed criterion, `launch._char_time_onset`)

y = ln M; y″ is the second-order central difference across s = 861 samples,
H ≈ 1.0e-4. l(t) = 1 where |y″| ≤ τ = 0.01; t* is the sample just after the
last l = 0, so |y″| ≤ τ from t* to T.

- `t_star_96x96_tau0.01.csv`, `t_star_table_96x96_tau0.01.png` — t*, m* = M(t*),
  M(T = 1) (CSV also has the fitted slope and τ)
- `lnM_96x96_tau0.01.png` — ln M with t* in red; `_zoom` windows each panel to
  t* ± 0.15 and ln M(t*) ± 0.5 (`_zoom_clipped` stops at [0, T])
- `second_derivative_96x96_tau0.01.png` — y″, no zoom; `_ylim0.3/0.2/0.1` —
  y in ±0.3 / ±0.2 / ±0.1 with ±τ dotted

(100, 0.1) and (100, 1) have no t*: y″ is still −0.2 to −0.7 at T = 1. These
two pairs are strongly grid-dependent (late decay rate ~0.4 at 16×16, ~0.6 at
32×32, ~1.2 at 96×96); at 16×16 they settle just before T, at 96×96 they do
not.

## `line_intercept_criterion/` — the original definition, for comparison

A line through log10 M at t = 0.4 and t = 0.5, extended to M = 1
(t* = −intercept/slope); old m* = M(10 t*). t* ≈ 0.071–0.083 for most pairs,
0.038 for (100, 10), and negative for (100, 0.1) and (100, 1).
`lnM_line_intercept_96x96.png` (with the line and its two points) and
`_zoom.png` (t* ± 0.15, ln M in [−0.5, 0.5]); table in
`t_star_line_intercept_96x96.csv`.

## Reproducing

`scripts/tau_mask_96.py [tau]` and `scripts/line_intercept_96.py` expect the
ten `tm_a{a}_b{b}.dat` files (download from
`/jobs/0b8d4022/outputs/checkpoints/ab_grid_char_time_a{a}_b{b}/timeseries_4.dat`)
next to the script.
