# t* on the 96×96 (a, b) grid, τ = 0.01 — 2026-10-07

Sources (96×96, v = 1, N_LIST = [0, 24, 48, 72], GPU, all sixteen pairs
(a, b) ∈ {0.1, 1, 10, 100}²):

- job `0b8d4022` — all sixteen pairs to **T = 1**;
- job `54cdf1f3` — the eight b ≤ 1 pairs to **T = 4**, still running. Each
  pair switches to its T = 4 series once that pair has finished (its final
  state saved); panels and table rows say which T they use. Done so far
  (2026-10-22): (0.1, 0.1), (0.1, 1), (1, 0.1), (1, 1), (10, 0.1), (10, 1). The job
  restarted from the first pair on 2026-10-17, so `scripts/t4_source.py` only
  counts final states saved by that run (see its docstring).

Both keep each pair's total-mass series (`timeseries_4.dat`, M itself, float64,
one sample per 5 steps); everything here is computed from those, with no
re-solve. Regenerate as more T = 4 pairs finish (see Reproducing).

Three criteria, one folder each. All use y = ln M with central differences
across s = 861 samples (H ≈ 1.0e-4), and the same right sweep: t* is the sample
just after the last one where the criterion's function exceeds τ = 0.01.

## Right-sweep |y″| ≤ τ (this folder; the committed criterion, `launch._char_time_onset`)

| (a, b) | T | t* |
|---|---|---|
| (0.1, 10), (0.1, 100), (1, 100), (10, 100) | 1 | 0.405–0.427 |
| (100, 100) | 1 | 0.447 |
| (100, 10), (1, 10), (10, 10) | 1 | 0.729, 0.750, 0.876 |
| (0.1, 1), (0.1, 0.1) | 4 | 1.545, 1.851 |
| (1, 1), (1, 0.1) | 4 | 2.304, 2.888 |
| (10, 1), (10, 0.1) | 4 | 2.908, 3.435 |
| (100, 0.1), (100, 1) | 1 (T = 4 pending) | none |

With slow switching onto the advective layer (b ≤ 1) y″ oscillates about zero
with shrinking swings before it settles; at T = 4 the first six such pairs
settle at t* = 1.5–3.4, later for smaller b and larger a. (10, 0.1) only just
qualifies: after t* = 3.435 y″ swings back up to 0.00993 at t ≈ 3.74, 0.7%
under τ, so that t* is marginal. The two still at T = 1 have |y″| well above
τ at the end of that run (still bending at −0.2 to −0.7). They are also
strongly grid-dependent (late decay rate ~0.4 at 16×16, ~0.6 at 32×32, ~1.2 at
96×96).

- `t_star_96x96_tau0.01.csv`, `t_star_table_96x96_tau0.01.png` — T, t*,
  m* = M(t*), M(T) (CSV also has the source job, fitted slope and τ)
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

κ gives a t* for fourteen pairs: 0.274–0.301 for b ≥ 10 and (0.1, 1), 0.494 for
(10, 10), 0.574 for (100, 10), 0.997 for (0.1, 0.1), and 1.450 / 1.563 /
1.822 / 2.338 for (1, 1) / (1, 0.1) / (10, 1) / (10, 0.1) at T = 4. The (0.1, 0.1) value looked like an end-of-run
artifact at T = 1 (κ dipped under τ only in the last samples); at T = 4 it is
unchanged and κ stays under τ to t = 4, so it was the real onset. Dividing by
y′² makes κ smaller than |y″| wherever the decay is fast (|y′| ≈ 5), so κ t*
is earlier than |y″| t* throughout.

## `line_intercept_criterion/` — the original definition

A line through log10 M at t = 0.4 and t = 0.5, extended to M = 1
(t* = −intercept/slope); old m* = M(10 t*). t* ≈ 0.068–0.083 for most pairs,
0.038 for (100, 10), and negative for (10, 0.1), (10, 1), (100, 0.1), (100, 1).
`lnM_line_intercept_96x96.png` (with the line and its two points), `_zoom.png`
(t* ± 0.15, ln M in [−0.5, 0.5]), and `t_star_line_intercept_96x96.csv`.

## `psd/` — |F[y″](ω)|²

The squared magnitude of the Fourier transform of y″, for each pair, from two
second derivatives on the same grid t_n = nH (H ≈ 1.0e-4): the central
difference above, and the second derivative of a Chebyshev fit of ln M on
[0, T]. P(ω) = |H Σ y″(t_n) e^{−iωt_n}|², no window, ω in rad per unit t.

- `psd_central_difference_96x96_full.png`, `psd_chebyshev_96x96_full.png` —
  full range up to Nyquist (π/H ≈ 3.1e4), log-log; red line at the peak, the
  largest |F|² over the plotted bins k ≥ 1 (ω = 0 is off the log axis), value
  in each panel title; H (time between the transformed y″ samples), Δω =
  2π/T (spacing between frequency points) and π/H in the title and each panel
- `..._peak_zoom.png` — the same spectra, ±8 bins (Δω = 2π/T) about the peak
- `psd_peaks_96x96.csv` — ω_peak per pair and method; `psd_96x96.npz` (ω and
  both spectra per pair), `psd_summary_96x96.csv` (Chebyshev degree and
  residual, and how far its y″ is from the central difference)

Both methods give the same ω_peak for every pair. Where the curve falls
monotonically (b ≥ 10 except a = 100, and (100, 100)) the peak is just the
first plotted bin, 2π/T.

`scripts/psd_96.py [t1_dir]`.

## Reproducing

`scripts/tau_mask_96.py [tau]`, `scripts/kappa_96.py [tau]` and
`scripts/line_intercept_96.py` take each pair's series through
`scripts/t4_source.py`: job `54cdf1f3`'s T = 4 series (read in place from
`data_output/jobs/54cdf1f3…/checkpoints/`) once the pair has finished, and
otherwise the sixteen T = 1 `tm_a{a}_b{b}.dat` files
(download from
`/jobs/0b8d4022/outputs/checkpoints/ab_grid_char_time_a{a}_b{b}/timeseries_4.dat`)
next to the script. All three cover the sixteen pairs; `tau_mask_96.py` and
`line_intercept_96.py` also take `main` / `additional` to reproduce the earlier
ten / six split.
