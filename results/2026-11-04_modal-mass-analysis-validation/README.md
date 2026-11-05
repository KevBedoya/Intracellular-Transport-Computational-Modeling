# Modal-expansion mass analysis vs the time-stepper

`computational_tools/mass_analysis_modal.py` (`comp_mass_analysis_modal_expansion`)
fills the same five mass series as `analysis_tools.comp_mass_analysis_respect_to_time`
from the full eigen-expansion of the solver's one-step operator, restricted to
its symmetric subspace. This study measures the difference in the total mass
M(t) between the two.

    e_j   = TM_step[j] - TM_modal[j]          (same samples, MA_collection_factor = 5)
    L_inf = max_j |e_j|
    L_2   = sqrt(h sum_j e_j^2),  h = 5 dT    (~ the L2 norm over [0, T])

288 runs: rg = ry = G, N_LIST = [0, G/4, G/2, 3G/4], centred seed, d_tube = 0,
(a, b) in {0.1, 1, 10, 100}^2;
G = 16, 24, 32, 48 at v = 0.1, 1, 10, T = 1; G = 64 at v = 1, T = 1;
G = 16, 32 at v = 1, T = 0.25, 4; and G = 96 at v = 1, T = 1 against the
stored GPU series of job 0b8d4022. Everything else is against the CPU
time-stepper, run fresh. `scripts/sweep.py`, then `scripts/analyse.py`.

## Worst case per grid (`worst_cases.csv`)

| G | T | worst L_inf | at (a, b, v) | worst L_2 | at (a, b, v) |
|---|---|---|---|---|---|
| 16 | 0.25 | 4.9e-13 | (100, 1, 1) | 1.5e-13 | (1, 0.1, 1) |
| 16 | 1 | 1.3e-12 | (100, 1, 1) | 8.6e-13 | (100, 1, 1) |
| 16 | 4 | 1.6e-12 | (100, 1, 1) | 2.8e-12 | (100, 1, 1) |
| 24 | 1 | 6.7e-12 | (100, 0.1, 0.1) | 3.6e-12 | (100, 0.1, 0.1) |
| 32 | 0.25 | 1.2e-11 | (100, 10, 1) | 4.2e-12 | (100, 100, 1) |
| 32 | 1 | 2.0e-11 | (100, 0.1, 0.1) | 1.3e-11 | (100, 0.1, 0.1) |
| 32 | 4 | 1.6e-11 | (100, 0.1, 1) | 2.6e-11 | (100, 0.1, 1) |
| 48 | 1 | 1.4e-10 | (100, 1, 0.1) | 9.2e-11 | (100, 1, 0.1) |
| 64 | 1 | 1.8e-10 | (1, 100, 1) | 1.0e-10 | (1, 100, 1) |
| 96 (GPU ref) | 1 | 1.7e-9 | (100, 0.1, 1) | 1.4e-9 | (100, 0.1, 1) |

## Error model (`error_model.json`, `error_model.png`)

Least squares on log10 of the 270 CPU-reference runs:

    L_inf ~ 1.37e-18 G^4.45 T^0.07 a^0.04 b^-0.01 v^-0.007     R^2 = 0.95, rms 0.19 decades
    L_2   ~ 7.6e-19  G^4.46 T^0.29 a^0.08 b^-0.02 v^-0.006     R^2 = 0.94, rms 0.22 decades

Multiplying by 10^0.59 (L_inf) or 10^0.85 (L_2), the largest residual, bounds
every run. Fitted on G <= 48 alone, the model predicts G = 64 and the G = 96 GPU
runs to 0.22 and 0.17 decades rms, under-predicting by at most 0.15 decades.

The grid dominates: G^4.45 tracks the step count K ~ T G^4 that the
time-stepper accumulates round-off over (the worst L_inf is 0.1-0.25 eps K at
T = 1 on every grid, eps K = 7e-12 at 16x16 and 1e-8 at 96x96), while a, b and v
move it by a factor ~1.3 at most over their swept ranges, and T matters little
because the absolute error stops growing once M has decayed. The error does not
follow cond(V), which varies by 10^4 at fixed G, so it is the time-stepper's
accumulated round-off that is being measured, not the modal sum's.
