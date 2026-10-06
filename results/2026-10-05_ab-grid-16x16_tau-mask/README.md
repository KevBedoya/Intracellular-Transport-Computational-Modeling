# Right-sweep threshold mask on y'' — 2026-10-05

Candidate characteristic-time criterion, applied to the 16×16 runs in
`../2026-10-05_ab-grid-16x16_second-derivative/` (T = 1) and
`../2026-10-05_ab-pairs-16x16_T4/` (T = 4; used for the six pairs with
b ≤ 1, a ≤ 10). See the README in the first folder for the setup.

With y = ln M and y'' its second-order central difference (h = 1.5e-4):

    l(t) = 1 if |y''(t)| <= tau, else 0

Sweeping from the right, t^ is the last sample with l = 0, and t* is the
sample immediately after it, so |y''| <= tau from t* to the end of the run.
If the last sample already has l = 0, there is no t*.

Files, one set per tau (1e-3 and 1e-2):

- `lnM_tau{tau}_16x16.png` — ln M with t* in red
- `second_derivative_tau{tau}_16x16.png` — y'' with t* in red
- `second_derivative_tau{tau}_16x16_ylim0.1.png` — same, y in [-0.1, 0.1],
  dotted lines at ±tau
- `t_star_tau{tau}_16x16.csv` — t* and M(t*) per pair

At tau = 1e-3, seven pairs have no t*: |y''| is still 3e-3 to 7e-3 at the
end of the run. At tau = 1e-2 all sixteen have one; several sit within a few
percent of T. `scripts/tau_mask.py [tau]` regenerates everything from the
two curves CSVs.
