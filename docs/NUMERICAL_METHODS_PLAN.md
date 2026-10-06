# Alternative numerical methods for φ(r,θ): assessment and plan

Measured on the project workstation, August 2026, against the schemes currently
in `computational_tools/`. Every figure below is either **measured** here or
**derived** from a measured one; extrapolations are labelled as such. Nothing is
taken from a textbook estimate.

The short version: the largest available win is not a faster implementation and
not a GPU. It is that **the spatial discretisation is first-order accurate when
it could be second-order**, for two specific and separately fixable reasons. At a
fixed grid that costs a factor of ~10³ in accuracy; since total work grows as
`G⁶`, buying that accuracy back by refinement is what makes 224² runs take days.

---

## 1. What was measured, and how

Four experiments, all reproducible from `docs/` (§12):

1. **Spectrum of the assembled operator.** The diffusive-layer stencil was
   transcribed from `u_density`/`j_*`/`u_center` into a sparse matrix and its
   extreme eigenvalues computed directly. This says what limits the timestep and
   how stiff the system is, rather than inferring it from the `compute_dT`
   formula.
2. **Order of accuracy.** The angularly-uniform (`n = 0`) sector has an exact
   answer to check against — the disc's fundamental Dirichlet eigenvalue
   `j₀,₁² = 5.783185963`. Convergence to it gives the spatial order with no
   reference run needed.
3. **Error localisation.** The same stencil on an *annulus* (no pole, no central
   patch) separates interior error from pole and boundary error.
4. **Kernel microbenchmarks.** Variants of the φ stencil under numba, against a
   pure-copy memory roofline, to decide whether a lower-level language is
   warranted.

---

## 2. Inventory: what schemes exist and what they share

| Layer | Where | Role |
|---|---|---|
| Spatial operator | `numerical_tools.u_density`, `u_density_rect`, `u_tube`, `u_tube_rect`, `u_center`, `j_r_r`, `j_l_r`, `j_r_t`, `j_l_t` | The scheme. One family, two variants (`d_tube = 0` vs finite extraction width). |
| Time loops | ~13 drivers in `analysis_tools.py`, plus `mfpt_comp_functions.py`, `time_analysis.py`, `super_comp.py` | Explicit Euler, duplicated per output type. |
| GPU port | `gpu/persistent_kernel.py` | **Reimplements** the operator inline. |
| Experimental copy | `experimental/computational_test/numerical_methods.py` | A third copy. |
| Reference solutions | `analytic_solution.py`, `analytic_benchmark.py` | Series solution + comparison harness. |
| Reduced model | `modal_model/` | Fourier–Bessel truncation, `F = Σ A_{n,j} J_{nN}(κ r) cos(nNθ) + …`. A model-reduction prototype with a documented limitation (a symmetric centred IC excites only `n = 0` cos-modes), **not** an equivalent-accuracy alternative solver. |

Two structural facts drive the plan:

* **One spatial operator, three copies.** Fixing the discretisation means
  changing `numerical_tools`, `gpu/persistent_kernel`, and the experimental copy
  — or deleting the latter two in favour of one source.
* **One time integrator, ~15 copies.** Every driver inlines its own
  `for k in range(K)` explicit-Euler loop. **Swapping the integrator is
  currently a 15-site change.** Consolidating the loops behind one driver is a
  prerequisite for everything in §8, not an optional tidy-up.

---

## 3. What actually costs: the three laws

**Stability.** `compute_dT` sets
`Δt = 0.1 · min(Δr², Δθ²Δr²) / (2D)`. Since `Δθ = 2π/N < 1` for `N > 6`, the
angular term always wins, giving `Δt ∝ Δr²Δθ² ∝ 1/(M²N²)`, hence `K ∝ G⁴` and
**total work ∝ G⁶**.

Measured, this is exactly right, and it is the *innermost ring* that binds:

| grid | ‖λ‖max | Δt stable = 2/‖λ‖max | angular CFL | radial CFL | binding | Δt code / Δt stable |
|---|---|---|---|---|---|---|
| 16² | 7.434e3 | 2.690e-4 | 3.012e-4 | 1.953e-3 | angular | 0.112 |
| 32² | 1.093e5 | 1.829e-5 | 1.883e-5 | 4.883e-4 | angular | 0.103 |
| 48² | 5.448e5 | 3.671e-6 | 3.719e-6 | 2.170e-4 | angular | 0.101 |
| 64² | 1.712e6 | 1.168e-6 | 1.177e-6 | 1.221e-4 | angular | 0.101 |

The radial CFL is ~100× looser and never binds. `‖λ‖max` scales as `G⁴` as
predicted (measured ratios 3.09 and 4.98 against 3.16 and 5.06 for `G⁴`).

The angular Laplacian carries a `1/r²`, so at the innermost ring the coefficient
is `1/(Δr²Δθ²)` — the classic polar pole problem. **The whole grid is paying a
timestep that only ring 0 needs.** Ring `m` could tolerate `(m+1)²` times more.

**Stiffness.** `‖λ‖min → 5.783` (measured 5.29 → 5.67 over 16²…64²), which is
`j₀,₁²` — a useful confirmation that the assembled operator is correct. The
stiffness ratio is therefore

| grid | 16² | 32² | 48² | 64² | 224² (extrapolated `G⁴`) |
|---|---|---|---|---|---|
| ‖λ‖max/‖λ‖min | 1.41e3 | 1.97e4 | 9.68e4 | 3.02e5 | **≈ 4.5e7** |

A stiffness ratio of `10⁷` with an explicit integrator is the textbook case for
an implicit or exponential method.

**Advection.** With `v = 10⁴`, the advective CFL `Δt ≤ Δr/|v|` is the constraint
that *survives* removing diffusive stiffness:

| grid | Δt advective | Δt code | headroom |
|---|---|---|---|
| 96² | 1.042e-6 | 2.324e-8 | **44.8×** |
| 160² | 6.250e-7 | 3.012e-9 | **207×** |
| 224² | 4.464e-7 | 7.840e-10 | **569×** |

So an integrator that removes only the angular stiffness buys 569× at 224²
before advection binds. Removing that too moves the limit to accuracy.

---

## 4. Finding A — the scheme is first-order in space, and it need not be

This is the headline, and it was not expected.

Convergence of the discrete `λmin` to `j₀,₁²`:

| M | λmin (as coded) | error | observed order |
|---|---|---|---|
| 16 | 5.288743163 | 4.944e-1 | |
| 32 | 5.545310388 | 2.379e-1 | 1.06 |
| 64 | 5.667273105 | 1.159e-1 | 1.04 |
| 128 | 5.726085413 | 5.710e-2 | 1.02 |
| 256 | 5.754863230 | 2.832e-2 | 1.01 |
| 512 | 5.769083265 | 1.410e-2 | 1.01 |

**Order 1.0.** A conservative finite-volume scheme on a smooth radial grid should
be order 2. The lost order is being paid for by refinement at `G⁶` cost.

### Cause: two half-cell errors, and both must be fixed

Isolating on an annulus `[0.5, 1]` — no pole, no central patch — and testing the
two suspects independently:

| volume weight | Dirichlet placement | observed order |
|---|---|---|
| inner face (as coded) | next cell centre (as coded) | 1.00 |
| inner face | physical face | 0.99 |
| centroid | next cell centre | 1.00 |
| **centroid** | **physical face** | **2.00** |

All four extrapolate to the same limit (≈ 39.0133), so they solve the same
problem — they differ only in order. The two defects are:

1. **Volume weight.** Cell `m` is the annulus `[(m+1)Δr, (m+2)Δr]`. Its face
   radii `(m+1)Δr` and `(m+2)Δr` are used correctly, but the divisor is
   `(m+1)Δr` — the *inner face* — where the annulus centroid is `(m+1.5)Δr`.
2. **Dirichlet placement.** `j_r_r` sets the neighbour to zero at `m = M-1`,
   putting `φ = 0` at the next cell *centre*, one full `Δr` beyond the last cell,
   rather than at the physical rim.

Neither fix alone recovers second order — which is why an obvious single-line
"fix" would have looked like it did nothing. Both together give exactly 2.00.

### The pole is *not* the problem

Testing the full disc with interior and boundary corrected, under three central-
patch treatments:

| central-patch flux distance | order | error at M=256 |
|---|---|---|
| `Δr` (as coded) | 2.00 | 2.760e-5 |
| `1.5Δr` (centroid separation) | 2.00 | 1.901e-4 |
| `1.5Δr − Δr/√2` (smooth-average) | 2.00 | 3.968e-5 |

All second order, and the *as-coded* distance has the smallest constant. The
central patch needs no change. This is worth knowing because the pole is the
part everyone expects to be wrong.

### What it is worth

At M = 256 the error falls from **2.832e-2 to 2.760e-5** — a factor of **1026 at
the same grid**, for a change confined to a weight and a boundary coefficient.

Turned around: work scales as `G⁶`, so at fixed accuracy a second-order scheme
runs on a much coarser grid. Being deliberately conservative and keeping a grid
that still resolves 16 microtubules and the central initial condition:

| target | grid needed, 1st order | grid needed, 2nd order | work ratio |
|---|---|---|---|
| accuracy of today's 224² | 224² | ~96² (conservative) | **161×** |

The naive extrapolation says ~16², but that is below the geometric floor — the
grid must still resolve the tube spacing and the initial patch, independent of
truncation error. **96² is the defensible number**; anything smaller needs a
resolution argument, not an accuracy argument.

---

## 5. Finding B — what to do about the timestep

Ranked by payoff-to-risk. Speedups are relative to today's explicit scheme at the
same accuracy.

### B1. Implicit / ADI in θ only (recommended first)

The angular operator at fixed `m` is a periodic tridiagonal (cyclic tridiagonal)
matrix — a Sherman–Morrison correction to a Thomas solve, `O(N)` per ring with a
small constant. Treating it implicitly and everything else explicitly (IMEX)
removes exactly the term that binds.

* **Steps saved:** 569× at 224², 207× at 160², 45× at 96² (measured headroom,
  §3), until advection binds.
* **Cost per step:** one cyclic-tridiagonal solve per ring, ~8 flops/unknown
  against ~20 for the explicit stencil; expect 2–3× per step in practice.
* **Net:** ~190–280× at 224².
* **Risk:** low. The angular operator is constant in time and diagonalisable by
  FFT if preferred; the split is standard Strang or Douglas.

### B2. Full ADI (Peaceman–Rachford / Douglas), both directions

Unconditionally stable, second order in time. `Δt` then set by accuracy, not
stability.

* **Steps:** with a solution timescale ~0.3 and a 1e-4 relative target,
  `Δt ~ 3e-3` → `K ~ 340`. Against `K = 1.275e9` today.
* **Cost per step:** ~4 tridiagonal sweeps, 5–10× the explicit step.
* **Net:** 10⁵–10⁶×, but see the caveat below.
* **Risk:** medium. Advection at `v = 10⁴` must go implicit too or it reintroduces
  a CFL; the two-layer coupling is a source term that wants an IMEX treatment.
  **The `10⁶` figure is a ceiling, not a forecast** — it assumes the coupling and
  the microtubule source term integrate cleanly at that `Δt`, which is untested.

### B3. Spectral in θ (FFT)

θ is periodic and uniformly sampled, so an FFT diagonalises the angular
Laplacian exactly: mode `k` sees `−k²/r²`. Each mode then evolves independently
in `r`, and the stiff `−k²/r²` becomes a scalar that an exponential integrator
handles exactly.

* **Attraction:** removes angular truncation error entirely (spectral accuracy in
  θ), so the angular grid can be much coarser.
* **Obstacle:** the microtubule coupling is **localised in θ** — a sum of
  delta-like sources on `N_LIST` rays. That is dense in Fourier space and
  couples all modes. This is why the existing `modal_model` needs a
  constraint-elimination least-squares step for `B`.
* **Verdict:** attractive for the pure-diffusion sector, awkward for the coupled
  problem. Not first.

### B4. Ring-wise subcycling (local time stepping)

Ring `m` tolerates `Δt ∝ (m+1)²`. Subcycle only the inner rings.

* **Payoff:** the number of ring-updates falls from `M·K` to `≈ K·Σ 1/(m+1)²·M`
  → a factor approaching `M²/ζ(2)`-ish in the ideal case, but conservation across
  differing timesteps needs care.
* **Verdict:** large potential, high correctness risk (mass conservation across
  the interfaces). Only worth it if B1/B2 disappoint.

### B5. Raise the safety factor

`compute_dT` uses `0.1 ×` the stability limit; measured, the true limit is
`2/‖λ‖max` and the code sits at **0.101×** of it. A factor `0.4` would still be
comfortably stable for a linear problem and is a **4× free speedup** with a
one-character change.

* **Risk:** low but not zero — the coupling and the advective term are not in the
  measured spectrum. Should be validated by a stability sweep, not assumed.

---

## 6. Finding C — implementation headroom is small, and not where expected

Microbenchmarks of the φ stencil at 96² (isolated from the ρ update and the
reductions):

| variant | µs/step | ns/patch | vs as-coded |
|---|---|---|---|
| as coded | 72.43 | 7.86 | 1.00× |
| hoisted per-ring reciprocals | 73.89 | 8.02 | **0.98×** |
| + branch-free microtubule test | 36.01 | 3.91 | **2.01×** |
| + `fastmath` | 28.11 | 3.05 | 2.58× |
| `parallel=True` (12 threads) | 45.36 | 4.92 | 1.60× (slower than serial) |
| pure copy (memory roofline) | 2.84 | 0.31 | 25.5× |

Three corrections to what one would guess:

* **The kernel is not division-bound.** Hoisting the per-ring reciprocals gives
  *nothing* (0.98×) — LLVM already hoists them out of the inner loop. This was my
  first hypothesis and the measurement killed it.
* **The microtubule branch is the cost.** `if n == tube_placements[mt_pos]`
  blocks vectorisation; replacing it with an arithmetic mask is worth **2.01×**.
* **`parallel=True` is slower at this size.** 9,216 patches is too little work to
  amortise thread dispatch. It would pay only at 224²+.

Note also that the isolated stencil runs at 7.86 ns/patch while the *full* step
measures 27.8 ns/patch — so the φ stencil is only ~28% of a timestep. The ρ
update, the three reductions and the centre update are the other ~72% and would
need the same treatment for the 2× to reach the whole solve.

---

## 7. What "agreement to O(ε)" can and cannot mean

This needs stating precisely, because the acceptance test differs by class and
one of the three cannot meet a tight tolerance *by construction*.

**Class 1 — same discretisation, different execution.**
Branch removal, reciprocal hoisting, vectorisation, GPU. Measured agreement of
the variants in §6 against the as-coded arithmetic: **2.1e-16**, i.e. 1 ulp. Where
the operation order is preserved exactly, agreement is bit-identical.
*Acceptance: ≤ 4 ulp, or bit-identity where claimed.*

**Class 2 — same spatial operator, different time integrator.**
B1, B2, B4, B5. Both schemes converge to the same semi-discrete ODE solution as
`Δt → 0`, so they can be made to agree to any tolerance — bounded below by the
*existing* scheme's own temporal error, which is
`≈ λ²ΔtT/2 = 5.783² × 7.84e-10 × 1/2 ≈ **1.3e-8**` at 224².
*Acceptance: agreement to ~1e-8; below that the current scheme is the limiting
party, not the new one.*

**Class 3 — different spatial discretisation.**
The second-order fix of §4. **This cannot agree with the current scheme to
O(ε), and should not be expected to.** It differs by the current scheme's
spatial error — `2.8e-2` at M = 256. Demanding tight agreement here would be
demanding that the fix not fix anything.
*Acceptance: (i) measured order 2.00 against `j₀,₁²`; (ii) closer to the
`analytic_benchmark` reference than the current scheme at every grid; (iii)
Richardson-extrapolated limits of old and new agree.*

That third class is the one with the payoff, and it needs the analytic benchmark
— not the current solver — as its reference.

---

## 8. Ranked plan

Each stage has a validation gate. Nothing proceeds past a failed gate.

**Stage 0 — consolidate the time loops.** ~15 drivers inline their own explicit-
Euler loop; the operator exists in three copies. Refactor to one operator and one
driver taking an integrator argument. *No numerical change; existing golden tests
must stay bit-identical.* This is a prerequisite, not an optimisation.

**Stage 1 — branch-free stencil.** §6, worth 2.01× on the stencil. Class 1;
gate: ≤ 4 ulp against the current output, golden tests re-baselined with the
measured drift recorded.

**Stage 2 — second-order spatial fix.** §4. Change the volume weight to the
centroid and place the Dirichlet condition on the rim. Class 3; gate: measured
order 2.00 against `j₀,₁²`, and improvement against `analytic_benchmark` at every
grid. **Highest payoff in the document (≈163× at fixed accuracy, 10³ at fixed
grid).**

**Stage 3 — raise the CFL safety factor** from 0.1 toward 0.4 (§B5), validated by
an empirical stability sweep including the coupling. 4×, one line.

**Stage 4 — IMEX with implicit θ.** §B1. ~190–280× at 224². Class 2; gate:
agreement to ~1e-8 against a current-scheme reference run at 96².

**Stage 5 — full ADI**, only if Stage 4 lands cleanly. §B2.

Cumulative, conservatively: Stage 2 (161×) × Stage 4 (190×) ≈ **3 × 10⁴** at
fixed accuracy, before Stage 1 and 3. The cost model puts a 224² characteristic-
time run at 694 CPU-hours (29 days); at 3 × 10⁴ the same accuracy costs about
**1.4 minutes**. At that point output writing and the numba compile dominate the
job, not the solve.

---

## 9. Memory

Two things are worth separating, because they differ by three orders of
magnitude.

| item | at 224², T = 1 | scaling |
|---|---|---|
| φ and ρ fields (2 slots × 2 layers) | **1.6 MB** | `32·M·N` bytes |
| mass timeseries, `MA_collection_factor = 50` | **1.02 GB** | `(K/MA)·40` bytes |
| same, `MA = 5` (the default) | **10.2 GB** | — |

**The fields are negligible; the timeseries are the entire memory problem**, and
they scale with `K`, i.e. with `1/Δt`. Every timestep improvement in §5 reduces
memory in exact proportion. At 96² with implicit-θ (`K ≈ 9.6e5`), sampling
*every* step costs 38 MB — so `MA_collection_factor` stops being a memory
parameter at all, and the 1.02 GB host-plus-device copy that forced `MA = 50` on
the GPU disappears.

Two independent wins, neither large in absolute terms:

* **ρ is stored dense** as `(2, M, N)` but only `|N_LIST|` rays are ever
  non-zero — a **14× waste** at `N = 224, 16` tubes (803 KB → 57 KB). Worth doing
  when the operator is touched anyway, mainly for cache behaviour rather than
  capacity.
* **The timeseries should stay float64.** This used to read "could be
  float32", when they fed a two-point log-linear fit insensitive at 1e-7. t*
  is now located from a second difference of ln M at a step of ~1e-4 (see
  `_char_time_onset` in `launch.py`). Rounding in that difference is about
  eps*|ln M|/h^2: ~1e-7 in float64, but ~30 in float32 -- far above the
  default tau of 1e-3.

---

## 10. Would a lower-level language be necessary?

**No — and the measurements say so fairly clearly.**

| path | measured/estimated gain |
|---|---|
| numba, as coded | 7.86 ns/patch (stencil, 96²) |
| numba, branch-free + fastmath | **3.05 ns/patch** (measured) |
| pure memory copy | 0.31 ns/patch (measured roofline) |
| C/C++ with hand-written AVX | ~1–1.5 ns/patch (estimated) |

numba already reaches within **9.8×** of the copy roofline once the branch is
removed, and the remaining gap is arithmetic intensity, not language. A C or
Fortran rewrite might buy another **2–3×**. Against §8's ~3 × 10⁴ from the
algorithm, that is noise — and it would be spent on a codebase that currently
holds the operator in three places and the integrator in fifteen.

Three qualifications, so this is not read as "never":

* **CUDA C would help the GPU path**, modestly — shared-memory tiling and warp
  reductions could plausibly beat the current numba.cuda kernel by 1.5–3×,
  chiefly by removing the serial per-block combine already flagged in
  `GPU_PLAN.md`. Still second to the algorithm.
* **The tridiagonal solves of Stage 4/5 are where libraries matter.** Rather than
  writing a solver, use `scipy.linalg.solve_banded` on CPU and cuSPARSE
  (`gtsv2StridedBatch`) on GPU — both are C already. This is the one place to
  reach outside Python, and it is a library call, not a rewrite.
* **Revisit after Stage 4.** Once `K` drops by 10⁴, per-step cost is a much
  larger share of runtime, and the case for hand-optimisation gets stronger. It
  is not stronger now.

---

## 11. Risks and open questions

**The order-2 fix changes published numbers.** Every result in
`results/2026-08-18_char-time-grid-study_N16/` was produced by the first-order
scheme. A corrected scheme will not reproduce them, and should not be expected to
(§7, class 3). The grid study would need re-running — cheap once Stage 2 and 4
land, but it must be a deliberate decision, and old results should be labelled
with the scheme that produced them.

**It may explain the m\* non-convergence.** That study found `m* ~ G^-1.204` with
no convergence. A first-order scheme does not by itself produce that, but the
interaction of an `O(Δr)` boundary misplacement with a quantity evaluated near
the rim is worth checking before concluding the effect is physical. **This is a
concrete, cheap test: re-run three grids under Stage 2 and see whether the
exponent moves.**

**`D` never reaches the spatial operator.** It appears only in `compute_dT`,
`compute_K` and `convert_K_to_T`; no stencil function takes it. So changing `D`
alters the timestep and the cost but not the equation being solved. Either the
model is non-dimensionalised to `D = 1` — in which case dividing `Δt` by `2D` is
wrong — or the operator is missing a factor. **This needs settling against the
derivation before any of the above, since it changes what the stable timestep
is.**

**The spectrum omits the coupling.** §3 assembled the diffusive layer only. The
microtubule source and the advective layer are a rank-`2M|N_LIST|` perturbation
and do not set the stiff end, but the stability *constant* under coupling was not
measured — which is exactly why Stage 3 needs an empirical sweep rather than a
calculation.

**The 96² figure in §4 is a judgement, not a measurement.** It comes from
requiring the grid to resolve the microtubule geometry, not from the error curve.
If a resolution study shows 64² suffices, the 161× becomes 1,838×.

---

## 12. Reproducing these measurements

The four scripts are throwaway and not committed. Each is short; what matters is
the construction:

* **Spectrum.** Transcribe `u_density` + `j_*` + `u_center` into a sparse matrix
  over unknowns `[central, φ(m,n) row-major]`, with radial coefficient
  `1/((m+1)Δr²)` and face weights `(m+2)`, `(m+1)`; angular coefficient
  `1/((m+1)²Δr²Δθ²)`; centre row `(Δθ/(πΔr²))(Σₙ φ₀ₙ − Nφ_c)`. Then
  `eigs(L, k=1, which="LM")` and `eigs(L, k=1, sigma=0)`.
* **Order.** Restrict to `n = 0` (angular terms vanish) — an `(M+1)`-square dense
  problem — and compare `λmin` against `scipy.special.jn_zeros(0,1)[0]**2`.
* **Localisation.** Same radial operator on `[R1, R2]` with Dirichlet ends,
  toggling volume weight (inner face vs centroid) and Dirichlet placement (next
  centre vs face, i.e. coefficient `1` vs `2`). Estimate order by Richardson,
  `p = log2((Lₕ − L_{h/2})/(L_{h/2} − L_{h/4}))`, so no exact value is needed.
* **Kernel.** Time the φ stencil alone under numba with `cache=True`, against a
  `phi[1] = phi[0]` copy loop over the same footprint for the roofline. Compare
  outputs with `max|a−b|/|a|` to confirm the ~1 ulp claim.

Grid conventions throughout: cell centres `r_m = (m+1)Δr`, `Δr = R/M`,
`Δθ = 2π/N`, absorbing rim, unit mass in the central patch.
