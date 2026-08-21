# GPU acceleration — assessment and plan

Measured on the project workstation (RTX 3060, 12 GB, cc 8.6, 28 SMs) against
the current CPU solver, August 2026. Every figure below was taken on that
machine; none is extrapolated from vendor claims.

> **Superseded in two places by the implementation.** The port now exists
> (`src/intracellular_transport/gpu/`), and measuring it corrected the estimates
> below. Read [Measured after implementation](#measured-after-implementation)
> first — the projections in the next section were taken with the reductions
> left out of the kernel, and are therefore optimistic.

## Measured after implementation

Two corrections, both from measuring the real kernel rather than a stencil-only
proxy.

**The speedup is 9× at 96², not 21×, and it grows with grid size.** The earlier
figures timed the stencil alone. Adding the mass and centre reductions costs a
roughly constant 12-16 µs per step — two extra grid-wide barriers and a serial
combine of the per-block partials — which dominates at small grids and amortises
at large ones:

| grid | GPU µs/step (with reductions) | ns/patch | CPU µs/step | speedup | GPU wall (T=1) |
|---|---|---|---|---|---|
| 48² | 27.2 | 11.8 | 64 | 2.4× | 1.2 min |
| 96² | 28.0 | 3.03 | 256 | 9.2× | 20 min |
| 128² | 34.1 | 2.08 | 455 | 13.4× | 1.3 hr |
| 160² | 43.5 | 1.70 | 712 | 16.4× | 4.0 hr |

CPU figures use the uncontended 27.8 ns/patch anchor. GPU figures exclude
one-off JIT compilation. The practical conclusion is unchanged and if anything
sharper: the GPU is not worth using below about 96², and is transformative above
128², where it turns days into hours.

The serial combine on a single thread is the obvious next optimisation — it is a
fixed cost per step, so removing it would lift the small-grid figures most.

**Numerical agreement is far better than predicted, and does not compound.**
This document estimated ~1e-10 relative agreement. Measured worst case across
32², 48² and 64² is **5.9e-14**, and critically the error does *not* grow with
step count:

| steps (K) | φ relative error |
|---|---|
| 10,624 | 1.8e-15 |
| 26,560 | 1.6e-15 |
| 53,121 | 7.4e-15 |
| 106,242 | 5.1e-15 |
| 212,485 | 3.7e-15 |

Over a 20× range in step count the disagreement stays at 1e-15 to 1e-14 and does
not trend upward. The scheme is diffusive and therefore contracting, so
round-off differences are damped rather than accumulated. That is what makes
extrapolation to a 332M-step run credible — though it remains an extrapolation:
the longest run compared directly is 212k steps.

Repeated GPU runs are **bit-identical to each other**, because the block count
and reduction order are fixed. So a GPU result is exactly reproducible on the
same machine even though it is not exactly equal to the CPU's.

`tests/test_gpu_agreement.py` holds the tolerance as a single constant and prints
the measured disagreement, so the acceptance decision can be made against
numbers rather than estimates.

## Summary

A GPU port is worth roughly **20-25×** across the grid sizes of interest — but
only if it is built as a *persistent kernel*. The obvious implementation, one
kernel launch per timestep, is **slower than the CPU** at the grids we actually
run.

| grid | K (T=1) | CPU wall | GPU wall | speedup |
|---|---|---|---|---|
| 96² | 43.0 M | 3.1 hr | **9 min** | 21× |
| 128² | 136.0 M | 17.2 hr | **50 min** | 21× |
| 160² | 332.0 M | 2.7 d | **2.9 hr** | 22× |
| 176² | 486.1 M | 4.9 d | **6.7 hr** | 17× |
| 192² | 688.5 M | 8.2 d | **9.2 hr** | 21× |
| 256² | 2.18 B | 45.9 d | **2.3 d** | 20× |

CPU costs ~27.8 ns per patch per step; the persistent GPU kernel costs ~1.33 ns.
Both scale linearly in patch count, which is why the ratio is roughly flat rather
than improving with grid size.

The practical consequence is not that existing runs get faster — it is that
192² becomes an overnight job instead of an eight-day one, and 256² becomes
possible at all.

## Two earlier estimates in this project were wrong

Worth recording, because both errors are easy to repeat.

**"30-90×"** — an early guess that assumed 5-15 µs of kernel launch overhead.
Measured launch overhead on this machine (Windows, WDDM) is 30 µs for an empty
kernel, and far higher with arguments. The guess was not grounded.

**"8.2×"** — a later figure that *was* measured, but against the CPU solver
before the 3.4× stencil optimisation, and using a per-launch port. Both halves
of the comparison were stale.

The lesson is that a GPU estimate is only meaningful against a specific CPU
baseline and a specific launch strategy. State both.

## Why the naive port loses

Measured per-step cost at 96², same physics in each case:

| approach | µs/step |
|---|---|
| two kernels per step (diffusive + advective) | 277 |
| one fused kernel per step | 187 |
| **persistent kernel, 2000 steps per launch** | **12.35** |

For comparison, the current CPU solver is **296 µs/step** at 96² — so the
per-launch versions are only marginally better than CPU, and at 48² and 64² they
are *worse*.

The reason is not the GPU. An **empty kernel taking 17 arguments costs 184 µs**
to launch, against 30 µs for an empty kernel with none:

| kernel | µs/launch |
|---|---|
| no arguments | 29.97 |
| 4 array arguments | 101.28 |
| 17 mixed arguments | 183.98 |

The fused kernel measured 187 µs and its empty 17-argument equivalent measured
184 µs. The GPU was completing the physics in roughly 3 µs and spending the rest
of the time in numba's host-side argument marshalling. Reducing the argument
count helps a little; it does not fix the problem.

## What does fix it: one launch, many steps

`cuda.cg.this_grid().sync()` provides a grid-wide barrier inside a kernel, so a
single cooperative launch can run thousands of timesteps, synchronising between
them. The marshalling cost is then paid once per chunk instead of once per step.

Confirmed available on the installed numba (0.57.1) and working on this device.
With a trivial kernel it reaches 1.48 µs/step; with the real per-patch physics it
reaches the 12-92 µs/step in the table above, at which point it is
**compute-bound rather than overhead-bound** — the tell is that cost per patch
is constant (1.25-1.60 ns) across every grid size measured.

### Constraint: cooperative launches cap the block count

Cooperative launch failed above **112 blocks** on this device (28 SMs × 4);
224 raised a `CudaAPIError`. This does not limit grid size, because a
grid-stride loop lets a fixed block count cover any number of patches — but it
does cap usable parallelism at ~14,336 threads. That is why 256² costs 92 µs
rather than four times the 128² figure: each thread is handling about 4.6
patches instead of one.

## Where the speedup is real, and where there is none

**Real, and it is the whole game:** the time-stepping loop. It is over 98% of
runtime for any long computation — profiling put the stencil alone at 98% — and
every one of the 18 registered computations goes through it. Porting that single
loop benefits all of them.

**No benefit:** figure rendering, CSV writing, the log-linear characteristic-time
fit, `analytic_benchmark`, and the modal model. Together these are a rounding
error against a multi-hour solve.

**Not worth it below about 96².** A 48² run already finishes in three minutes;
setup and transfer would not be repaid.

## Numerical agreement: bit-identical is not achievable

This needs stating plainly, because the CPU work in this project has held itself
to a bit-identity standard and the GPU cannot meet it.

* The **per-patch stencil can** be bit-identical. IEEE-754 double arithmetic is
  deterministic, so the same operations applied in the same order produce the
  same bits on either device.
* The **reductions cannot.** `calc_mass_diff`, `calc_mass_adv` and `u_center`
  are sequential accumulations on the CPU. On a GPU they become tree reductions,
  and floating-point addition is not associative — a different summation order
  gives a different result in the last bits.

Those differences are ~1e-16 relative per reduction and compound over hundreds
of millions of steps. The scheme is diffusive and stable, so this will not
diverge, but `t*` is extracted from a log-linear fit and will differ in its
lower digits.

**Expect agreement to something like 1e-10 relative, not to the 17 significant
figures the CPU currently reproduces.** Two runs in the August 2026 grid study
matched their earlier values exactly; GPU results would not have.

The acceptance criterion therefore has to change from "bit-identical" to "agrees
within a stated tolerance", and *what tolerance is scientifically acceptable is a
decision for the researcher, not an implementation detail.* This should be
settled before the port is written, not after.

## Should this be a separate package?

Yes, but a deliberately thin one:

```
src/intracellular_transport/gpu/
    persistent_kernel.py   the cooperative kernel and its device functions
    driver.py              same interface as the CPU driver
```

And explicitly **not**:

* a parallel GPU port of the eleven njit loops in `analysis_tools.py`. That is
  precisely how this repository previously ended up with five divergent copies
  of the solver and four different `u_density` implementations. Port the one
  inner loop that matters.
* a separate analysis or output path. Setup, the fit, and all CSV/PNG writing
  stay shared.

The CPU implementation remains the reference and the authority on correctness.
Device selection is a parameter, defaulting to CPU.

## Sequencing: checkpointing first

The chunked driver that checkpointing requires — run N steps, return, persist a
small amount of state, repeat — is *the same structure* a persistent GPU kernel
needs. Building checkpointing first means the GPU path drops into an existing
seam. Building the GPU first means building that scaffolding twice and then
reconciling them.

Recommended order:

1. **Checkpointing** (see the plan in this directory): chunk the driver, kernel
   returns its carried scalars, timeseries move to `np.memmap`.
2. **Decide the numerical tolerance** for accepting GPU results.
3. **Port the inner loop** as a persistent kernel behind the same driver
   interface, with the reductions kept on the device.
4. **Validate** against CPU at 48² and 96² within the agreed tolerance, and
   re-derive `t*` for a configuration whose CPU value is already known.
5. Extend to the grids that were previously out of reach.

Step 4 is the one to not rush: a GPU result that is subtly wrong is worse than no
GPU result, and the reduction-ordering differences mean the usual bit-identity
safety net does not apply.

## Reproducing these measurements

The benchmarks behind this document are not committed — they are throwaway
scripts. Reconstructing them needs three ingredients:

* time `num.comp_DL_AL_kp1_2step` plus the per-step extras (`calc_mass_diff`,
  `calc_mass_adv`, `u_center`, the two layer copies) for the CPU figure;
* for the GPU figure, a `@cuda.jit` kernel taking `cuda.cg.this_grid()`, looping
  `nsteps` internally with `g.sync()` at the end of each step, ping-ponging
  through a leading axis of length 2, launched with 112 blocks × 128 threads;
* precompute microtubule membership as a dense array. An early version looped
  over `N_LIST` inside every patch, which alone accounted for the difference
  between 277 µs and 187 µs per step.

Note that CPU figures taken while other solver jobs are running carry
memory-bandwidth contention; the numbers above use an uncontended 96² anchor of
256.5 µs/step and scale by patch count.
