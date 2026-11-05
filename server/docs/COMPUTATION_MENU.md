# Computation menu and how to launch jobs

Every computation is launched the same way: `POST /jobs` with a computation name
and a parameter object. The call returns immediately with a job id — it does not
wait for the solver. A worker on the server picks the job up and runs it, up to
4 concurrently.

Set up access first: [`CONNECTING.md`](CONNECTING.md).

```bash
export ITCM=http://100.83.174.69:8000
```

---

## Launching a job

```bash
curl -s -X POST $ITCM/jobs -H 'Content-Type: application/json' -d '{
  "computation": "Characteristic Time (mass vs v)",
  "params": {
    "rg_param": 96, "ry_param": 96,
    "v_LIST": [10000], "w_param": 100, "T_param": 1,
    "N_LIST": [0, 4, 8, 12],
    "show_plt": false
  },
  "submitted_by": "your-name"
}'
```

```json
{"job_id": "aa3a237f1b2c...", "status": "queued"}
```

Keep the `job_id`. Omitted optional parameters take the defaults listed per
computation below. Send real JSON types — `96` and `[0, 4, 8]`, not `"96"` and
`"[0, 4, 8]"`.

Always pass `"show_plt": false`. It is a desktop-GUI setting that tries to open
a plot window; on a headless server it does nothing useful.

### Choosing a device (`Mass Analysis`, `Characteristic Time (mass vs v)`, `Characteristic Time (a,b grid)`)

These three computations accept `"device"`, which selects where the time-stepping
loop runs. It defaults to `"cpu"`, so existing job definitions are unaffected.

| value | behaviour |
|---|---|
| `"cpu"` | Reference implementation. Never touches CUDA. The default. |
| `"gpu"` | Fails the job if the GPU cannot run this configuration, rather than quietly using the CPU. |
| `"auto"` | Prefers the GPU, falls back to the CPU and logs why. |

The GPU is worth using at **96×96 and above**, where it is roughly 9× faster,
rising to ~16× at 160×160. Below 96×96 the per-step reduction overhead dominates
and the gain is small.

**GPU results are not bit-identical to CPU results.** The mass and centre
reductions are parallel tree reductions on the GPU and sequential accumulations
on the CPU, and floating-point addition is not associative. Measured agreement is
**~1e-14 relative** on `t*` and `m*`, and the error does not grow with step
count. Repeated GPU runs are bit-identical *to each other*. If you need results
directly comparable to a previously published CPU number at full precision, use
`"cpu"`.

The GPU path does not support `d_tube != 0` or off-centre initial conditions.
Requesting `"gpu"` with either is rejected at submission with a 400.

For all three, `device` only matters with `"engine": "time_stepper"` (the
default for `Mass Analysis`). The `"modal"` engine (the default for the two
characteristic-time computations) has no time-stepping loop to place (see [Choosing an engine](#choosing-an-engine-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid)),
so `device` is accepted and ignored there.

### Choosing an engine (`Mass Analysis`, `Characteristic Time (mass vs v)`, `Characteristic Time (a,b grid)`)

| value | behaviour |
|---|---|
| `"modal"` | **The default for the characteristic time.** Builds the solver's own one-step operator, diagonalises it once and reads M(t) off the modal sum M(t) = Σ aₙ e^(−λₙ t). t* is the right-sweep rule applied to the **exact** y'' of that sum (no differencing), with the crossing of \|y''\| = tau root-found rather than snapped to a sample. Also returns `lambda_1` and `lambda_2`. Seconds per point **whatever `T_param` is**. |
| `"time_stepper"` | **The default for `Mass Analysis`.** The full solve to `T_param`, as before 2026-11-04; t* from the central difference of the sampled series, at sample resolution. The only engine that uses `device`, `mass_checkpoint` and `extend_from`. |

The two agree on `t*` to within **1.0e-5** and on `m*` to **6e-5** relative on
every 96×96 pair with a finite t* at `tau = 0.01`, and the underlying M(t) agrees
to round-off (`results/2026-11-04_modal-mass-analysis-validation/`). Numbers are
not bit-identical across engines: compare like with like.

For `Mass Analysis`, `"modal"` fills the same five series at the same samples
from the modal sum, and writes the same files. Its worst total-mass difference
from the time-stepper is 1.8e-10 at 64×64 and 1.7e-9 at 96×96, and grows as
about G^4.45 — the time-stepper's own accumulated round-off
(`results/2026-11-04_modal-mass-analysis-validation/`). There is no time loop,
but the series still has `T_param / (5 dT)` samples to evaluate and write, so its
cost does grow with `T_param`, though far more slowly than a solve does.

With `"modal"`, `extend_from` is refused at submission (400) — the cost does not
depend on T, so just submit the longer T — and `workers` is ignored: the points
run one after another in the job's process, the eigensolve itself multithreaded.

### The PSD of y'' (`compute_psd`)

`Mass Analysis` and both characteristic-time computations accept
`"compute_psd": true`. With it they also compute

    P(ω) = | H Σₙ y''(tₙ) e^(−iωtₙ) |²,   y = ln(total mass),   tₙ = nH,

the squared magnitude of the Fourier transform of `y''` over `[0, T_param]`, with
no window and no detrending — the convention of
`results/2026-10-07_ab-grid-96x96_tau0.01/scripts/psd_96.py`. `y''` is the
central difference of `ln M` at `H ≈ 1e-4` on either engine. The modal engine
could give `y''` exactly, but its early pulse is narrower than `H`, so point
samples of it would misweight the low frequencies where the peak sits. The
record is zero-padded 8×, so the peak and the crossing are not snapped to the
`2π/T` bins (this adds no resolution).

The figure plots `P` on **linear axes, zoomed on the peak**. It reports:

- `psd_omega_peak`, `psd_peak` — where the largest `P` is (ω = 0 included), and its value.
- `psd_fwhm` — **2 × the half-width on the left of the peak**, `2 (ω_peak − ω_left)`
  with `P(ω_left) = P_peak / 2`. The right flank is not used, because leakage
  from other modes can widen it.
- `psd_fwhm_side` — `left` normally. It is `mirrored` when `P` stays above
  half-maximum all the way to ω = 0; `P` is even in ω, so the crossing is then
  read on the negative side (a peak at ω = 0 is FWHM = 2 × its right crossing).

### Checking on it

```bash
curl -s $ITCM/jobs/<job_id>            # full record; accepts an 8-char prefix
curl -s $ITCM/jobs?status=running      # everything currently running
```

`status` moves `queued` → `running` → `succeeded` | `failed`. On failure, the
`error` field carries the solver's own message. Scalar results (an MFPT value,
a duration) appear in the `result` field; everything else writes files.

### Collecting results

```bash
curl -s $ITCM/jobs/<job_id>/outputs
```

```json
{"job_id": "...", "status": "succeeded",
 "files": [{"path": "char_time_analysis/2026-08-18_11-11/char_t_analysis_data.csv", "bytes": 64},
           {"path": "char_time_analysis/2026-08-18_11-11/char_t_analysis_plot.png", "bytes": 14750}]}
```

Download one file, or everything (the `tr -d '\r'` keeps the loop working under
Git Bash on Windows, whose `python3` emits CRLF; harmless on macOS and Linux):

```bash
curl -s -o data.csv "$ITCM/jobs/<job_id>/outputs/char_time_analysis/2026-08-18_11-11/char_t_analysis_data.csv"

curl -s $ITCM/jobs/<job_id>/outputs \
  | python3 -c 'import sys,json;[print(f["path"]) for f in json.load(sys.stdin)["files"]]' \
  | tr -d '\r' \
  | while read p; do mkdir -p "$(dirname "$p")"; curl -s -o "$p" "$ITCM/jobs/<job_id>/outputs/$p"; done
```

### Cancelling

```bash
curl -s -X DELETE $ITCM/jobs/<job_id>
```

Only works while `queued`; a running job returns `409`. There is no way to stop
a running solver through the API — ask the admin.

### Discovering parameters live

The server is the authority; this document can drift from it.

```bash
curl -s $ITCM/computations
curl -s "$ITCM/computations/Characteristic%20Time%20(mass%20vs%20v)"
```

The schema endpoint returns required parameters, optional parameters with their
defaults, and a one-line hint for each.

---

## Cost: read this before submitting a large grid

Everything in this section is about the time-stepping loop. The characteristic-time
computations on their default `"modal"` engine do not run it: their cost is an
eigensolve per point, independent of `T_param` — seconds at 96×96 (measured
figures in `server/estimate.py`, `_MODAL_POINTS`).

Runtime is governed by the scheme's stability limit,
`dT = 0.1 · dTheta² · dRad² / (2D)`, which gives `K ∝ G⁴` timesteps and
therefore **total work ∝ G⁶** for a `G × G` grid. Doubling the grid is roughly
**64× the work**, not 4×.

Wall time per job at `T = 1` (one solver core):

| Grid | Timesteps (T=1) | Wall time | |
|---|---|---|---|
| 16 × 16 | 33,200 | ~0.7 min | measured |
| 32 × 32 | 531,214 | ~1.0 min | measured |
| 48 × 48 | 2,689,274 | ~3.8 min | measured |
| 96 × 96 | 43,028,399 | ~3.3 hr | measured |
| 128 × 128 | 135,990,990 | ~18 hr | extrapolated |
| 192 × 192 | 688,454,390 | ~8.7 days | extrapolated |

Grids at or below 32 × 32 are dominated by numba JIT compilation (~20-30 s of
each run), not by the solve, which is why they do not fall off as steeply as
`G⁶` would suggest. From 48 × 48 upward the `G⁶` scaling holds and the
extrapolations are reliable.

Cost scales linearly in `T_param`, so `T = 2` doubles these. Four jobs run
concurrently at near-full speed (measured 3.90× throughput on 4 slots), so a
four-configuration sweep costs about the same wall time as one job.

**Jobs submitted through the server are checkpointed.** State is persisted
roughly every 30 seconds, so a reboot or a crash resumes from the last boundary
rather than restarting. Resume only happens for an identical parameter set — a
changed configuration starts fresh instead of silently continuing the wrong
trajectory. Runs launched directly from a script or the desktop GUI are *not*
checkpointed; that path is unchanged.

Prefer 48×48 while exploring, and reserve 96×96 and above for runs you actually
need. At 96×96 and above, `"device": "gpu"` cuts these times by roughly 9-16×.

---

## Shared parameters

Most computations draw on the same set. Only the ones specific to a computation
are repeated in its entry below.

### Domain and physics

| Parameter | Type | Meaning |
|---|---|---|
| `rg_param` | int | Radial rings in the domain (M) |
| `ry_param` | int | Angular rays in the domain (N) |
| `N_LIST` | list[int] | Microtubule angular positions, each in `[0, ry_param-1]`. E.g. `[0,4,8,12]` for 4 evenly spaced tubes |
| `v_param` | float | Particle velocity along the advective layer. Negative is inward |
| `w_param` | float | Mutual switch rate between diffusive and advective layers (sets `a = b = w`) |
| `a_param` / `b_param` | float | The two switch rates set separately: `a` onto the diffusive layer, `b` onto the advective layer. Taken instead of `w_param` by `Mass Analysis`; passing the same value for both is the `w` case |
| `T_param` | float | Dimensionless solution duration |
| `d_tube` | float | Advective-to-diffusive extraction width around each microtubule. `0.0` = extraction only on the tube ray. **Must lie in `[0, max]`**, where `max = (j + 0.5)·(1/rg_param)·(2π/ry_param)` and `j` is half the tightest gap between neighbouring tubes in `N_LIST`, so extraction regions cannot overlap. The solver silently substitutes `max` for anything outside that range, so `POST /jobs` refuses it instead; `POST /helpers/d_tube` with `{"params": {rg_param, ry_param, N_LIST, d_tube}}` reports the verdict and `max_d_tube` without submitting, and the UI shows it under the field |
| `domain_radius` | float | Domain radius. Default `1.0` |
| `D` | float | Diffusion coefficient. Default `1.0` |

### Initial condition

| Parameter | Type | Meaning |
|---|---|---|
| `center_init_cond` | bool | `true` (default) seeds unit mass in the central patch |
| `m_init` | int | Ring index for an off-centre start, `[0, rg_param-1]`. Used only when `center_init_cond` is `false` |
| `n_init` | int | Ray index for an off-centre start, `[0, ry_param-1]` |

### Collection and termination

| Parameter | Type | Meaning |
|---|---|---|
| `checkpoint_collect_container` | list[float] | Points at which to snapshot. **Time** stamps for time-dependent computations, **mass** fractions for mass-dependent ones |
| `mass_retention_threshold` | float | Mass fraction at which a mass-dependent run stops. Default `0.01` |
| `MA_collection_factor` | int | Timesteps between mass samples. Default `5` |
| `mass_checkpoint` | int | Timesteps between progress log lines. Default `1000000` |
| `T_fixed_ring_seg` | float | Fractional ring position for angular-dependence sampling. Default `0.5` |
| `R_fixed_angle` | int | Ray index for radial-dependence sampling. `-1` = auto |

### Output control

| Parameter | Type | Meaning |
|---|---|---|
| `save_png` | bool | Write figures alongside the CSVs. Default `true` |
| `show_plt` | bool | Open a desktop plot window. **Always pass `false`** |
| `heatplot_colorscheme` | str | Matplotlib colormap for heatmaps. Default `"viridis"` |
| `heatplot_border` / `heat_plot_border` | bool | Draw patch borders on heatmaps |
| `display_extraction` | bool | Mark extraction regions on heatmaps |

### Mass-dependent versus time-dependent

Many computations come in a pair. **Time-dependent** variants take `T_param` and
run to a fixed dimensionless time. **Mass-dependent** variants take
`mass_retention_threshold` and run until that fraction of mass remains — the
duration is an output rather than an input, so it is not known in advance.

---

## The 19 computations

### Scalar results (no files — read the `result` field)

#### `Compute MFPT until time T`
Mean first-passage time, integrating to a fixed `T_param`.
**Required:** `rg_param`, `ry_param`, `N_LIST`, `v_param`, `w_param`, `T_param`
**Output:** `result = {"MFPT": <float>}`, sometimes with `duration`.

#### `Compute MFPT until mass %`
Mean first-passage time, integrating until `mass_retention_threshold` of mass
remains rather than to a fixed time.
**Required:** `rg_param`, `ry_param`, `N_LIST`, `v_param`, `w_param`
**Output:** `result = {"MFPT": <float>, "duration": <float>}`.

#### `Time Until Mass Depletion`
Dimensionless time for mass to fall to `mass_threshold`.
**Required:** `rg_param`, `ry_param`, `N_LIST`, `v_param`, `w_param`
**Optional:** `mass_threshold` (default `0.01`)
**Output:** `result = {"duration": <float>}`.

---

### Characteristic time

#### `Characteristic Time (mass vs v)`
Sweeps a list of velocities. For each, finds the characteristic time `t*` —
the **onset of steady exponential decay** of the total mass — and reports the
retained mass `m*` at `char_time_mult · t*` (at `t*` itself by default).

`t*` is judged from the second derivative of `y = ln(total mass)`: the decay is
steady where `ln M` is a straight line, i.e. `y'' = 0`. Each sample is marked
`l(t) = 1` if `|y''(t)| ≤ tau` and `0` otherwise; **sweeping from the right**,
`t*` is the sample just after the last one with `l = 0`, so `|y''| ≤ tau` from
`t*` to the end of the run. `y''` is a second-order central difference taken at
a step of about 1e-4 in time (the double-precision rule of thumb), whatever the
grid's own sample spacing. That is the `"time_stepper"` engine; the default
`"modal"` engine applies the same right sweep to the exact `y''` of the modal sum
and root-finds the last crossing (see [Choosing an engine](#choosing-an-engine-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid)).
The fit columns in the CSV are a least-squares line through `log10(total mass)`
over `[t*, T]`.
**Required:** `rg_param`, `ry_param`, `v_LIST` (list[float]), `w_param`, `T_param`, `N_LIST`
**Optional:** `tau` (default `1e-3`) — the steady-decay threshold; smaller is
stricter and gives a later `t*`.
`engine` — `modal` (default) or `time_stepper`. See [Choosing an engine](#choosing-an-engine-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
`char_time_mult` (default `1`, range `[1, 10]`) — `m*` is `M(char_time_mult · t*)`;
`10` reproduces the pre-2026-09-28 practice of reading the mass at `10·t*`. The
modal engine evaluates M at any time, even past `T_param`; the time-stepper reads
the nearest sample and writes `m_star` as NaN past `T_param`.
`compute_psd` (default `false`) — also compute the power spectrum of `y''`; see [The PSD of y''](#the-psd-of-y-compute_psd).
`device` — `cpu` (default), `gpu`, or `auto`; time-stepper only. See [Choosing a device](#choosing-a-device-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
**Output:** `char_time_analysis/<timestamp>/`, and a job `result` of
`{"output_dirs": [...], "csv": ..., "tau": ..., "char_time_mult": ..., "engine": ...,
"t_star": [{"v", "t_star", "m_star", "lambda_1", "lambda_2", "lambda_2_imag"}, ...]}`
(each entry also carries the four `psd_` values with `compute_psd`)
— one entry per velocity, `t_star`/`m_star` `null` where no t* was found, the
`lambda` entries `null` from the time-stepper.
The desktop app prints the same table in its output panel.
- `char_t_analysis_data.csv` — one row per velocity: `v, t_star, m_star`, then
  `fit_slope, fit_intercept` (the line over the steady segment),
  `fit_window_t1, fit_window_t2` (that segment, `[t*, T]`),
  `total_mass_at_t1, total_mass_at_t2` (the mass at its ends; `total_mass_at_t1`
  is always `M(t*)`), `tau`, `char_time_mult`, and `lambda_1`, `lambda_2`,
  `lambda_2_imag` — the slowest decay rate of M and the slowest excited rate
  after it (real part, and the size of its imaginary part when it is one of a
  complex pair); empty from the time-stepper. With `compute_psd`, also
  `psd_omega_peak, psd_peak, psd_fwhm, psd_fwhm_side`
- `char_t_analysis_plot.png` — `m*` against `v`, log x-axis
- `char_t_psd_y2.png` — with `compute_psd`: one peak-zoom panel per velocity

> **`T_param` must leave room for the decay to settle.** The onset is only
> known after the solve. At `v = 1`, 16×16, `tau = 1e-3` it is about 0.5–0.7
> for fast switching onto the advective layer (`b = 100`), but 2.7–3.8 — or
> beyond `T = 4` — for `b ≤ 1`, where the decay rate oscillates before
> settling. A point whose `|y''|` is still above `tau` at the end of the run
> writes `t_star` and `m_star` as empty fields (NaN) with a warning, and the
> rest of the sweep is kept; raise `T` or `tau`. The study behind these numbers
> is in `results/2026-10-05_ab-grid-16x16_tau-mask/`.
>
> **This definition replaced two earlier ones**: a line through `log10 M` at
> `t = 0.4` and `t = 0.5` extrapolated to its intercept, with `m*` read at
> `10·t*` (before 2026-09-28), and a threshold on `|y''| / y'²` that had to hold
> for 0.1 (2026-09-28 to 2026-10-05). Numbers from different definitions are
> not comparable.
>
> On the time-stepper, one velocity per job is the usual choice: a `v_LIST` of
> five values costs five full solves sequentially inside one job, whereas five
> separate jobs run four at a time. On the modal engine a point costs seconds,
> so sweep freely.

#### `Characteristic Time (a,b grid)`
The same fit, swept over the two switch rates instead of over velocity. `a_list`
holds the rates onto the diffusive layer, `b_list` the rates onto the advective
layer, and **every pair in `a_list × b_list` is solved** — a 4×4 grid is sixteen
eigensolves (seconds each) on the modal engine, sixteen full solves on the
time-stepper. `v_param` is a single value here, not a list: the grid is already
two-dimensional.
**Required:** `rg_param`, `ry_param`, `a_list` (list[float]), `b_list` (list[float]), `v_param`, `T_param`, `N_LIST`
**Optional:** `tau` (default `1e-3`), `engine`, `char_time_mult`, `compute_psd`, as above.
`device` — `cpu` (default), `gpu`, or `auto`; time-stepper only. See [Choosing a device](#choosing-a-device-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
`workers` — time-stepper only: how many points to solve at once on the CPU; `0`
(default) uses one per core, less one.
**Output:** `ab_grid_char_time/<timestamp>/`, and a job `result` of the same
shape as above with `a`, `b` in each `t_star` entry, plus `device`.

Every filename ends in the parameters held fixed across the run — `v`, the
microtubule count and the grid — written as `v10000_N4_96x96` for the example
above. They are not columns, so a file lifted out of its timestamped directory
still says what produced it.

- `ab_grid_char_t_<stem>.csv` — one row per `(a, b)`: `a, b, t_star, m_star`, then the fit columns (`fit_slope`, `fit_intercept`, the window, and the two masses it was fitted to), `tau`, `char_time_mult` and the three `lambda` columns
- `ab_grid_m_star_vs_a_<stem>.png` — `m*` against `a`, one curve per `b`
- `ab_grid_m_star_vs_b_<stem>.png` — `m*` against `b`, one curve per `a`
- `ab_grid_m_star_intensity_<stem>.png` — `m*` over the grid, `a` on the vertical axis and `b` on the horizontal, one square per solved point coloured by magnitude
- `ab_grid_psd_y2_<stem>.png` — with `compute_psd`: one peak-zoom panel per `(a, b)`, `a` down and `b` across; the CSV gains the four `psd_` columns

The two slice plots use a log x-axis when the rates span a decade or more and are
all positive — which is the usual sweep — and a linear one otherwise, since `0`
is a legal rate and a log axis would drop it silently.

```bash
curl -s -X POST $ITCM/jobs -H 'Content-Type: application/json' -d '{
  "computation": "Characteristic Time (a,b grid)",
  "params": {
    "rg_param": 96, "ry_param": 96,
    "a_list": [1, 10, 100], "b_list": [1, 10, 100],
    "v_param": 10000, "T_param": 1,
    "N_LIST": [0, 4, 8, 12],
    "show_plt": false, "workers": 4
  },
  "submitted_by": "your-name"
}'
```

> **The two devices parallelise this differently, and the estimate reflects it.**
> On the CPU the points run in separate processes, `workers` at a time, so a 3×3
> grid over 4 workers costs three rounds rather than nine solves. On the GPU they
> run one at a time: a single cooperative kernel already occupies every SM, so a
> second concurrent point would only time-slice with the first.
>
> `workers` defaults to one per core less one, which is right for a machine to
> itself. The job worker already runs up to four jobs at a time, so **pin
> `workers` when submitting a grid into a busy queue** or it will contend with
> whatever else is running.
>
> The degenerate-point behaviour is the same as for the velocity sweep above:
> a point whose `|y''|` is still above `tau` at the end of the run records
> `t_star` and `m_star` as NaN with a warning. With slow switching onto the
> advective layer (`b ≤ 1`) that needs `T` of about 4. A single point at `a_list = b_list = [w]` reproduces
> `Characteristic Time (mass vs v)` at that `w` exactly, which is asserted by
> `tests/test_ab_grid_char_time.py`.

---

### Mass analysis

#### `Mass Analysis`
Mass on each layer as a function of time.
**Required:** `rg_param`, `ry_param`, `v_param`, `a_param`, `b_param`, `T_param`, `N_LIST`
**Optional:** `engine` — `time_stepper` (default) or `modal`. See [Choosing an engine](#choosing-an-engine-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
`device` — `cpu` (default), `gpu`, or `auto`; time-stepper only. See [Choosing a device](#choosing-a-device-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
`compute_psd` (default `false`) — see [The PSD of y''](#the-psd-of-y-compute_psd).
**Output:** five CSV series under `mass_analysis_results/`, plus PNGs when
`save_png` is true —
`diffusive/`, `advective/`, `total/`, `advective_over_total/`,
`advective_over_initial/`; with `compute_psd`, `psd_y2/` holds the peak-zoom
PNG and a one-row CSV of the peak and FWHM. The job `result` is
`{"output_dirs": [...], "engine": ...}`, plus `"psd_y2": {"psd_omega_peak",
"psd_peak", "psd_fwhm", "psd_fwhm_side", "H", "d_omega", "pad_factor"}` with
`compute_psd`.

```bash
curl -s -X POST $ITCM/jobs -H 'Content-Type: application/json' -d '{
  "computation": "Mass Analysis",
  "params": {"rg_param": 96, "ry_param": 96, "v_param": 1, "a_param": 10,
             "b_param": 0.1, "T_param": 4, "N_LIST": [0, 24, 48, 72],
             "engine": "modal", "compute_psd": true},
  "submitted_by": "your-name"
}'
```

> This computation takes the two switch rates separately rather than a single
> `w_param`; `a_param = b_param = w` reproduces exactly what it did before,
> filenames included — equal rates are still written `w=<rate>`, and only a
> genuinely asymmetric pair is written `a=<a>_b=<b>`.

#### `Jrr Mass Sum Over Time`
Cumulative mass leaving the outer boundary, summed from the radial current
`J_rr` over time.
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `T_param`, `N_LIST`
**Optional:** `collection_factor` (default `5`), `collection_factor_limit` (default `1000`)
**Output:** CSV (+ PNG) under `mass_analysis_results/diffusive/`.

---

### MFPT point collections

Both sample MFPT at each entry of `checkpoint_collect_container`, giving a curve
rather than a single number. `checkpoint_collect_container` holds **times** for
the time-dependent variant and **mass fractions** for the mass-dependent one.

#### `MFPT point collection (time T dep.)`
**Required:** `rg_param`, `ry_param`, `N_LIST`, `v_param`, `w_param`, `T_param`, `checkpoint_collect_container`
**Output:** CSV (+ PNG) under `mfpt-results/`, one row per checkpoint.

#### `MFPT point collection (Mass % dep.)`
**Required:** `rg_param`, `ry_param`, `N_LIST`, `v_param`, `w_param`, `checkpoint_collect_container`
**Optional:** `mass_retention_threshold` (default `0.01`)
**Output:** CSV (+ PNG) under `mfpt-results/`, one row per checkpoint.

---

### Density profiles

The angular pair gives diffusive-layer density against angle θ at a fixed ring
(`T_fixed_ring_seg`). The radial pair gives both layers' density against radius
at a fixed ray (`R_fixed_angle`). Each samples at every entry of
`checkpoint_collect_container`.

#### `Phi angular dependence (collection: time T dep.)`
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `T_param`, `N_LIST`, `checkpoint_collect_container`
**Optional:** `T_fixed_ring_seg` (default `0.5`)
**Output:** CSV (+ PNG) under `diffusive-v-theta/`, one series per checkpoint.

#### `Phi angular dependence (collection: Mass % dep.)`
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `N_LIST`, `checkpoint_collect_container`
**Optional:** `T_fixed_ring_seg` (default `0.5`), `mass_retention_threshold` (default `0.01`)
**Output:** CSV (+ PNG) under `diffusive-v-theta/`, one series per checkpoint.

#### `Phi/Rho radial dependence (collection: time T dep.)`
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `T_param`, `N_LIST`, `checkpoint_collect_container`
**Optional:** `R_fixed_angle` (default `-1`, meaning auto)
**Output:** CSV (+ PNG) under `density-results/radial-dependence/` —
`diffusive/` for φ and `rho/` for the advective layer.

#### `Phi/Rho radial dependence (collection: Mass % dep.)`
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `N_LIST`, `checkpoint_collect_container`
**Optional:** `R_fixed_angle` (default `-1`), `mass_retention_threshold` (default `0.01`)
**Output:** CSV (+ PNG) under `density-results/radial-dependence/` —
`diffusive/` for φ and `rho/` for the advective layer.

---

### Heatmaps

Both render full 2-D diffusive-layer density snapshots at each checkpoint on the
polar grid.

#### `Static Heatplot Analysis (collection: time T dep.)`
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `T_param`, `N_LIST`, `checkpoint_collect_container`
**Optional:** `heatplot_colorscheme` (default `"viridis"`), `heatplot_border`, `display_extraction`
**Output:** one PNG per checkpoint under `heatmaps/`, plus the underlying CSV.

#### `Static Heatplot Analysis (collection: Mass % dep.)`
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `N_LIST`, `checkpoint_collect_container`
**Optional:** `heatplot_colorscheme`, `heatplot_border`, `display_extraction`, `mass_retention_threshold`
**Output:** one PNG per checkpoint under `heatmaps/`, plus the underlying CSV.

These are the largest outputs — a snapshot is a full `rg × ry` field, so keep
`checkpoint_collect_container` short at large grids.

---

### Combined and matrix

#### `Full Analysis (time t dep.)`
Runs the mass, density, MFPT and heatmap analyses in a single pass over one
time integration, instead of re-solving per analysis. Use it when you want
everything for one parameter set.
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `T_param`, `N_LIST`
**Optional:** `Timestamp_List` (list[float], snapshot times, each ≤ `T_param`),
`T_fixed_ring_seg`, `R_fixed_angle`, plus the heatmap options
**Output:** writes into all of the directories above; the job's `output_dirs`
lists what was produced.

#### `Angular Trajectory Matrix`
Builds the angular diffusion matrix and central vector for the configuration.
**Required:** `rg_param`, `ry_param`, `v_param`, `w_param`, `N_LIST`,
`checkpoint_collect_container`
**Output:** PNG visualisations.

> The function returns the matrix and central vector as numpy arrays. Those are
> stringified when the job result is serialised to JSON, so treat the PNGs as
> the usable output rather than the `result` field. This one is mainly a
> development aid; the other computations are the ones written for the job API.

---

### Parameter sweeps

#### `BC Parameter Dependence (w sweep)`
Boundary-condition behaviour across a list of switch rates, holding the grid
fixed.
**Required:** `rg_param`, `ry_param`, `v_param`, `T_param`, `N_LIST`,
`w_LIST` (list[float]), `checkpoint`
**Output:** CSV (+ PNG) under `diffusive-v-theta/`, one series per `w`.

#### `BC Parameter Dependence (grid-size sweep)`
The same measurement across a list of grid sizes, holding `w` fixed — the
convergence study.
**Required:** `v_param`, `T_param`, `w_param`, `N_amount` (int, number of evenly
spaced microtubules), `checkpoint`, `grid_list` (list[int], square grid sizes)
**Output:** CSV under `diffusive-v-theta/`.

> This one takes `N_amount` rather than `N_LIST`, since the tube positions are
> regenerated per grid size. Mind the `G⁶` cost: `grid_list` of
> `[32, 48, 64, 96]` is dominated entirely by the last entry.

---

## Worked example, start to finish

```bash
export ITCM=http://100.83.174.69:8000

# 1. queue a 48x48 characteristic-time run (~4 min)
JOB=$(curl -s -X POST $ITCM/jobs -H 'Content-Type: application/json' -d '{
  "computation": "Characteristic Time (mass vs v)",
  "params": {"rg_param":48,"ry_param":48,"v_LIST":[10000],"w_param":100,
             "T_param":1,"N_LIST":[0,6,12,18,24,30,36,42],"show_plt":false},
  "submitted_by": "kevin"
}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
echo "$JOB"

# 2. come back later
curl -s $ITCM/jobs/$JOB | python3 -m json.tool

# 3. fetch everything it produced
curl -s $ITCM/jobs/$JOB/outputs \
  | python3 -c 'import sys,json;[print(f["path"]) for f in json.load(sys.stdin)["files"]]' \
  | tr -d '\r' \
  | while read p; do mkdir -p "$(dirname "$p")"; curl -s -o "$p" "$ITCM/jobs/$JOB/outputs/$p"; done
```

To sweep microtubule counts, submit one job per count — they run four at a time,
so four configurations cost about one job's wall time rather than four.
