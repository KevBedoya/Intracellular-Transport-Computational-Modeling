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
retained mass `m*` at `t*`.

`t*` is judged from the second derivative of `y = ln(total mass)`: the decay is
steady where `ln M` is a straight line, i.e. `y'' = 0`. The threshold is applied
to the normalised curvature `κ = |y''| / y'²` (the rate at which the local
e-folding time drifts), which is dimensionless, so one `tau` means the same thing
at every velocity and switch rate. `t*` is the time after which `κ` **stays
below `tau` for the rest of the solve**, and that steady segment must last at
least 0.1. Derivatives are taken at a fixed spacing of 1e-3 in time, so `t*` does
not depend on grid resolution. The fit columns in the CSV are a least-squares line
through `log10(total mass)` over that steady segment.
**Required:** `rg_param`, `ry_param`, `v_LIST` (list[float]), `w_param`, `T_param`, `N_LIST`
**Optional:** `tau` (default `1e-3`) — the steady-decay threshold; smaller is
stricter and gives a later `t*`.
`device` — `cpu` (default), `gpu`, or `auto`. See [Choosing a device](#choosing-a-device-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
**Output:** `char_time_analysis/<timestamp>/`
- `char_t_analysis_data.csv` — one row per velocity: `v, t_star, m_star`, then
  `fit_slope, fit_intercept` (the line over the steady segment),
  `fit_window_t1, fit_window_t2` (that segment, `[t*, T]`),
  `total_mass_at_t1, total_mass_at_t2` (the mass at its ends) and `tau`
- `char_t_analysis_plot.png` — `m*` against `v`, log x-axis

> **`T_param` must leave room for the steady decay.** `T ≤ 0.1` is rejected
> within a second. Above that, the onset is only known after the solve: it is
> typically 0.25–0.45 at `tau = 1e-3`, so `T` of about 1 is the safe choice. A
> point whose decay has not settled below `tau` with 0.1 of the record left writes
> `t_star` and `m_star` as empty fields (NaN) with a warning, and the rest of
> the sweep is kept.
>
> **This definition replaced an earlier one** (a line through `log10 M` at
> `t = 0.4` and `t = 0.5`, extrapolated to its intercept, with `m*` read at
> `10·t*`). Numbers from before the change are not comparable with those after.
>
> One velocity per job is the usual choice: a `v_LIST` of five values costs five
> full solves sequentially inside one job, whereas five separate jobs run four
> at a time.

#### `Characteristic Time (a,b grid)`
The same fit, swept over the two switch rates instead of over velocity. `a_list`
holds the rates onto the diffusive layer, `b_list` the rates onto the advective
layer, and **every pair in `a_list × b_list` is solved** — a 4×4 grid is sixteen
full solves. `v_param` is a single value here, not a list: the grid is already
two-dimensional.
**Required:** `rg_param`, `ry_param`, `a_list` (list[float]), `b_list` (list[float]), `v_param`, `T_param`, `N_LIST`
**Optional:** `tau` (default `1e-3`), as above.
`device` — `cpu` (default), `gpu`, or `auto`. See [Choosing a device](#choosing-a-device-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
`workers` — how many points to solve at once on the CPU; `0` (default) uses one
per core, less one.
**Output:** `ab_grid_char_time/<timestamp>/`

Every filename ends in the parameters held fixed across the run — `v`, the
microtubule count and the grid — written as `v10000_N4_96x96` for the example
above. They are not columns, so a file lifted out of its timestamped directory
still says what produced it.

- `ab_grid_char_t_<stem>.csv` — one row per `(a, b)`: `a, b, t_star, m_star`, then the fit columns (`fit_slope`, `fit_intercept`, the window, and the two masses it was fitted to)
- `ab_grid_m_star_vs_a_<stem>.png` — `m*` against `a`, one curve per `b`
- `ab_grid_m_star_vs_b_<stem>.png` — `m*` against `b`, one curve per `a`
- `ab_grid_m_star_intensity_<stem>.png` — `m*` over the grid, `a` on the vertical axis and `b` on the horizontal, one square per solved point coloured by magnitude

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
> The `T_param` rule and the degenerate-point behaviour are the same as for the
> velocity sweep above: `T ≤ 0.1` is refused, and a point whose decay does not
> settle records `t_star` and `m_star` as NaN with a warning. A single point at `a_list = b_list = [w]` reproduces
> `Characteristic Time (mass vs v)` at that `w` exactly, which is asserted by
> `tests/test_ab_grid_char_time.py`.

---

### Mass analysis

#### `Mass Analysis`
Mass on each layer as a function of time.
**Required:** `rg_param`, `ry_param`, `v_param`, `a_param`, `b_param`, `T_param`, `N_LIST`
**Optional:** `device` — `cpu` (default), `gpu`, or `auto`. See [Choosing a device](#choosing-a-device-mass-analysis-characteristic-time-mass-vs-v-characteristic-time-ab-grid).
**Output:** five CSV series under `mass_analysis_results/`, plus PNGs when
`save_png` is true —
`diffusive/`, `advective/`, `total/`, `advective_over_total/`,
`advective_over_initial/`.

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
