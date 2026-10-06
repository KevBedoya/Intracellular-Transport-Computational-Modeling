# UI plan — a browser front end for the compute API

## The problem

The API works and results download fine, but launching a job by hand is
error-prone in a way that gets worse the more you use it:

- A `POST /jobs` body is a nested JSON object with 6-9 parameters, quoted inside
  a shell command. One misplaced brace and `curl` reports nothing useful.
- Parameters must be **real JSON types**. `"rg_param": "96"` and
  `"N_LIST": "[0,4,8]"` look right and are wrong.
- `N_LIST` has to be computed by hand per grid — 16 evenly spaced tubes on a
  112 grid is `[0,7,14,…,105]`, and getting it wrong silently changes the
  physics rather than erroring.
- The defaults and hints live in `PARAMETER_SCHEMAS`, so using them from a
  terminal means reading a second document or curling the schema endpoint and
  transcribing by eye.
- **Nothing validates the parameters at submit time.** Verified: a body missing
  a required parameter, carrying a wrong-typed value, or naming a parameter that
  does not exist is accepted with a `201` and fails later at run time, having
  already claimed a worker slot. Only the computation *name* is checked.

A form removes every one of these by construction.

## The central idea: generate the form from the schema

`GET /computations/<name>` already returns each computation's required
parameters, optional parameters with their defaults, and a one-line hint per
parameter — the same `PARAMETER_SCHEMAS` the desktop GUI uses.

**The UI should render its form from that response and hold no parameter
knowledge of its own.** The consequence is the whole reason to build it this
way: registering a nineteenth computation in `COMPUTATION_FUNCTIONS` plus its
schema gives it a working UI with no front-end change at all. Any design where
the form is hand-written duplicates 300 lines of parameter definitions and
starts drifting the first time a parameter is added — which is exactly the
failure mode this codebase already had with five divergent copies of the solver.

## Recommendation: a single-page app served by the existing Flask app

Node and npm *are* installed, so React is available. I still recommend against
it here:

| | single-file HTML/JS | React + build |
|---|---|---|
| Build step | none | `npm install`, bundle, deploy |
| Serving | `GET /ui` from the Flask app already running | separate static host or Flask static dir |
| CORS | none — same origin as the API | needs config or a proxy |
| Deploy | edit the file, reload the browser | rebuild, redeploy |
| Fits 2-3 users | yes | yes, with more moving parts |

For a tool used by a handful of people against a single-origin API, the build
pipeline is pure overhead, and every extra step is one more thing to fix at
11pm before a deadline. One `templates/ui.html` plus a small `static/ui.js`,
served by a new `GET /ui` route, keeps the entire front end inside the same
process that already runs. If it later grows into something multi-user with
accounts, that is the point to reconsider.

No new Python dependencies: Flask and Jinja2 are already installed, and
`flask-cors` is not needed precisely because the page is same-origin.

## What the API already supports, unchanged

| Need | Endpoint |
|---|---|
| Populate the computation picker | `GET /computations` |
| Build the parameter form | `GET /computations/<name>` |
| Queue a job | `POST /jobs` |
| Queue table with live status | `GET /jobs?status=&limit=` |
| Job detail, including the exact params used | `GET /jobs/<id>` |
| Cancel a queued job | `DELETE /jobs/<id>` |
| List a job's outputs with sizes | `GET /jobs/<id>/outputs` |
| Download a file | `GET /jobs/<id>/outputs/<path>` |
| Queue depth for a header badge | `GET /health` |

Two things already work in the UI's favour:

- **PNG previews need no API change.** `job_output_file` serves images with an
  `image/*` mimetype and `as_attachment=False`, so a plot can be shown inline
  with a plain `<img src>` pointing straight at the endpoint.
- **`GET /jobs/<id>` returns the full `params` object**, so "clone this job and
  change one value" is just repopulating the form from an existing record — no
  extra state to keep.

## What should be added to the API

1. **Validate parameters at submit time** (the gap verified above). `POST /jobs`
   should check the body against `PARAMETER_SCHEMAS`: required present, no
   unknown keys, values coercible to the expected type. Return `400` with a
   per-field error map so the form can highlight the offending inputs. This is
   worth doing regardless of the UI — it stops a typo from consuming a worker
   slot and failing hours later.
2. **A cost estimate**, either an endpoint or a documented formula the page
   computes itself. Work scales as `G⁶`; the page should say "this will take
   about 8 hours" *before* the click, not after. Anchoring on the measured
   96² = 3.28 hr and scaling by `(G/96)⁶` is accurate enough.
3. **Optional: a sweep endpoint.** See phase 3.

## Screens

**One page, three regions.** The desktop GUI's shape already works; this mirrors
it rather than inventing something new.

### 1. Launch panel (left)

- Computation dropdown, grouped the way `COMPUTATION_MENU.md` groups them
  (scalar results / characteristic time / mass analysis / MFPT / density /
  heatmaps / sweeps), because a flat list of 18 similar names is hard to scan.
- Parameter form generated from the schema:
  - required fields first, then optional in a collapsed "advanced" section
    prefilled with defaults — most runs only touch the required ones
  - the hint text as help under each field, not a hover tooltip, so it is
    readable without discovery
  - input type driven by the default's type: number, checkbox for bools,
    comma-separated text for lists
- **An `N_LIST` helper.** A "tubes" number input that computes evenly spaced
  indices for the current grid, with the resulting list shown and still
  editable. This removes the single most error-prone manual step.
- Cost estimate and a confirmation step once the estimate exceeds ~1 hour.
- Submit button, disabled until the form validates.

### 2. Queue (right, top)

Table of recent jobs: grid, computation, status, wall time, who submitted it.
Poll `GET /jobs` every 3-5 seconds — for this many users that is cheaper and far
simpler than SSE or websockets, and the data is small. Cancel button on queued
rows. Clicking a row opens it below.

### 3. Results (right, bottom)

For the selected job: the parameters it ran with, its `result` field for the
scalar computations, and its output files. PNGs shown inline; CSVs offered as a
download and, since they are typically a handful of rows, rendered as a small
table. A "download all" that fetches every file.

## Phasing

**Phase 1 — replaces the terminal.** `GET /ui` route, computation picker,
schema-driven form with type coercion, submit, queue table with polling.
This is the bulk of the value: after this nobody hand-writes JSON.

**Phase 2 — closes the loop.** Results panel, inline PNG preview, CSV table,
download-all, clone-a-job.

**Phase 3 — the thing the terminal cannot do well.** A **sweep builder**: pick
one parameter, give it a list of values, and submit one job per value as a named
group. This is directly motivated by the N=16 grid study, which needed a bespoke
script to submit five jobs varying only the grid and then collect them back into
one report. Generalising that — vary `rg_param`/`ry_param` over
`[48, 64, 80, 96, 112]`, or `v_param` over a list — turns a scripting task into
a form. It also wants a small amount of server support: a `group_id` on the job
record so a sweep's jobs can be listed and reported together.

Phase 3 is where a UI stops being a convenience and starts enabling work that is
currently too tedious to do casually.

## Details the UI must get right

- **Type coercion at the boundary.** The form collects strings; the API needs
  real types. Coerce using the schema's default as the type hint, and reject
  rather than guess when a value will not coerce.
- **Force `show_plt: false`.** It is a desktop-GUI setting that tries to open a
  plot window on the server. Do not expose it as a checkbox; set it and hide it.
- **Warn on short `T` for the characteristic-time computations.** t* is
  where |y''| of ln M settles below tau for good, which is about 0.5-0.7 for
  b = 100 but 2.7-3.8 or later for b <= 1. A T too short for the pair comes
  back NaN; say so in the form instead of making the user learn it from a job.
- **Surface the NaN case honestly.** A point whose decay never settles below
  `tau` within `T` records `t_star` and `m_star` as NaN. Render that as "no
  steady decay within T" rather than a blank cell that looks like a bug.
- **Show `error` on failed jobs.** It carries the solver's own message and is
  usually immediately actionable.
- **No auth, and the page should say so.** Anyone reaching it can queue and
  cancel work. A visible "you are on the tailnet as <host>" line in the header
  is honest and cheap. Real per-user attribution needs the API token discussed
  in `CONNECTING.md`; until then `submitted_by` is a free-text field, so the UI
  should at least prefill it from a value the user sets once rather than leaving
  it blank.

## Open questions

1. **Should the UI be able to cancel *running* jobs?** The API deliberately
   refuses (`409`) because a running job owns a subprocess and possibly hours of
   partial output. Adding it means the worker has to kill a child and decide
   what happens to the partial output directory.
2. **Retention.** Nothing prunes `data_output/jobs/<id>/`. A 112² run's
   timeseries is ~640 MB in memory but its output is small, so this is not
   urgent — but a "delete this job's outputs" button, or a retention policy,
   becomes wanted once the queue has a few hundred entries.
3. **Should the sweep builder write the report too?** The grid study produces a
   `.tex`/`.pdf` from a script. Whether that generalises into the UI, or stays
   a per-study script, is worth deciding before building phase 3.
