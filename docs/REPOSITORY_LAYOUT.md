# Repository layout

The repository was previously organised by month (`Dec-2024/`, `Jan-2025/`,
… `June-2026/`), with a full copy of the source package inside each. That meant
five divergent copies of the solver — the core `u_density` kernel existed in
four genuinely different implementations — so a fix applied to one copy left the
others stale, and results from different directories were not necessarily
comparable.

The layout is now organised **by kind, not by date**. There is one copy of the
code; dates live in result directory *names*, where they are metadata rather
than structure.

## Directories

| Path | Contents |
|---|---|
| `src/intracellular_transport/` | The one and only source package. Code only — no generated output. |
| `server/` | Job queue, HTTP API, and the run/setup scripts for using this machine as a compute server. |
| `tests/` | Golden-master and API tests, plus their fixtures. |
| `scripts/` | One-off analysis and maintenance scripts. Not imported by the package. |
| `docs/` | Written material: reports, derivations, notes, problem statements. |
| `results/` | Numerical results and figures worth keeping, one directory per study. |
| `data_output/` | **Generated** output. Scratch — safe to delete. Not committed. |
| `packaging/` | Build configuration and application icons. |

### `src/` layout and imports

The package lives at `src/intracellular_transport/`. Two entries go on
`sys.path` because the codebase uses both import styles:

* **flat** — `from computational_tools import numerical_tools`
  resolves against `src/intracellular_transport/`
* **dotted** — `from intracellular_transport.gui_components import main_gui`
  resolves against `src/`

`main.py:package_on_path()` sets both up, and `server/*.py` and `tests/*.py` do
the same at the top of each file. `subprocess_launcher.launch_subprocess()`
passes all of it to child processes via `PYTHONPATH`, because a child started
with `python -m …` does not inherit the parent's `sys.path`.

Normalising onto a single import style is worthwhile future work, but it touches
every module and is deliberately not bundled with the directory move.

### `results/` naming

`YYYY-MM[-DD]_short-description`, so directories sort chronologically. Where the
original name encoded parameters they are preserved in a legible form —
`w=10^4` became `w1e4` (not `w-104`, which reads as one hundred and four), and
`V=-10` became `Vneg10`.

`results/2026-08-15_char-time-sweep/` also carries a `raw/` subdirectory with
the per-configuration CSV and PNG behind the published tables, labelled by
microtubule count (`96x96/N04`, `48x48/N24`, …). Those files previously existed
only under an ignored `data_output/` path, so the data behind a committed
report was not in version control, and the mapping from timestamped directory
to tube count lived only in a console log.

### `data_output/` versus `results/`

`data_output/` is where the solver writes as it runs, at the repository root so
that generated data never lands inside `src/`. It is disposable. When a run
produces something worth keeping, it gets promoted into `results/` under a
descriptive name. `ITCM_OUTPUT_ROOT` overrides the location for one process;
the job worker uses that to give every job its own directory.

## Recovering the old layout

The month-organised tree is tagged:

```
git show snapshot/month-layout --stat
git checkout snapshot/month-layout      # detached, to browse
git show snapshot/month-layout:May-2025/project_src_package_2025/computational_tools/numerical_tools.py
```

One tag covers all of it: every month directory existed simultaneously at that
commit, so separate per-month tags would all have pointed at the same object.
