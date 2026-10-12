# comp-state-ledger — saved solver states, and how to continue a job from one

A characteristic-time solve run through the compute server keeps its **final
state** when it finishes: the two field layers plus the carried scalars, at the
end of the run. A later job can then **extend that run to a larger T** — continue
from where it stopped instead of re-solving from t = 0. At 96×96 on the GPU a
pair costs about 81 minutes per unit of T, so extending T = 1 → 4 saves a
quarter of the work.

This directory is the record of which states exist. The states themselves are
large-ish and live with the job's outputs on the compute server, not in git;
the ledger tells you where they are and what produced them.

## What is recorded

`entries/<job id>__<point>.json`, one per saved state, written automatically
when a job finishes (by `computational_tools/state_ledger.py`):

| field | meaning |
|---|---|
| `job_id`, `point` | the job and the point within it, e.g. `ab_grid_char_time_a1_b0.1` |
| `state_dir` | where the state is: `<data_output>/jobs/<job id>/checkpoints/<point>/` |
| `t_reached`, `k`, `samples` | how far the saved state got (time, step, mass samples) |
| `T_requested` | the T the job was asked for (`t_reached` is ≤ T by under 5 steps) |
| `grid`, `a`, `b`, `v`, `N_LIST` | the parameters that produced it |
| `extension_digest` | fingerprint of every parameter except T; an extension must match it |
| `saved_utc` | when it was written |

Each `state_dir` holds `final_state.npz` (the state), `final_state.json` (the
same record plus the full parameter set), `run.json`, and the five mass
timeseries `timeseries_0..4.dat` up to `t_reached`. All of these are needed to
extend; delete the directory and the ledger entry together.

Entries are written on the machine that ran the job, so new ones appear as
untracked files in the server's working copy — commit them to keep the record.

## Continuing a job

Submit the same computation with **every parameter identical except
`T_param`** (larger), plus `extend_from` set to the earlier job's id:

```bash
curl -s -X POST $ITCM/jobs -H 'Content-Type: application/json' -d '{
  "computation": "Characteristic Time (a,b grid)",
  "params": {
    "rg_param": 96, "ry_param": 96,
    "a_list": [0.1, 1, 10, 100], "b_list": [0.1, 1],
    "v_param": 1, "T_param": 8,
    "N_LIST": [0, 24, 48, 72], "device": "gpu", "tau": 0.01,
    "extend_from": "<earlier job id, or a unique prefix>"
  }
}'
```

For each point the new job finds the earlier state through this ledger, checks
that it belongs to the same run, copies the samples already taken into its own
(longer) series, and continues the solve from `t_reached` to the new T. The
result is the same as a straight run to the new T: bit for bit on the CPU, and
to the usual ~1e-14 GPU tolerance on the GPU (`tests/test_final_state_extension.py`).

- `extend_from` also accepts a path: the earlier job's `checkpoints/`
  directory, or one point's state directory.
- The grid may be a subset of the earlier job's points; each point is matched
  by its tag. A point with no saved state fails with a message naming it.
- **Refused:** any difference other than T — grid, rates, velocity, tubes,
  `d_tube`, initial condition, `MA_collection_factor` — and a new T not larger
  than `t_reached`. Resuming into a different configuration would produce a
  trajectory belonging to neither run.
- The desktop app and direct scripts do not checkpoint, so `extend_from` works
  only for jobs run through the compute server (or with `ITCM_CHECKPOINT_DIR`
  set).

## Which state is kept, and why

The state is saved at step k_align = `MA_collection_factor` × (number of mass
samples), the first step whose sample the run had no room for — not at the
final step K. When K is not a multiple of the collection factor, the last sample
before K is computed but dropped; resuming from K would then file every later
sample one slot early. Resuming from k_align files each one exactly where a
straight run would have.

States saved before 2026-10-11 do not exist: until then a finished job deleted
its state. Job `0b8d4022` (96×96, T = 1) therefore cannot be extended.
