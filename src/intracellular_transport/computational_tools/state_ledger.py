"""A record, kept in the repository, of every saved final solver state.

A finished checkpointed solve keeps its final state (checkpointing.save_final)
so a later run can extend it to a larger T instead of starting over. The state
itself lives with the job's outputs -- on the compute server under
data_output/jobs/<job id>/checkpoints/<point tag>/ -- which is not in git. This
module keeps the index of what exists in ``comp-state-ledger/entries/``: one
JSON file per saved state, named ``<job id>__<point tag>.json``, recording where
the state is, how far it got and the parameters that produced it.

See comp-state-ledger/README.md for how to continue a job from its saved state.

The ledger directory is ``<repo>/comp-state-ledger`` unless ITCM_STATE_LEDGER
names another (the tests do, so they never write into the repository).
"""

import json
import os
import re


def ledger_dir():
    override = os.environ.get("ITCM_STATE_LEDGER")
    if override:
        return override
    from system_configuration import file_paths as fp

    return os.path.join(str(fp.BASE_DIR), "comp-state-ledger")


def entries_dir():
    return os.path.join(ledger_dir(), "entries")


def _job_and_tag(state_dir):
    """(job id, point tag) for a state directory.

    Job checkpoints sit at .../jobs/<job id>/checkpoints/<tag>/. Anything else
    -- a local run with ITCM_CHECKPOINT_DIR set by hand -- is keyed "local".
    """
    parts = os.path.normpath(os.path.abspath(state_dir)).split(os.sep)
    tag = parts[-1]
    if len(parts) >= 4 and parts[-2] == "checkpoints" and parts[-4] == "jobs":
        return parts[-3], tag
    return "local", tag


def _entry_name(job, tag):
    return re.sub(r"[^A-Za-z0-9_.=+-]", "_", f"{job}__{tag}") + ".json"


def record(state_dir, meta):
    """Write (or overwrite) the ledger entry for the final state in state_dir."""
    job, tag = _job_and_tag(state_dir)
    os.makedirs(entries_dir(), exist_ok=True)
    p = meta.get("params") or {}
    entry = {
        "job_id": job,
        "point": tag,
        "state_dir": os.path.abspath(state_dir),
        "t_reached": meta["t"],
        "k": meta["k"],
        "samples": meta["MA_k_step"],
        "T_requested": meta.get("T_requested"),
        "grid": f"{p.get('rg_param')}x{p.get('ry_param')}",
        "a": p.get("a"), "b": p.get("b"), "v": p.get("v"),
        "N_LIST": p.get("N_LIST"),
        "extension_digest": meta.get("extension_digest"),
        "saved_utc": meta.get("saved_utc"),
    }
    path = os.path.join(entries_dir(), _entry_name(job, tag))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def entries():
    """Every ledger entry, as dicts."""
    out = []
    d = entries_dir()
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as f:
                out.append(json.load(f))
    return out


def resolve(source, tag):
    """The state directory to extend point ``tag`` from.

    ``source`` is either a job id (full or an unambiguous prefix, looked up in
    the ledger) or a path: a job's checkpoints directory (``tag`` is appended)
    or a state directory itself.
    """
    source = str(source).strip()
    if os.path.isdir(source):
        cand = os.path.join(source, tag)
        return cand if os.path.isdir(cand) else source

    hits = [e for e in entries()
            if e["point"] == tag and str(e["job_id"]).startswith(source)]
    jobs = {e["job_id"] for e in hits}
    if len(jobs) > 1:
        raise ValueError(f"extend_from={source!r} matches several jobs: "
                         f"{sorted(jobs)}; give more of the job id")
    if not hits:
        raise ValueError(
            f"no saved final state for point {tag!r} from job {source!r} in "
            f"the ledger ({entries_dir()}). See comp-state-ledger/README.md.")
    return hits[0]["state_dir"]
