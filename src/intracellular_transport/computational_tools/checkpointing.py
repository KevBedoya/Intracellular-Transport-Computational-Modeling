"""Durable state for long solves.

A multi-day run has two kinds of state, and they want completely different
handling because they differ in size by three orders of magnitude:

* **Resume state** -- the two field layers plus four scalars. Under 1 MB even at
  160x160 (the layers are only 0.82 MB there). Cheap to write repeatedly.
* **Accumulated output** -- the five mass timeseries. 2.7 GB at 160x160, and
  append-only: a run never rewrites a sample it already took.

So the resume state is snapshotted whole, while the timeseries are backed by
``np.memmap`` and left on disk continuously. That choice does double duty: the
samples survive a crash without being copied anywhere, and they stop occupying
resident memory, which at 160x160 is 2.7 GB per job that no longer has to be
held.

numba writes into a memmap exactly as it writes into an ordinary array (a
memmap is an ndarray subclass), so the solver kernel needs no change at all.

Resuming into a *different* configuration would silently produce nonsense, so
every checkpoint carries a fingerprint of the parameters that defined the run
and is refused if it does not match.
"""

import hashlib
import json
import os

import numpy as np

STATE_FILE = "resume_state.npz"
TIMESERIES_PREFIX = "timeseries"
FINGERPRINT_FILE = "run.json"

# Bumped if the on-disk layout changes in a way older checkpoints cannot satisfy.
FORMAT_VERSION = 1


def fingerprint(**params):
    """A stable digest of the parameters that define a run.

    Anything that changes the trajectory belongs in here. Ordering is normalised
    so the same run always hashes the same way, and arrays are converted to
    lists so N_LIST participates by value.
    """
    norm = {}
    for key in sorted(params):
        val = params[key]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        elif isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = float(val)
        norm[key] = val
    norm["_format"] = FORMAT_VERSION
    blob = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest(), norm


def allocate_timeseries(relative_k, count=5, directory=None, resume=False):
    """Return ``count`` float64 timeseries of length ``relative_k``.

    With no directory they are ordinary in-RAM arrays, which is the historical
    behaviour. With a directory they are memmaps: on disk, crash-durable, and
    not counted against resident memory.

    ``resume=True`` reopens existing files in place ('r+') instead of creating
    them, so samples already collected are kept.
    """
    if directory is None:
        return [np.zeros(relative_k, dtype=np.float64) for _ in range(count)]

    os.makedirs(directory, exist_ok=True)
    mode = "r+" if resume else "w+"
    out = []
    for i in range(count):
        path = os.path.join(directory, f"{TIMESERIES_PREFIX}_{i}.dat")
        if mode == "r+" and not os.path.exists(path):
            # Asked to resume but this series is missing: start it rather than
            # failing the whole run. The state file is the authority on how far
            # the solve got, and a missing series can only mean an interrupted
            # first allocation.
            out.append(np.memmap(path, dtype=np.float64, mode="w+",
                                 shape=(relative_k,)))
            continue
        out.append(np.memmap(path, dtype=np.float64, mode=mode,
                             shape=(relative_k,)))
    return out


def flush_timeseries(series):
    for s in series:
        if isinstance(s, np.memmap):
            s.flush()


class Checkpoint:
    """Reads and writes the resume state for one run.

    Usage from a solver driver::

        cp = Checkpoint(directory, fp_digest)
        resumed = cp.try_resume(D_LAYER, A_LAYER)   # fills the layers in place
        ...
        cp.save(k, D_LAYER, A_LAYER, scalars, series)
    """

    def __init__(self, directory, digest, params=None, every=1):
        self.directory = directory
        self.digest = digest
        self.params = params
        self.every = max(int(every), 1)
        self._writes = 0
        os.makedirs(directory, exist_ok=True)
        # Only write the fingerprint if none exists. Overwriting it here would
        # destroy the very record try_resume needs to compare against, so a
        # checkpoint from a different configuration would be silently accepted.
        if params is not None and not os.path.exists(self.fingerprint_path):
            self._write_fingerprint(params)

    # ---------------------------------------------------------------- paths
    @property
    def state_path(self):
        return os.path.join(self.directory, STATE_FILE)

    @property
    def fingerprint_path(self):
        return os.path.join(self.directory, FINGERPRINT_FILE)

    def _write_fingerprint(self, params):
        payload = {"digest": self.digest, "params": params}
        tmp = self.fingerprint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, self.fingerprint_path)

    # --------------------------------------------------------------- resume
    def has_state(self):
        return os.path.exists(self.state_path)

    def stored_digest(self):
        try:
            with open(self.fingerprint_path) as f:
                return json.load(f).get("digest")
        except (OSError, ValueError):
            return None

    def try_resume(self, D_LAYER, A_LAYER):
        """Load a checkpoint into the given layers.

        Returns ``(k, central_patch, dl_mass, al_mass, MA_k_step)`` on success,
        or None if there is nothing to resume from. A checkpoint whose
        fingerprint does not match is refused rather than used: resuming into a
        different configuration would produce a trajectory belonging to neither
        run, and no error would ever surface.
        """
        if not self.has_state():
            return None

        stored = self.stored_digest()
        if stored is not None and stored != self.digest:
            raise ValueError(
                f"checkpoint in {self.directory} belongs to a different run "
                f"(stored fingerprint {stored[:12]}..., this run "
                f"{self.digest[:12]}...). Refusing to resume; delete the "
                f"directory to start over.")

        with np.load(self.state_path) as z:
            if z["D"].shape != D_LAYER.shape or z["A"].shape != A_LAYER.shape:
                raise ValueError(
                    f"checkpoint layer shape {z['D'].shape} does not match this "
                    f"run's {D_LAYER.shape}")
            D_LAYER[...] = z["D"]
            A_LAYER[...] = z["A"]
            return (int(z["k"]), float(z["central_patch"]),
                    float(z["dl_mass"]), float(z["al_mass"]),
                    int(z["MA_k_step"]))

    # ----------------------------------------------------------------- save
    def save(self, k, D_LAYER, A_LAYER, scalars, series=None, force=False):
        """Persist the resume state. Returns True if it actually wrote.

        Written to a temporary file and renamed, so a crash mid-write leaves the
        previous checkpoint intact rather than a truncated one -- otherwise the
        first power cut during a save would destroy the very thing being kept.

        The timeseries are flushed first: the state file records how many
        samples exist, so it must never be newer than the samples themselves.
        """
        self._writes += 1
        if not force and (self._writes % self.every) != 0:
            return False

        if series is not None:
            flush_timeseries(series)

        central_patch, dl_mass, al_mass, MA_k_step = scalars
        tmp = self.state_path + ".tmp"
        np.savez(tmp,
                 k=np.int64(k),
                 central_patch=np.float64(central_patch),
                 dl_mass=np.float64(dl_mass),
                 al_mass=np.float64(al_mass),
                 MA_k_step=np.int64(MA_k_step),
                 D=np.asarray(D_LAYER),
                 A=np.asarray(A_LAYER))
        # np.savez appends .npz when handed a path without one
        produced = tmp if os.path.exists(tmp) else tmp + ".npz"
        os.replace(produced, self.state_path)
        return True

    def clear(self):
        """Remove the resume state, leaving the timeseries alone.

        Called on successful completion: the run is finished, so a stale
        checkpoint would only invite a pointless resume.
        """
        for p in (self.state_path, self.state_path + ".tmp"):
            try:
                os.remove(p)
            except OSError:
                pass
