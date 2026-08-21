"""Chunking the time loop must not change the answer.

Checkpointing rests on one property: stopping the solve at an arbitrary timestep
and resuming from the carried state is exactly equivalent to running straight
through. If that does not hold to the last bit, a resumed run is not the run it
claims to be.

These tests assert it directly -- the same solve split into 1, 2, 3, 7 and many
chunks, and at deliberately awkward boundaries that do not divide the collection
interval, must all produce identical arrays. Awkward boundaries matter because
mass samples are only recorded every MA_collection_factor steps, so a chunk edge
landing mid-interval is exactly where an off-by-one would hide.

Run directly (``python tests/test_checkpoint_resume.py``) or under pytest.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
_PKG = os.path.join(_SRC, "intracellular_transport")
for _p in (_PKG, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from computational_tools import analysis_tools as ant       # noqa: E402
from computational_tools import numerical_tools as num      # noqa: E402
from computational_tools import supplements as sup          # noqa: E402

MA_FACTOR = 5

# Kept small so the suite stays quick; a few thousand steps is enough to cross
# the collection branch many times and to exercise extraction regions.
CASES = [
    dict(rg=24, ry=24, T=0.02, tubes=4, d_tube=0.0, w=100.0, v=1e4),
    dict(rg=32, ry=32, T=0.01, tubes=8, d_tube=0.0, w=100.0, v=1e4),
    dict(rg=32, ry=32, T=0.01, tubes=4, d_tube=0.03, w=10.0, v=1e3),
    dict(rg=40, ry=32, T=0.008, tubes=8, d_tube=0.0, w=100.0, v=1e4),  # non-square
]


def _solve(case, chunk_steps):
    """Run one case with a given chunk size; return the final state."""
    rg, ry, T = case["rg"], case["ry"], case["T"]
    N_LIST = np.linspace(0, ry - (ry // case["tubes"]), case["tubes"], dtype=int)
    K = num.compute_K(rg, ry, T)
    relative_k = max(int(math.floor(K / MA_FACTOR)), 1)

    D_LAYER, A_LAYER = sup.initialize_layers(rg, ry)
    series = [np.zeros(relative_k, dtype=np.float64) for _ in range(5)]

    state = ant.comp_mass_analysis_respect_to_time(
        rg, ry, case["w"], case["w"], case["v"], T, N_LIST, D_LAYER, A_LAYER,
        series[0], series[1], series[2], series[3], series[4],
        MA_FACTOR, relative_k, case["d_tube"], 1.0, 1.0, 10 ** 18,
        True, 0, 0, chunk_steps=chunk_steps)

    return D_LAYER, A_LAYER, series, state, K


def _label(case):
    return (f"{case['rg']}x{case['ry']} T={case['T']} "
            f"tubes={case['tubes']} d_tube={case['d_tube']}")


def test_chunking_does_not_change_the_result():
    """Any chunk size gives the same answer as running in one pass."""
    failures = []
    for case in CASES:
        D0, A0, s0, st0, K = _solve(case, chunk_steps=0)      # single pass

        # Sizes chosen to land on and off multiples of MA_FACTOR, and to include
        # one that does not divide K evenly so the final chunk is short.
        sizes = [K, K // 2 + 1, K // 3, 7, 1000, MA_FACTOR, MA_FACTOR * 2 + 1]
        for size in sizes:
            if size < 1:
                continue
            D1, A1, s1, st1, _ = _solve(case, chunk_steps=size)
            tag = f"{_label(case)}  chunk={size}"
            if not np.array_equal(D0, D1):
                failures.append(f"{tag}: D_LAYER differs "
                                f"(max |delta| {np.max(np.abs(D0 - D1)):.3e})")
            if not np.array_equal(A0, A1):
                failures.append(f"{tag}: A_LAYER differs "
                                f"(max |delta| {np.max(np.abs(A0 - A1)):.3e})")
            for j, (a, b) in enumerate(zip(s0, s1)):
                if not np.array_equal(a, b):
                    failures.append(f"{tag}: timeseries {j} differs "
                                    f"(max |delta| {np.max(np.abs(a - b)):.3e})")
            if st0 != st1:
                failures.append(f"{tag}: carried state differs {st0!r} vs {st1!r}")

    assert not failures, "chunking changed the result:\n  " + "\n  ".join(failures)


def test_single_step_chunks_match():
    """The pathological case: one chunk per timestep.

    Slow, so only the smallest configuration -- but it is the strongest form of
    the property, since every possible boundary is exercised at once.
    """
    case = CASES[0]
    D0, A0, s0, st0, _ = _solve(case, chunk_steps=0)
    D1, A1, s1, st1, _ = _solve(case, chunk_steps=1)
    assert np.array_equal(D0, D1), "D_LAYER differs with one chunk per step"
    assert np.array_equal(A0, A1), "A_LAYER differs with one chunk per step"
    for j, (a, b) in enumerate(zip(s0, s1)):
        assert np.array_equal(a, b), f"timeseries {j} differs with one chunk per step"
    assert st0 == st1, "carried state differs with one chunk per step"


def test_on_chunk_hook_reports_every_boundary():
    """The checkpoint hook fires once per chunk, with the timestep reached."""
    case = CASES[0]
    rg, ry, T = case["rg"], case["ry"], case["T"]
    N_LIST = np.linspace(0, ry - (ry // case["tubes"]), case["tubes"], dtype=int)
    K = num.compute_K(rg, ry, T)
    relative_k = max(int(math.floor(K / MA_FACTOR)), 1)
    D_LAYER, A_LAYER = sup.initialize_layers(rg, ry)
    series = [np.zeros(relative_k, dtype=np.float64) for _ in range(5)]

    seen = []
    chunk = 500
    ant.comp_mass_analysis_respect_to_time(
        rg, ry, case["w"], case["w"], case["v"], T, N_LIST, D_LAYER, A_LAYER,
        series[0], series[1], series[2], series[3], series[4],
        MA_FACTOR, relative_k, case["d_tube"], 1.0, 1.0, 10 ** 18,
        True, 0, 0, chunk_steps=chunk,
        on_chunk=lambda k, state: seen.append((k, state)))

    expected = list(range(chunk, K, chunk)) + [K]
    assert [k for k, _ in seen] == expected, (
        f"hook fired at {[k for k, _ in seen][:6]}…, expected {expected[:6]}…")
    assert seen[-1][0] == K, "hook did not report reaching the final timestep"
    assert all(len(s) == 4 for _, s in seen), "hook state is not the four scalars"


def _run_with_checkpoint(case, directory, chunk, stop_after=None, resume=False):
    """Solve with checkpointing, optionally aborting after N chunks.

    ``stop_after`` simulates a crash: the driver is abandoned partway with its
    checkpoint on disk, exactly as a killed process would leave it.
    """
    from computational_tools import checkpointing as chk

    rg, ry, T = case["rg"], case["ry"], case["T"]
    N_LIST = np.linspace(0, ry - (ry // case["tubes"]), case["tubes"], dtype=int)
    K = num.compute_K(rg, ry, T)
    relative_k = max(int(math.floor(K / MA_FACTOR)), 1)

    digest, params = chk.fingerprint(
        rg=rg, ry=ry, a=case["w"], b=case["w"], v=case["v"], T=T,
        N_LIST=N_LIST, d_tube=case["d_tube"], relative_k=relative_k,
        MA_collection_factor=MA_FACTOR)

    series = chk.allocate_timeseries(relative_k, 5, directory=directory,
                                     resume=resume)
    cp = chk.Checkpoint(directory, digest, params)
    D_LAYER, A_LAYER = sup.initialize_layers(rg, ry)

    class Abort(Exception):
        pass

    calls = {"n": 0}

    def hook(k, state):
        calls["n"] += 1
        if stop_after is not None and calls["n"] >= stop_after:
            raise Abort()

    try:
        ant.comp_mass_analysis_respect_to_time(
            rg, ry, case["w"], case["w"], case["v"], T, N_LIST,
            D_LAYER, A_LAYER, series[0], series[1], series[2], series[3],
            series[4], MA_FACTOR, relative_k, case["d_tube"], 1.0, 1.0,
            10 ** 18, True, 0, 0,
            chunk_steps=chunk, on_chunk=hook, checkpoint=cp)
    except Abort:
        chk.flush_timeseries(series)
        return None, None, None, True

    return D_LAYER, A_LAYER, [np.array(s) for s in series], False


def test_interrupted_run_resumes_to_the_same_answer():
    """A run killed midway and resumed equals one that was never interrupted.

    This is the property checkpointing exists to provide. Anything less and a
    resumed multi-day solve is not the solve it reports itself to be.
    """
    import shutil
    import tempfile

    for case in CASES:
        D_ref, A_ref, s_ref, _, _ = _solve(case, chunk_steps=0)

        tmp = tempfile.mkdtemp(prefix="ckpt_")
        try:
            K = num.compute_K(case["rg"], case["ry"], case["T"])
            chunk = max(K // 5, 1)

            # abort after two chunks, leaving a checkpoint behind
            _, _, _, aborted = _run_with_checkpoint(
                case, tmp, chunk, stop_after=2)
            assert aborted, "the simulated interruption did not happen"
            assert os.path.exists(os.path.join(tmp, "resume_state.npz")), \
                "no checkpoint was written before the interruption"

            # resume and finish
            D_r, A_r, s_r, _ = _run_with_checkpoint(
                case, tmp, chunk, resume=True)

            label = _label(case)
            assert np.array_equal(D_ref, D_r), \
                f"{label}: D_LAYER differs after resume " \
                f"(max |delta| {np.max(np.abs(D_ref - D_r)):.3e})"
            assert np.array_equal(A_ref, A_r), \
                f"{label}: A_LAYER differs after resume " \
                f"(max |delta| {np.max(np.abs(A_ref - A_r)):.3e})"
            for j, (a, b) in enumerate(zip(s_ref, s_r)):
                assert np.array_equal(a, b), \
                    f"{label}: timeseries {j} differs after resume " \
                    f"(max |delta| {np.max(np.abs(a - b)):.3e})"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_resume_refuses_a_mismatched_configuration():
    """A checkpoint from another run must be refused, not silently used.

    Resuming into different parameters would yield a trajectory belonging to
    neither run, and nothing downstream could detect it.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="ckpt_mismatch_")
    try:
        case = CASES[0]
        K = num.compute_K(case["rg"], case["ry"], case["T"])
        _, _, _, aborted = _run_with_checkpoint(case, tmp, max(K // 4, 1),
                                                stop_after=1)
        assert aborted

        other = dict(case)
        other["w"] = case["w"] * 2          # same shapes, different physics
        try:
            _run_with_checkpoint(other, tmp, max(K // 4, 1), resume=True)
        except ValueError as e:
            assert "different run" in str(e), f"unexpected message: {e}"
        else:
            raise AssertionError(
                "a checkpoint from a different configuration was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_timeseries_memmap_matches_in_ram():
    """Backing the timeseries with memmaps changes nothing numerically."""
    import shutil
    import tempfile

    case = CASES[1]
    _, _, s_ram, _, _ = _solve(case, chunk_steps=0)
    tmp = tempfile.mkdtemp(prefix="ckpt_mm_")
    try:
        K = num.compute_K(case["rg"], case["ry"], case["T"])
        _, _, s_mm, _ = _run_with_checkpoint(case, tmp, max(K // 3, 1))
        for j, (a, b) in enumerate(zip(s_ram, s_mm)):
            assert np.array_equal(a, b), \
                f"timeseries {j} differs between RAM and memmap backing"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_chunking_does_not_change_the_result()
    print("OK: chunk size does not affect the result (4 configurations x 7 sizes)")
    test_single_step_chunks_match()
    print("OK: one chunk per timestep matches a single pass")
    test_on_chunk_hook_reports_every_boundary()
    print("OK: the checkpoint hook fires at every chunk boundary")
    test_timeseries_memmap_matches_in_ram()
    print("OK: memmap-backed timeseries match in-RAM arrays")
    test_interrupted_run_resumes_to_the_same_answer()
    print("OK: an interrupted run resumes to the identical answer (4 configs)")
    test_resume_refuses_a_mismatched_configuration()
    print("OK: a checkpoint from a different configuration is refused")
