"""A finished run keeps its final state, and a longer run can continue from it.

The property everything here rests on: running to T1, keeping the final state,
and extending to T2 > T1 gives exactly what a straight run to T2 gives -- the
field layers and every mass sample, bit for bit on the CPU.

The trap is sample alignment. A run stores floor(K / MA_collection_factor)
samples, so when K is not a multiple of the collection factor the last sampled
step before K is computed and dropped. Resuming from K would then file every
later sample one slot early. The state is kept at k_align (the first step with
no room for its sample) instead; both cases are exercised below.

Also covered: refusal to extend a different run, the comp-state-ledger record and
lookup by job id, the full path through collect_ab_grid_char_time, and the GPU.

Run directly (``python tests/test_final_state_extension.py``) or under pytest.
"""
import json
import math
import os
import shutil
import sys
import tempfile

# Never write into the repository's ledger from a test.
os.environ.setdefault("ITCM_STATE_LEDGER",
                      os.path.join(tempfile.gettempdir(), "itcm_test_state_ledger"))
os.environ.setdefault("ITCM_OUTPUT_ROOT",
                      os.path.join(tempfile.gettempdir(), "itcm_ab_grid_tests"))
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src", "intracellular_transport"),
           os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from computational_tools import analysis_tools as ant          # noqa: E402
from computational_tools import checkpointing as chk           # noqa: E402
from computational_tools import numerical_tools as num         # noqa: E402
from computational_tools import state_ledger                   # noqa: E402
from computational_tools import supplements as sup             # noqa: E402

MCF = 5
RG = RY = 24
N_LIST = np.array([0, 6, 12, 18], dtype=np.int64)
A_RATE, B_RATE, V = 10.0, 0.1, 1.0


def _params(T):
    K = num.compute_K(RG, RY, T)
    rk = int(math.floor(K / MCF))
    digest, params = chk.fingerprint(
        rg_param=RG, ry_param=RY, a=A_RATE, b=B_RATE, v=V, T=T, N_LIST=N_LIST,
        d_tube=0.0, domain_radius=1.0, D=1.0, MA_collection_factor=MCF,
        relative_k=rk, center_init_cond=True, m_init=0, n_init=0)
    return K, rk, digest, params


def _solve(T, directory=None, resume=False, chunk=0, a=A_RATE):
    """One solve; checkpointed into ``directory`` when given."""
    K, rk, digest, params = _params(T)
    if a != A_RATE:
        digest, params = chk.fingerprint(**{**params, "a": a})
    D_LAYER, A_LAYER = sup.initialize_layers(RG, RY)
    if directory is None:
        series = chk.allocate_timeseries(rk, 5)
        cp = None
    else:
        series = chk.allocate_timeseries(rk, 5, directory=directory, resume=resume)
        cp = chk.Checkpoint(directory, digest, params)
    ant.comp_mass_analysis_respect_to_time(
        RG, RY, a, B_RATE, V, T, N_LIST, D_LAYER, A_LAYER,
        series[0], series[1], series[2], series[3], series[4], MCF, rk, 0.0,
        1.0, 1.0, 10 ** 18, True, 0, 0, chunk_steps=chunk, checkpoint=cp)
    return D_LAYER, A_LAYER, [np.array(s) for s in series]


def _t_pair(aligned):
    """(T1, T2) with K1 % MCF == 0 when aligned, != 0 otherwise."""
    dT = num.compute_dT(RG, RY)
    for k1 in range(4000, 6000):
        if (k1 % MCF == 0) == aligned:
            T1 = (k1 + 0.5) * dT
            if num.compute_K(RG, RY, T1) == k1:
                return T1, 2.3 * T1
    raise AssertionError("no suitable T1")


def test_extension_equals_a_straight_run():
    for aligned in (True, False):
        T1, T2 = _t_pair(aligned)
        K1 = num.compute_K(RG, RY, T1)
        assert (K1 % MCF == 0) == aligned
        D_ref, A_ref, s_ref = _solve(T2)

        tmp = tempfile.mkdtemp(prefix="final_state_")
        try:
            short, longer = os.path.join(tmp, "short"), os.path.join(tmp, "long")
            _solve(T1, short, chunk=777)            # awkward chunks on purpose
            meta = chk.read_final_meta(short)
            _, rk1, _, _ = _params(T1)
            assert meta is not None, "no final state kept"
            assert meta["k"] == MCF * rk1 and meta["MA_k_step"] == rk1, meta
            assert not os.path.exists(os.path.join(short, chk.STATE_FILE)), \
                "the rolling resume state should still be cleared"

            K2, rk2, digest2, params2 = _params(T2)
            chk.prepare_extension(short, longer, digest2, params2, rk2)
            D_x, A_x, s_x = _solve(T2, longer, resume=True, chunk=1234)

            what = "aligned" if aligned else "K1 % 5 != 0"
            assert np.array_equal(D_ref, D_x), f"{what}: D_LAYER differs"
            assert np.array_equal(A_ref, A_x), f"{what}: A_LAYER differs"
            for j, (r, x) in enumerate(zip(s_ref, s_x)):
                assert np.array_equal(r, x), \
                    f"{what}: timeseries {j} differs, first at sample " \
                    f"{int(np.argmax(r != x))} of {len(r)}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_extension_refuses_a_different_run():
    T1, T2 = _t_pair(False)
    tmp = tempfile.mkdtemp(prefix="final_state_refuse_")
    try:
        short = os.path.join(tmp, "short")
        _solve(T1, short)
        _, rk2, digest2, params2 = _params(T2)

        other = {**params2, "a": A_RATE * 2}
        d_other = chk.fingerprint(**other)[0]
        try:
            chk.prepare_extension(short, os.path.join(tmp, "x"), d_other, other, rk2)
        except ValueError as e:
            assert "different run" in str(e), e
        else:
            raise AssertionError("a final state from a different run was accepted")

        _, rk1, d1, p1 = _params(T1)
        try:
            chk.prepare_extension(short, os.path.join(tmp, "y"), d1, p1, rk1)
        except ValueError as e:
            assert "nothing to extend" in str(e), e
        else:
            raise AssertionError("extending to the same T was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ledger_records_and_resolves_job_states():
    """A job's final state lands in the ledger and is found by job id prefix."""
    tmp = tempfile.mkdtemp(prefix="ledger_")
    old = os.environ.get("ITCM_STATE_LEDGER")
    os.environ["ITCM_STATE_LEDGER"] = os.path.join(tmp, "ledger")
    try:
        T1, _ = _t_pair(True)
        state = os.path.join(tmp, "jobs", "deadbeef1234", "checkpoints", "pointA")
        _solve(T1, state)
        rows = state_ledger.entries()
        assert len(rows) == 1, rows
        e = rows[0]
        assert e["job_id"] == "deadbeef1234" and e["point"] == "pointA", e
        assert os.path.samefile(e["state_dir"], state)
        assert math.isclose(e["t_reached"], chk.read_final_meta(state)["t"])
        assert os.path.samefile(state_ledger.resolve("deadb", "pointA"), state)
        assert os.path.samefile(
            state_ledger.resolve(os.path.join(tmp, "jobs", "deadbeef1234",
                                              "checkpoints"), "pointA"), state)
        try:
            state_ledger.resolve("deadb", "pointB")
        except ValueError as err:
            assert "no saved final state" in str(err)
        else:
            raise AssertionError("an unknown point resolved")
    finally:
        if old is None:
            os.environ.pop("ITCM_STATE_LEDGER", None)
        else:
            os.environ["ITCM_STATE_LEDGER"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def _grid_through_launch(T, ckpt_dir, extend_from=None, device="cpu"):
    from launch_functions import launch

    old = os.environ.get("ITCM_CHECKPOINT_DIR")
    if ckpt_dir is None:
        os.environ.pop("ITCM_CHECKPOINT_DIR", None)
    else:
        os.environ["ITCM_CHECKPOINT_DIR"] = ckpt_dir
    try:
        r = launch.collect_ab_grid_char_time(
            16, 16, [10.0], [0.1, 1.0], 1.0, T, [0, 4, 8, 12], show_plt=False,
            workers=1, tau=0.01, device=device, extend_from=extend_from)
    finally:
        if old is None:
            os.environ.pop("ITCM_CHECKPOINT_DIR", None)
        else:
            os.environ["ITCM_CHECKPOINT_DIR"] = old
    return pd.read_csv(r["csv"]), r


COLS = ["t_star", "m_star", "fit_slope", "fit_intercept", "fit_window_t1",
        "fit_window_t2", "total_mass_at_t1", "total_mass_at_t2"]


def _same(a, b, rtol):
    for c in COLS:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        ok = both_nan | np.isclose(x, y, rtol=rtol, atol=0)
        assert ok.all(), f"{c}: {x} vs {y}"


def test_grid_job_extends_through_the_ledger():
    """End to end: a T = 0.5 grid job, extended to T = 1 by job id, equals T = 1."""
    tmp = tempfile.mkdtemp(prefix="grid_ext_")
    old = os.environ.get("ITCM_STATE_LEDGER")
    os.environ["ITCM_STATE_LEDGER"] = os.path.join(tmp, "ledger")
    try:
        _grid_through_launch(0.5, os.path.join(tmp, "jobs", "aaaa1111", "checkpoints"))
        assert len(state_ledger.entries()) == 2         # one per (a, b) point
        ext, _ = _grid_through_launch(1.0, os.path.join(tmp, "jobs", "bbbb2222",
                                                        "checkpoints"),
                                      extend_from="aaaa")
        fresh, _ = _grid_through_launch(1.0, None)
        _same(ext, fresh, rtol=0)                       # bit for bit on the CPU
        for p in ("a10_b0.1", "a10_b1"):
            with open(os.path.join(tmp, "jobs", "bbbb2222", "checkpoints",
                                   f"ab_grid_char_time_{p}", "extended_from.json")) as f:
                assert "aaaa1111" in json.load(f)["source"]
    finally:
        if old is None:
            os.environ.pop("ITCM_STATE_LEDGER", None)
        else:
            os.environ["ITCM_STATE_LEDGER"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_gpu_extension_matches_a_straight_gpu_run():
    from gpu import persistent_kernel as pk

    if not pk.is_available():
        print("no CUDA device; skipping")
        return
    tmp = tempfile.mkdtemp(prefix="grid_ext_gpu_")
    old = os.environ.get("ITCM_STATE_LEDGER")
    os.environ["ITCM_STATE_LEDGER"] = os.path.join(tmp, "ledger")
    try:
        _grid_through_launch(0.5, os.path.join(tmp, "jobs", "cccc3333", "checkpoints"),
                             device="gpu")
        ext, r = _grid_through_launch(1.0, os.path.join(tmp, "jobs", "dddd4444",
                                                        "checkpoints"),
                                      extend_from="cccc", device="gpu")
        assert r["device"] == "gpu"
        fresh, _ = _grid_through_launch(1.0, None, device="gpu")
        _same(ext, fresh, rtol=1e-9)
    finally:
        if old is None:
            os.environ.pop("ITCM_STATE_LEDGER", None)
        else:
            os.environ["ITCM_STATE_LEDGER"] = old
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
