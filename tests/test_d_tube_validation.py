"""d_tube is checked against the solver's own bound, at submission and in the UI.

The solver supports only non-overlapping extraction regions, and it does not
refuse a d_tube outside [0, limit]: build_d_tube_mapping_no_overlap silently
replaces it with the limit. So a job submitted with such a value would run a
width nobody asked for. These tests pin three things:

  * the limit the router computes is the one the solver applies, exactly;
  * POST /jobs refuses an out-of-range d_tube, naming the field;
  * POST /helpers/d_tube -- what the form asks while you type -- agrees.

Run directly (``python tests/test_d_tube_validation.py``) or under pytest.
"""
import os
import sys
import tempfile

os.environ.setdefault("ITCM_OUTPUT_ROOT",
                      os.path.join(tempfile.gettempdir(), "itcm_ab_grid_tests"))
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src", "intracellular_transport"),
           os.path.join(_ROOT, "src"), os.path.join(_ROOT, "server")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                             # noqa: E402

from computational_tools import struct_init                    # noqa: E402
from computational_tools import supplements as sup             # noqa: E402
from multiprocessing_tools import computation_router as router  # noqa: E402

CONFIGS = [
    (16, 16, [0, 4, 8, 12]),
    (96, 96, list(range(0, 96, 4))),
    (32, 48, [0, 5, 17, 30]),          # uneven gaps, non-square grid
    (32, 32, [0, 1]),                  # adjacent tubes: the tightest case
]

BASE = dict(rg_param=16, ry_param=16, v_param=10.0, w_param=10.0,
            T_param=1.0, N_LIST=[0, 4, 8, 12])
COMPUTATION = "Compute MFPT until time T"


def _solver_limit(rg, ry, n_list):
    """The bound as build_d_tube_mapping_no_overlap computes it."""
    j_sup = sup.j_max_bef_overlap(ry, np.array(n_list, dtype=np.int64))
    return sup.solve_d_rect(1, rg, ry, j_sup, 0)


def test_limit_is_the_solvers_limit():
    for rg, ry, n_list in CONFIGS:
        assert router.d_tube_limit(rg, ry, n_list) == _solver_limit(rg, ry, n_list), \
            (rg, ry, n_list)


def test_boundary_is_the_solvers_clamp_test():
    for rg, ry, n_list in CONFIGS:
        limit = router.d_tube_limit(rg, ry, n_list)
        assert router.check_d_tube(rg, ry, n_list, 0.0)["valid"]
        assert router.check_d_tube(rg, ry, n_list, limit)["valid"]
        assert router.check_d_tube(rg, ry, n_list, limit / 2)["valid"]
        assert not router.check_d_tube(rg, ry, n_list, limit * (1 + 1e-12))["valid"]
        assert not router.check_d_tube(rg, ry, n_list, -1e-9)["valid"]


def test_an_invalid_value_really_is_clamped_by_the_solver():
    """The reason this check exists: past the limit the solver quietly uses
    the limit, so two different requests produce the same physics."""
    rg, ry, n_list = CONFIGS[0]
    limit = router.d_tube_limit(rg, ry, n_list)
    n = np.array(n_list, dtype=np.int64)
    at_limit = struct_init.build_d_tube_map_dense(rg, ry, n, limit)
    beyond = struct_init.build_d_tube_map_dense(rg, ry, n, 10 * limit)
    negative = struct_init.build_d_tube_map_dense(rg, ry, n, -1.0)
    assert (beyond == at_limit).all() and (negative == at_limit).all()


def test_submission_refuses_out_of_range_d_tube():
    limit = router.d_tube_limit(16, 16, BASE["N_LIST"])
    assert router.validate_params(COMPUTATION, {**BASE, "d_tube": 0.0}) == {}
    assert router.validate_params(COMPUTATION, {**BASE, "d_tube": limit}) == {}
    for bad in (limit * 1.01, 1.0, -0.01):
        errors = router.validate_params(COMPUTATION, {**BASE, "d_tube": bad})
        assert set(errors) == {"d_tube"}, errors
        assert f"{limit:.10g}" in errors["d_tube"], errors["d_tube"]


def test_d_tube_is_not_judged_against_a_broken_n_list():
    """A bad N_LIST is the error to report; d_tube cannot be judged without it."""
    errors = router.validate_params(
        COMPUTATION, {**BASE, "N_LIST": [0, 4, 4], "d_tube": 5.0})
    assert "N_LIST" in errors and "d_tube" not in errors, errors


def test_helper_endpoint_agrees_with_submission():
    import api

    client = api.app.test_client()
    limit = router.d_tube_limit(16, 16, BASE["N_LIST"])

    def ask(**params):
        r = client.post("/helpers/d_tube", json={"params": params})
        assert r.status_code == 200, r.get_data(as_text=True)
        return r.get_json()

    grid = dict(rg_param=16, ry_param=16, N_LIST=BASE["N_LIST"])
    ok = ask(**grid, d_tube=limit / 2)
    assert ok["status"] == "valid" and ok["max_d_tube"] == limit, ok
    bad = ask(**grid, d_tube=2 * limit)
    assert bad["status"] == "invalid" and "silently" in bad["message"], bad
    assert ask(**grid)["status"] == "incomplete"
    assert ask(rg_param=16, ry_param=16, d_tube=0.01)["status"] == "incomplete"
    assert ask(rg_param=16, ry_param=16, N_LIST=[0, 99],
               d_tube=0.01)["status"] == "incomplete"

    assert client.post("/helpers/d_tube", json={}).status_code == 400

    # POST /jobs applies the same rule, so the form cannot be bypassed. The
    # refusal happens before anything is written to the job store.
    r = client.post("/jobs", json={"computation": COMPUTATION,
                                   "params": {**BASE, "d_tube": 2 * limit}})
    assert r.status_code == 400
    assert "d_tube" in r.get_json()["fields"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_")]
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
