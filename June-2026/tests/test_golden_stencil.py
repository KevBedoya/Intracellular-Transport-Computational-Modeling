"""Golden-master test for the PDE stencil.

``fixtures/golden_stencil.npz`` was generated from the solver as it stood at
commit 2bb8e34, immediately BEFORE the hot-path performance work.  Every case
below must still reproduce that state bit-for-bit.

This is the regression net for optimisation: any change that is meant to be
purely a speed-up has to leave these arrays untouched.  A deliberate change to
the physics is expected to fail here -- when that happens, regenerate the
fixture with ``python tests/test_golden_stencil.py --regenerate`` in the SAME
commit that changes the behaviour, and say why in the commit message.

Run directly (``python tests/test_golden_stencil.py``) or under pytest.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "project_src_package_2025")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from computational_tools import numerical_tools as num          # noqa: E402
from computational_tools import supplements as sup              # noqa: E402
from computational_tools import struct_init                     # noqa: E402

FIXTURE = os.path.join(_HERE, "fixtures", "golden_stencil.npz")

# Chosen to exercise distinct code paths, not just the default configuration:
#   * d_tube = 0 collapses the extraction loop in u_tube_rect to one iteration;
#     the non-zero values exercise the multi-step extraction region.
#   * non-square grids separate the ring count from the ray count.
CASES = [
    (48, 48, 8, 0.0),
    (48, 48, 8, 0.01),
    (48, 48, 4, 0.05),
    (64, 48, 8, 0.0),     # non-square
    (64, 48, 4, 0.02),    # non-square
    (32, 32, 16, 0.0),
    (32, 32, 16, 0.03),
    (96, 96, 24, 0.0),    # production configuration
]

STEPS = 30
W = 100.0
V = -float(10 ** 4)


def _run_case(rg, ry, n_tubes, d_tube):
    """Advance the stencil STEPS times; return (D_LAYER, A_LAYER, central)."""
    N_LIST = np.linspace(0, ry - (ry // n_tubes), n_tubes, dtype=int)
    dRad = num.compute_dRad(rg, 1.0)
    dThe = num.compute_dThe(ry)
    dT = num.compute_dT(rg, ry, 1.0, 1.0)
    central = num.compute_init_cond_cent(rg, 1.0)
    d_map = struct_init.build_d_tube_map_dense(rg, ry, N_LIST, d_tube, 1.0)

    D_LAYER, A_LAYER = sup.initialize_layers(rg, ry)
    for _ in range(STEPS):
        num.comp_DL_AL_kp1_2step(ry, rg, d_map, D_LAYER, central, A_LAYER,
                                 N_LIST, dRad, dThe, dT, W, W, V, d_tube)
        central = num.u_center(D_LAYER, 0, dRad, dThe, dT, central,
                               A_LAYER, N_LIST, V)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
    return D_LAYER, A_LAYER, central


def test_stencil_matches_golden():
    """Every case reproduces the recorded state exactly."""
    assert os.path.exists(FIXTURE), (
        f"missing golden fixture {FIXTURE}; regenerate with --regenerate")
    ref = np.load(FIXTURE)
    failures = []
    for idx, case in enumerate(CASES):
        D, A, c = _run_case(*case)
        label = f"{case[0]}x{case[1]} N={case[2]} d_tube={case[3]}"
        if not np.array_equal(ref[f"D{idx}"], D):
            failures.append(f"{label}: D_LAYER differs "
                            f"(max |delta| = {np.max(np.abs(ref[f'D{idx}'] - D)):.3e})")
        if not np.array_equal(ref[f"A{idx}"], A):
            failures.append(f"{label}: A_LAYER differs "
                            f"(max |delta| = {np.max(np.abs(ref[f'A{idx}'] - A)):.3e})")
        if ref[f"c{idx}"][0] != c:
            failures.append(f"{label}: central patch differs "
                            f"({ref[f'c{idx}'][0]!r} vs {c!r})")
    assert not failures, "stencil output changed:\n  " + "\n  ".join(failures)


def test_outermost_ring_cutoff_is_ring_indexed():
    """u_tube_rect must cut the advective flux at the last RING, not the last RAY.

    Regression test for a bug that was invisible on square grids: the check
    read `m == len(phi[k][m]) - 1`, comparing the ring index against the ray
    count.  On a grid with rings > rays that cut the flux at the wrong ring
    AND read one past the end of the array at the true outermost ring (an
    IndexError under NUMBA_BOUNDSCHECK=1, silent corruption without it).

    The golden cases above do not catch this: they start from a centred pulse
    and run only 30 steps, so the outer rings are still exactly zero and both
    branches agree.  This probe fills the domain so the branch actually shows.
    """
    rg, ry = 64, 48                      # rings > rays
    rho = np.zeros((2, rg, ry))
    phi = np.zeros((2, rg, ry))
    rho[0] = 1.0
    phi[0] = 1.0
    dRad, dThe, dT = 1.0 / rg, 2 * np.pi / ry, 1e-6

    def at(m):
        return num.u_tube_rect(rho, phi, 0, m, 0, 100.0, 100.0, -1e4,
                               dT, dRad, dThe, 0.0)

    # The cutoff sets j_r = 0, which changes the result substantially.
    assert at(rg - 1) < 0.5, "no flux cutoff at the outermost ring"
    # Neighbouring interior rings, including the old (wrong) ray-count index,
    # must all be uncut.
    for m in (ry - 2, ry - 1, ry, rg - 2):
        assert at(m) > 0.9, f"unexpected flux cutoff at interior ring m={m}"


def _regenerate():
    out = {}
    for idx, case in enumerate(CASES):
        D, A, c = _run_case(*case)
        out[f"D{idx}"] = D
        out[f"A{idx}"] = A
        out[f"c{idx}"] = np.array([c])
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.savez(FIXTURE, **out)
    print(f"regenerated {FIXTURE} ({len(CASES)} cases)")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        test_stencil_matches_golden()
        print(f"OK: all {len(CASES)} cases bit-identical to the golden fixture")
        test_outermost_ring_cutoff_is_ring_indexed()
        print("OK: outermost-ring flux cutoff is ring-indexed")
