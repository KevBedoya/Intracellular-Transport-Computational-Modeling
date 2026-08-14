"""
Standalone verification: do the Gram/cosine separability metrics hide
visually 'subpar' alignment in the normalized radial/angular trajectories?

Run:  python _verify_separability.py [RG] [RY] [T]
Default grid kept moderate so the explicit Euler march is fast; the
phenomenon (cosine ~1 while normalized overlays separate) is grid-independent.
"""
import sys, os
import numpy as np

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_src_package_2025")
sys.path.insert(0, PKG)

from launch_functions import launch
from computational_tools import mat_computations as mc

RG = int(sys.argv[1]) if len(sys.argv) > 1 else 24
RY = int(sys.argv[2]) if len(sys.argv) > 2 else 24
T  = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

# N=4 evenly spaced microtubules
N_LIST = [int(round(i * RY / 4)) for i in range(4)]
v_param = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
w_param = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

print(f"grid={RG}x{RY}  T={T}  N_LIST={N_LIST}  v={v_param} w={w_param}")

diff, central = launch.compute_ang_traj_mat(
    RG, RY, v_param, w_param, N_LIST,
    checkpoint_collect_container=[T],
    save_png=False, show_plt=False,
)

D = diff[0].astype(np.float64)          # (RG-1) x RY  truncated density matrix
print("D shape:", D.shape, " min/max:", D.min(), D.max())


def cosine_metrics(M):
    """Replicate the report's pipeline on column-normalized M."""
    X = M.copy()
    mc.normalize_columns(X)             # in-place unit-norm columns
    C = X.T @ X
    return C, mc.mean_alignment_error(C), mc.max_alignment_error(C)


# --- 1. Reproduce the report's diagnostics -------------------------------
C_A, A_mean, A_max = cosine_metrics(D.T)   # angular trajectories (cols of D^T)
C_B, B_mean, B_max = cosine_metrics(D)     # radial  trajectories (cols of D)
eta = mc.rank1_energy_ratio(D)

print("\n--- Report-style diagnostics (these SUGGEST separability) ---")
print(f"Angular  E_mean={A_mean:.3e}   E_max={A_max:.3e}")
print(f"Radial   E_mean={B_mean:.3e}   E_max={B_max:.3e}")
print(f"Rank-1 energy ratio eta={eta:.8f}   residual energy 1-eta={1-eta:.3e}")


# --- 2. What the eye actually sees on the normalized overlay -------------
def overlay_spread(M):
    """Normalize columns to unit norm, then per-row spread across columns.
    This is exactly what the trajectory plot overlays."""
    X = M.copy()
    mc.normalize_columns(X)
    # spread across trajectories at each sample point
    row_range = X.max(axis=1) - X.min(axis=1)
    # relative to the typical magnitude at that sample
    typ = np.abs(X).mean(axis=1) + 1e-300
    rel = row_range / typ
    return X, row_range, rel

XA, A_range, A_rel = overlay_spread(D.T)   # angular overlay
XB, B_range, B_rel = overlay_spread(D)     # radial overlay

# worst pairwise cosine -> angle and unit-vector L2 gap
def worst_pair(C):
    off = np.abs(C) - np.eye(C.shape[0])   # kill diagonal
    c = np.abs(C)[~np.eye(C.shape[0], dtype=bool)].min()
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    l2 = np.sqrt(2 * (1 - c))
    return c, ang, l2

cA, angA, l2A = worst_pair(C_A)
cB, angB, l2B = worst_pair(C_B)

print("\n--- What the normalized OVERLAY actually shows ---")
print("ANGULAR trajectories:")
print(f"  worst pairwise |cos|={cA:.6f}  -> angle={angA:.2f} deg, unit-L2 gap={l2A:.3f}")
print(f"  max per-sample spread across curves = {A_range.max():.3f}")
print(f"  max per-sample RELATIVE spread       = {A_rel.max()*100:.1f}%")
print("RADIAL trajectories:")
print(f"  worst pairwise |cos|={cB:.6f}  -> angle={angB:.2f} deg, unit-L2 gap={l2B:.3f}")
print(f"  max per-sample spread across curves = {B_range.max():.3f}")
print(f"  max per-sample RELATIVE spread       = {B_rel.max()*100:.1f}%")

# where does the cosine-invisible deviation live? (low-magnitude samples)
meanB = np.abs(XB).mean(axis=1)
order = np.argsort(meanB)
print("\n  radial samples sorted by magnitude (low->high), with rel spread:")
for i in order[:5]:
    print(f"    ring {i:2d}: |val|~{meanB[i]:.3e}   rel spread={B_rel[i]*100:6.1f}%")
for i in order[-3:]:
    print(f"    ring {i:2d}: |val|~{meanB[i]:.3e}   rel spread={B_rel[i]*100:6.1f}%")

# --- 3. Produce the actual plots the user referenced --------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    from data_visualization import plot_functions as pf
    pf.plt.ion = lambda *a, **k: None
    import matplotlib.pyplot as mpl
    sel_rings = list(range(0, D.shape[0], max(1, D.shape[0] // 6)))
    sel_rays  = list(range(0, D.shape[1], max(1, D.shape[1] // 6)))
    # patch plt.show to save instead
    orig_show = mpl.show
    saved = []
    def _save(*a, **k):
        for num in mpl.get_fignums():
            fp = f"_verify_traj_{num}.png"
            mpl.figure(num).savefig(fp, dpi=110, bbox_inches="tight")
            saved.append(fp)
        mpl.close("all")
    mpl.show = _save
    pf.plt.show = _save
    pf.plot_normalized_separable_trajectories(
        D.copy(), v_param, w_param, T, len(N_LIST), float(central[0]),
        selected_rings=sel_rings, selected_rays=sel_rays,
        include_central_patch=False,
    )
    print("\nSaved plots:", saved)
except Exception as e:
    print("\n[plot step skipped]:", repr(e))
