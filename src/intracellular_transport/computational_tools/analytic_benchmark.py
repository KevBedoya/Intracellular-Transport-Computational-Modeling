"""
Benchmark harness: numerical PDE solver vs. analytic solution (eqs 24 & 25).

Runs the existing numerical collectors to obtain phi(r,theta,t) and rho(r,t)
trajectories, evaluates the analytic solution from :mod:`analytic_solution` on
the *same* polar grid and at the *same* realized (floor'd) checkpoint times, and
reports error metrics + overlay plots.

Pure Python/NumPy (no numba).  The numerical collectors it calls ARE numba-jitted;
this module treats them as opaque array-fillers and does all comparison in NumPy.

Reconciling the discrete coupling
---------------------------------
The discrete scheme uses scalar ``w_param`` for both on/off rates but carries
geometric 1/(r*dThe) factors (see plan).  ``derive_continuum_params`` maps
``w_param`` -> continuum (a, b); we lead reporting with coupling-robust
observables (angular mean g(r), decay rate sigma, normalized a(r)).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write PNGs, no display
import matplotlib.pyplot as plt

# Numerical side (numba-jitted collectors + grid helpers)
from computational_tools import numerical_tools as num
from computational_tools import supplements as sup
from computational_tools import analysis_tools as ant
from computational_tools import analytic_solution as A


# ----------------------------------------------------------------------------
# Numerical runs (reuse existing collectors, return raw arrays)
# ----------------------------------------------------------------------------
def _realized_times(checkpoints, rg_param, ry_param, domain_radius=1.0, D=1.0):
    """Map requested checkpoint times to the times the solver actually samples.

    The collectors fire a snapshot at k == floor(t/dT), i.e. at time
    floor(t/dT)*dT.  We must compare the analytic field at THESE times.
    """
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    stamps = np.floor(np.asarray(checkpoints, float) / dT)
    return stamps * dT, dT


def run_numerical_phi(rg_param, ry_param, N_LIST, v_param, w_param, checkpoints,
                      domain_radius=1.0, D=1.0, d_tube=0.0, b_param=None):
    """Run the heatmap collector; return full phi field snapshots + center.

    Returns dict with HM_DL (n_chk, rg, ry), HM_C (n_chk,), realized_times (n_chk,).
    Mirrors launch.heatmap_production_time_dep's setup exactly.  ``w_param`` is the
    on-rate a; ``b_param`` the off-rate b (defaults to w_param).
    """
    b_eff = w_param if b_param is None else b_param
    N_LIST = sorted(int(x) for x in N_LIST)
    chk = sorted(float(x) for x in checkpoints)
    realized, dT = _realized_times(chk, rg_param, ry_param, domain_radius, D)

    # T_param must exceed the largest stamp; pad like the launch wrapper.
    T_param = max(chk) + len(chk) * dT + 5 * dT

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    n = len(chk)
    HM_DL = np.zeros((n, rg_param, ry_param), dtype=np.float64)
    HM_C = np.zeros(n, dtype=np.float64)
    MFPT = np.zeros(n, dtype=np.float64)

    ant.comp_diffusive_snapshots_time_dep(
        rg_param, ry_param, w_param, b_eff, v_param, T_param, N_LIST,
        D_LAYER, A_LAYER, HM_DL, HM_C, MFPT, chk, domain_radius, D, 10 ** 9, d_tube)

    return {"HM_DL": HM_DL, "HM_C": HM_C, "realized_times": realized,
            "checkpoints": np.asarray(chk), "dT": dT}


def run_numerical_radial(rg_param, ry_param, N_LIST, v_param, w_param, checkpoints,
                         R_fixed_angle=0, domain_radius=1.0, D=1.0, d_tube=0.0,
                         b_param=None):
    """Run the radial collector at a fixed angle; return phi(r) and rho(r) profiles.

    Returns dict: PvR (n_chk, rg+1) [index 0 = center], RvR (n_chk, rg),
    realized_times.  ``w_param``=on-rate a, ``b_param``=off-rate b (default w_param).
    """
    b_eff = w_param if b_param is None else b_param
    N_LIST = sorted(int(x) for x in N_LIST)
    chk = sorted(float(x) for x in checkpoints)
    realized, dT = _realized_times(chk, rg_param, ry_param, domain_radius, D)
    T_param = max(chk) + len(chk) * dT + 5 * dT

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    n = len(chk)
    PvR = np.zeros((n, rg_param + 1), dtype=np.float64)
    RvR = np.zeros((n, rg_param), dtype=np.float64)

    ant.comp_diffusive_rad_snapshots_time_dep(
        rg_param, ry_param, w_param, b_eff, v_param, T_param, N_LIST,
        D_LAYER, A_LAYER, R_fixed_angle, PvR, RvR, chk, domain_radius, D, 10 ** 9, d_tube)

    return {"PvR": PvR, "RvR": RvR, "realized_times": realized,
            "checkpoints": np.asarray(chk), "dT": dT}


def run_numerical_angular(rg_param, ry_param, N_LIST, v_param, w_param, checkpoints,
                          T_fixed_ring_frac=0.5, domain_radius=1.0, D=1.0, d_tube=0.0,
                          b_param=None):
    """Run the angular collector at a fixed ring; return phi(theta) profiles.

    Returns dict: PvT (n_chk, ry) phi(theta) at ring floor(rg*frac), ring_index,
    realized_times.  ``w_param``=on-rate a, ``b_param``=off-rate b (default w_param).
    """
    b_eff = w_param if b_param is None else b_param
    N_LIST = sorted(int(x) for x in N_LIST)
    chk = sorted(float(x) for x in checkpoints)
    realized, dT = _realized_times(chk, rg_param, ry_param, domain_radius, D)
    T_param = max(chk) + len(chk) * dT + 5 * dT

    D_LAYER, A_LAYER = sup.initialize_layers(rg_param, ry_param)
    n = len(chk)
    PvT = np.zeros((n, ry_param), dtype=np.float64)

    ant.comp_diffusive_angle_snapshots_time_dep(
        rg_param, ry_param, w_param, b_eff, T_param, v_param, N_LIST,
        D_LAYER, A_LAYER, PvT, chk, T_fixed_ring_frac, d_tube, domain_radius, D, 10 ** 9)

    ring_index = int(np.floor(rg_param * T_fixed_ring_frac))
    return {"PvT": PvT, "ring_index": ring_index, "realized_times": realized,
            "checkpoints": np.asarray(chk), "dT": dT}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def l2_relative_error(num_field, ana_field, rg_param, ry_param, domain_radius=1.0):
    """Area-weighted relative L2 error over the disk (weight r_m*dRad*dThe)."""
    dRad = domain_radius / rg_param
    dThe = 2.0 * np.pi / ry_param
    r = (np.arange(rg_param) + 1) * dRad
    w = (r * dRad * dThe)[:, None]            # area element, broadcast over theta
    diff2 = np.sum(w * (num_field - ana_field) ** 2)
    ref2 = np.sum(w * num_field ** 2)
    return np.sqrt(diff2 / ref2) if ref2 > 0 else np.nan


def angular_mean_profile(field_2d):
    """g(r): mean over theta."""
    return field_2d.mean(axis=1)


def cosNtheta_amplitude(field_2d, N, theta):
    """a(r): amplitude of the cos(N theta) component (projection)."""
    return (2.0 / len(theta)) * (field_2d @ np.cos(N * theta))


def decay_rate_fit(norm_series, times):
    """Dominant decay rate sigma from slope of log||.|| vs t (returns sigma>0)."""
    times = np.asarray(times, float)
    y = np.log(np.asarray(norm_series, float))
    slope, _ = np.polyfit(times, y, 1)
    return -slope


def field_l2_norm(field_2d, rg_param, ry_param, domain_radius=1.0):
    dRad = domain_radius / rg_param
    dThe = 2.0 * np.pi / ry_param
    r = (np.arange(rg_param) + 1) * dRad
    w = (r * dRad * dThe)[:, None]
    return np.sqrt(np.sum(w * field_2d ** 2))


# ----------------------------------------------------------------------------
# Analytic field on the matching grid (uncoupled / Stage 0)
# ----------------------------------------------------------------------------
def analytic_phi_uncoupled_field(rg_param, ry_param, realized_times,
                                  n_modes=60, domain_radius=1.0, ic="patch"):
    """Build analytic phi(r,theta,t) snapshots (n_chk, rg, ry) for a=b=0.

    Angularly symmetric, so each column is identical.  Also returns the center
    values phi(0,t).
    """
    r_centers, _, theta = A.build_eval_grid(rg_param, ry_param, domain_radius)
    coeffs, kappa = A.fourier_bessel_coeffs_uncoupled(rg_param, n_modes,
                                                      domain_radius, ic=ic)
    n = len(realized_times)
    HM = np.zeros((n, rg_param, ry_param))
    center = np.zeros(n)
    for i, t in enumerate(realized_times):
        prof = A.phi_uncoupled(r_centers, t, coeffs, kappa, domain_radius)
        HM[i] = prof[:, None]                      # broadcast across theta
        center[i] = A.phi_uncoupled(np.array([0.0]), t, coeffs, kappa,
                                    domain_radius)[0]
    return HM, center


# ----------------------------------------------------------------------------
# Staged report
# ----------------------------------------------------------------------------
def benchmark_stage0(rg_param, ry_param=None, checkpoints=(0.02, 0.05, 0.1, 0.2, 0.3),
                     n_modes=60, domain_radius=1.0, ic="patch", verbose=True):
    """Stage 0: uncoupled (w=0, v=0) numerical vs. Fourier-Bessel analytic.

    Returns a results dict (per-checkpoint L2 rel err, center err, sigma_num,
    sigma_analytic).
    """
    if ry_param is None:
        ry_param = rg_param
    N_LIST = [0]                                   # placeholder MT (coupling off)
    chk = sorted(float(x) for x in checkpoints)

    numres = run_numerical_phi(rg_param, ry_param, N_LIST, v_param=0.0,
                               w_param=0.0, checkpoints=chk,
                               domain_radius=domain_radius)
    realized = numres["realized_times"]
    HM_num = numres["HM_DL"]
    C_num = numres["HM_C"]

    HM_ana, C_ana = analytic_phi_uncoupled_field(
        rg_param, ry_param, realized, n_modes=n_modes,
        domain_radius=domain_radius, ic=ic)

    l2 = np.array([l2_relative_error(HM_num[i], HM_ana[i], rg_param, ry_param,
                                     domain_radius) for i in range(len(chk))])
    center_rel = np.abs(C_num - C_ana) / np.abs(C_ana)

    # decay rates from field L2 norm
    norms_num = np.array([field_l2_norm(HM_num[i], rg_param, ry_param, domain_radius)
                          for i in range(len(chk))])
    norms_ana = np.array([field_l2_norm(HM_ana[i], rg_param, ry_param, domain_radius)
                          for i in range(len(chk))])
    # use the late-time window (last 3 points) for a clean single-mode slope
    win = slice(max(0, len(chk) - 3), len(chk))
    sigma_num = decay_rate_fit(norms_num[win], realized[win])
    sigma_ana = decay_rate_fit(norms_ana[win], realized[win])
    sigma_theory = A.uncoupled_decay_rate(domain_radius)

    res = {"grid": (rg_param, ry_param), "realized_times": realized,
           "l2_rel_err": l2, "center_rel_err": center_rel,
           "sigma_num": sigma_num, "sigma_ana": sigma_ana,
           "sigma_theory": sigma_theory,
           "HM_num": HM_num, "HM_ana": HM_ana}

    if verbose:
        print(f"\n=== Stage 0 (uncoupled) grid {rg_param}x{ry_param}, ic={ic} ===")
        print(f"{'t':>8} {'L2 rel err':>12} {'center rel':>12}")
        for i in range(len(chk)):
            print(f"{realized[i]:8.4f} {l2[i]:12.4e} {center_rel[i]:12.4e}")
        print(f"decay sigma: numerical={sigma_num:.4f}  "
              f"analytic={sigma_ana:.4f}  theory={sigma_theory:.4f}")
    return res


def _shape_overlap(u, v):
    """Cosine similarity of two real profiles (sign-insensitive shape match)."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.abs(u @ v) / (nu * nv))


def benchmark_coupled(rg_param, ry_param, N_LIST, v_param, w_param,
                      checkpoints=(0.05, 0.1, 0.2, 0.3), domain_radius=1.0,
                      stage="1", verbose=True):
    """Coupled benchmark via robust observables + Bessel structural shapes.

    The companion derivations prove the coupled operator has no low-mode
    separable eigenstructure (cos(N theta) has zero theta-derivative at the MT
    ray, so it cannot carry the MT cusp).  A tight analytic decay rate therefore
    requires the full non-self-adjoint resolvent spectrum.  Here we instead
    validate the *structure* predicted by eq (24):
      phi ~ e^{-sigma t} [ g(r) + a(r) cos(N theta) + ... ],  g~J_0, a~J_N,
    by extracting g(r), a(r), the angular energy split, and the decay rate from
    the numerical run and comparing the channel shapes to the analytic Bessel
    profiles.  N = number of microtubules = len(N_LIST).
    """
    N = len(N_LIST)
    chk = sorted(float(x) for x in checkpoints)
    r_centers, _, theta = A.build_eval_grid(rg_param, ry_param, domain_radius)

    numres = run_numerical_phi(rg_param, ry_param, N_LIST, v_param, w_param,
                               chk, domain_radius=domain_radius)
    realized = numres["realized_times"]
    HM = numres["HM_DL"]

    # rho(r,t) on an MT ray (eq 25): use the radial collector at an MT angle
    radres = run_numerical_radial(rg_param, ry_param, N_LIST, v_param, w_param,
                                  chk, R_fixed_angle=N_LIST[0],
                                  domain_radius=domain_radius)
    RvR = radres["RvR"]                                   # (n, rg) rho(r)
    PvR = radres["PvR"]                                    # (n, rg+1) phi(r) at MT ray
    rho_norms = np.array([np.linalg.norm(RvR[i]) for i in range(len(chk))])
    rho_wall = np.abs(RvR[:, -1])                         # rho at r=1 (should ~0)

    # phi decay rate from field L2 norm (late window) -- needed by the AL functional
    norms = np.array([field_l2_norm(HM[i], rg_param, ry_param, domain_radius)
                      for i in range(len(chk))])
    win = slice(max(0, len(chk) - 3), len(chk))
    sigma_num = decay_rate_fit(norms[win], realized[win])
    sigma_uncoupled = A.uncoupled_decay_rate(domain_radius)

    # Direct eq-(25) check: apply analytic AL functional R[phi(.,0)] to the
    # numerical phi MT-ray trace and compare shape to numerical rho.  The solver
    # negates v_param internally, so test both signs and keep the better match.
    rho_func_overlap = np.zeros(len(chk))
    rho_analytic = np.zeros_like(RvR)                     # best-match R[phi] profile
    dThe = 2.0 * np.pi / ry_param
    for i in range(len(chk)):
        phi_ring = PvR[i, 1:]                              # drop center, ring grid
        best = 0.0
        for vsign in (v_param, -v_param):
            if vsign == 0:
                continue
            rho_a = A.al_functional(phi_ring, r_centers, sigma_num,
                                    w_param, w_param, vsign, domain_radius, dThe=dThe)
            ov = _shape_overlap(RvR[i], rho_a)
            if ov > best:
                best = ov
                rho_analytic[i] = rho_a
        rho_func_overlap[i] = best

    # per-checkpoint channel extraction
    g = np.array([angular_mean_profile(HM[i]) for i in range(len(chk))])      # (n,rg)
    a = np.array([cosNtheta_amplitude(HM[i], N, theta) for i in range(len(chk))])
    # angular energy split: ||mean||^2 vs ||cosN component||^2 (area weighted)
    dRad = domain_radius / rg_param
    wr = (np.arange(rg_param) + 1) * dRad * dRad * (2 * np.pi / ry_param)
    e_mean = np.array([np.sum(wr * g[i] ** 2) for i in range(len(chk))])
    e_cosN = np.array([0.5 * np.sum(wr * a[i] ** 2) for i in range(len(chk))])
    frac_cosN = e_cosN / (e_mean + e_cosN)

    # structural shape comparison at the latest checkpoint (cleanest single-mode)
    g_ref = A.bessel_mode_shape(0, 1, r_centers, domain_radius)
    aN_ref = A.bessel_mode_shape(N, 1, r_centers, domain_radius)
    g_overlap = _shape_overlap(g[-1], g_ref)
    a_overlap = _shape_overlap(a[-1], aN_ref)

    res = {"grid": (rg_param, ry_param), "N": N, "realized_times": realized,
           "g": g, "a": a, "frac_cosN": frac_cosN,
           "sigma_num": sigma_num, "sigma_uncoupled": sigma_uncoupled,
           "g_shape_overlap_J0": g_overlap, "a_shape_overlap_JN": a_overlap,
           "RvR": RvR, "rho_norms": rho_norms, "rho_wall": rho_wall,
           "rho_func_overlap": rho_func_overlap, "rho_analytic": rho_analytic,
           "HM": HM, "r_centers": r_centers}

    if verbose:
        print(f"\n=== Coupled Stage {stage}: grid {rg_param}x{ry_param}, "
              f"N={N}, w={w_param}, v={v_param} ===")
        print(f"{'t':>8} {'||phi||_2':>12} {'cosN frac':>12} {'max|a(r)|':>12}")
        for i in range(len(chk)):
            print(f"{realized[i]:8.4f} {norms[i]:12.4e} {frac_cosN[i]:12.4e} "
                  f"{np.max(np.abs(a[i])):12.4e}")
        print(f"decay sigma (phi): numerical={sigma_num:.4f}  "
              f"uncoupled kappa01^2={sigma_uncoupled:.4f}  "
              f"shift={sigma_num - sigma_uncoupled:+.4f}")
        print(f"channel shape overlap (cosine sim):  "
              f"g(r) vs J_0={g_overlap:.4f}   a(r) vs J_{N}={a_overlap:.4f}")
        print(f"eq(25) rho: R[phi] shape overlap (latest t)={rho_func_overlap[-1]:.4f}; "
              f"max rho(r=1)={np.max(rho_wall):.2e} (BC rho(1)=0); "
              f"max|rho|={np.max(np.abs(RvR)):.3e}")
    return res


# ----------------------------------------------------------------------------
# Overlay plots (numerical vs analytic)
# ----------------------------------------------------------------------------
_DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "..",
                               "results", "analytic_benchmark")


def _ensure_outdir(outdir):
    outdir = outdir or _DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_stage0_overlay(res, outdir=None, domain_radius=1.0):
    """phi(r) numerical (points) vs analytic Fourier-Bessel (lines) per checkpoint."""
    outdir = _ensure_outdir(outdir)
    rg, ry = res["grid"]
    r_centers, _, _ = A.build_eval_grid(rg, ry, domain_radius)
    HM_num, HM_ana = res["HM_num"], res["HM_ana"]
    times = res["realized_times"]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(times):
        col = cmap(i / max(1, len(times) - 1))
        gnum = HM_num[i].mean(axis=1)
        gana = HM_ana[i].mean(axis=1)
        ax.plot(r_centers, gnum, "o", ms=3, color=col, alpha=0.7)
        ax.plot(r_centers, gana, "-", color=col, label=f"t={t:.3f}")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\phi(r)$")
    ax.set_title(f"Stage 0 uncoupled: numerical (o) vs analytic (-), {rg}x{ry}")
    ax.legend(fontsize=8)
    path = os.path.join(outdir, f"stage0_phi_overlay_{rg}x{ry}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_coupled_overlay(res, w_param, v_param, outdir=None, domain_radius=1.0,
                         stage="1"):
    """Three-panel overlay: g(r) vs J_0, a(r) vs J_N, rho(r) vs R[phi]."""
    outdir = _ensure_outdir(outdir)
    rg, ry = res["grid"]
    N = res["N"]
    r = res["r_centers"]
    times = res["realized_times"]
    g, a, RvR = res["g"], res["a"], res["RvR"]

    g_ref = A.bessel_mode_shape(0, 1, r, domain_radius)
    aN_ref = A.bessel_mode_shape(N, 1, r, domain_radius)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # g(r): normalized numerical vs J_0
    gl = g[-1] / np.max(np.abs(g[-1]))
    axes[0].plot(r, gl, "o", ms=3, label="numerical g(r) (norm)")
    axes[0].plot(r, g_ref * np.sign(gl[0]), "-", label=r"analytic $J_0(\kappa_{0,1}r)$")
    axes[0].set_title(f"angular-mean g(r) [overlap {res['g_shape_overlap_J0']:.3f}]")
    axes[0].set_xlabel("r"); axes[0].legend(fontsize=8)
    # a(r): normalized numerical vs J_N
    al = a[-1] / np.max(np.abs(a[-1]))
    axes[1].plot(r, al, "o", ms=3, label="numerical a(r) (norm)")
    axes[1].plot(r, aN_ref * np.sign(al[np.argmax(np.abs(al))]), "-",
                 label=rf"analytic $J_{{{N}}}(\kappa_{{{N},1}}r)$")
    axes[1].set_title(f"cos(N$\\theta$) amplitude a(r) [overlap {res['a_shape_overlap_JN']:.3f}]")
    axes[1].set_xlabel("r"); axes[1].legend(fontsize=8)
    # rho(r): numerical vs analytic AL functional R[phi] (scaled to numerical peak)
    rho_num = RvR[-1]
    rho_ana = res["rho_analytic"][-1]
    # signed least-squares amplitude (the AL functional's overall sign depends on
    # the v / (a/v) convention; the reported overlap is sign-insensitive cosine sim)
    denom = float(rho_ana @ rho_ana)
    sc = float(rho_num @ rho_ana) / denom if denom > 0 else 1.0
    axes[2].plot(r, rho_num, "o", ms=3, label=r"numerical $\rho(r)$")
    axes[2].plot(r, rho_ana * sc, "-", label=r"analytic $R[\phi]$ (scaled)")
    axes[2].set_title(f"AL density $\\rho(r)$ [R[phi] overlap {res['rho_func_overlap'][-1]:.3f}]")
    axes[2].set_xlabel("r"); axes[2].legend(fontsize=8)
    fig.suptitle(f"Coupled Stage {stage}: {rg}x{ry}, N={N}, w={w_param}, v={v_param}  "
                 f"(latest t={times[-1]:.3f})")
    path = os.path.join(outdir, f"coupled_stage{stage}_overlay_{rg}x{ry}_N{N}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def _lsq_scale(target, model):
    """Signed least-squares amplitude that best fits `model` to `target`."""
    target = np.asarray(target, float); model = np.asarray(model, float)
    d = float(model @ model)
    return float(target @ model) / d if d > 0 else 1.0


def plot_verification_trajectories(rg_param=32, ry_param=32, N_LIST=(0, 8, 16, 24),
                                   v_param=10.0, w_param=10.0,
                                   checkpoints=(0.05, 0.1, 0.2, 0.3),
                                   R_fixed_angle=None, T_fixed_ring_frac=0.5,
                                   domain_radius=1.0, D=1.0,
                                   outdir="verification_plots", n_modes=80):
    """Overlay numerical vs analytic radial and angular trajectories (separate plots).

    Produces two PNGs (no display) under ``outdir``:
      * radial  : phi(r) at a fixed (microtubule) angle and rho(r), numerical
                  (points) vs analytic (lines), per checkpoint time.
      * angular : the angular fluctuation  phi(theta) - <phi>_theta  at a fixed
                  ring, numerical (points) vs analytic A*cos(N theta) (lines).

    Analytic implementation (matched to the numerical defaults R=1, D=1, central
    -patch delta IC, realized floor(t/dT) sample times):
      - radial mean phi(r,t): leading diffusive Fourier-Bessel series
        ``analytic_solution.phi_uncoupled`` (J_0 channel of eq 24).
      - rho(r,t): analytic AL functional ``al_functional`` (R[phi] of eq 25).
      - angular fluctuation: the cos(N theta) modulation channel of eq 24.
    Each analytic curve is amplitude-matched to its numerical counterpart by
    signed least-squares (the absolute modulation amplitude requires the coupled
    secular-root solve; see analytic-benchmark notes), so the overlay verifies
    the analytic *functional form*.  N = number of microtubules = len(N_LIST).
    """
    N = len(N_LIST)
    N_LIST = sorted(int(x) for x in N_LIST)
    if R_fixed_angle is None:
        R_fixed_angle = N_LIST[0]                        # a microtubule ray
    chk = sorted(float(x) for x in checkpoints)
    os.makedirs(outdir, exist_ok=True)

    r_centers, _, theta = A.build_eval_grid(rg_param, ry_param, domain_radius)
    coeffs, kappa = A.fourier_bessel_coeffs_uncoupled(rg_param, n_modes,
                                                      domain_radius, ic="patch")

    # ---- numerical runs ----
    rad = run_numerical_radial(rg_param, ry_param, N_LIST, v_param, w_param, chk,
                               R_fixed_angle=R_fixed_angle, domain_radius=domain_radius, D=D)
    ang = run_numerical_angular(rg_param, ry_param, N_LIST, v_param, w_param, chk,
                                T_fixed_ring_frac=T_fixed_ring_frac,
                                domain_radius=domain_radius, D=D)
    times = rad["realized_times"]
    PvR, RvR = rad["PvR"], rad["RvR"]
    PvT = ang["PvT"]
    ring_m = ang["ring_index"]
    r_ring = (ring_m + 1) * (domain_radius / rg_param)

    cmap = plt.get_cmap("viridis")
    ncol = max(1, len(chk) - 1)

    # ===================== RADIAL figure =====================
    fig_r, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, t in enumerate(times):
        col = cmap(i / ncol)
        # numerical phi(r) at the MT ray (drop center index 0)
        phi_num = PvR[i, 1:]
        # analytic radial diffusive profile (J_0 channel), scaled to numerical
        phi_ana = A.phi_uncoupled(r_centers, t, coeffs, kappa, domain_radius)
        s = _lsq_scale(phi_num, phi_ana)
        axes[0].plot(r_centers, phi_num, "o", ms=3, color=col, alpha=0.7)
        axes[0].plot(r_centers, phi_ana * s, "-", color=col, label=f"t={t:.3f}")
        # rho(r): numerical vs analytic AL functional applied to analytic phi.
        # dThe weights the on-rate by r*dThe to match the scheme's coupling
        # asymmetry (reproduces the off-center rho peak); see al_functional.
        rho_num = RvR[i]
        best, rho_ana_best = -1.0, None
        sigma = A.uncoupled_decay_rate(domain_radius)
        dThe = 2.0 * np.pi / ry_param
        for vs in (v_param, -v_param):
            if vs == 0:
                continue
            ra = A.al_functional(phi_ana, r_centers, sigma, w_param, w_param, vs,
                                 domain_radius, dThe=dThe)
            ov = abs(_shape_overlap(rho_num, ra))
            if ov > best:
                best, rho_ana_best = ov, ra
        sr = _lsq_scale(rho_num, rho_ana_best)
        axes[1].plot(r_centers, rho_num, "o", ms=3, color=col, alpha=0.7)
        axes[1].plot(r_centers, rho_ana_best * sr, "-", color=col, label=f"t={t:.3f}")
    axes[0].set_xlabel("r"); axes[0].set_ylabel(r"$\phi(r,\theta_{MT})$")
    axes[0].set_title("Radial DL trajectory: numerical (o) vs analytic (-)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("r"); axes[1].set_ylabel(r"$\rho(r)$")
    axes[1].set_title("Radial AL trajectory: numerical (o) vs analytic R[$\\phi$] (-)")
    axes[1].legend(fontsize=8)
    fig_r.suptitle(f"Radial trajectories  {rg_param}x{ry_param}, N={N}, v={v_param}, "
                   f"a=b=w={w_param}, R={domain_radius}, D={D}  (MT ray)")
    radial_path = os.path.join(outdir, f"radial_traj_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}.png")
    fig_r.tight_layout(); fig_r.savefig(radial_path, dpi=130); plt.close(fig_r)

    # ===================== ANGULAR figure =====================
    fig_a, ax = plt.subplots(figsize=(8, 5))
    deg = np.degrees(theta)
    cosNt = np.cos(N * theta)
    for i, t in enumerate(times):
        col = cmap(i / ncol)
        # numerical angular fluctuation at the fixed ring
        prof = PvT[i]
        fluct_num = prof - prof.mean()
        # analytic modulation: A * cos(N theta), amplitude = LSQ fit to numerical
        amp = _lsq_scale(fluct_num, cosNt)
        fluct_ana = amp * cosNt
        ax.plot(deg, fluct_num, "o", ms=3, color=col, alpha=0.7)
        ax.plot(deg, fluct_ana, "-", color=col, label=f"t={t:.3f}")
    # mark microtubule angular positions
    for nl in N_LIST:
        ax.axvline(np.degrees(theta[nl]), color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel(r"$\phi(\theta) - \langle\phi\rangle_\theta$  (angular fluctuation)")
    ax.set_title(f"Angular DL trajectory at r={r_ring:.3f}: numerical (o) vs "
                 f"analytic $A\\cos({N}\\theta)$ (-)\n"
                 f"{rg_param}x{ry_param}, N={N}, v={v_param}, a=b=w={w_param} "
                 f"(dotted = MT rays)")
    ax.legend(fontsize=8)
    angular_path = os.path.join(outdir, f"angular_traj_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}.png")
    fig_a.tight_layout(); fig_a.savefig(angular_path, dpi=130); plt.close(fig_a)

    return {"radial_plot": radial_path, "angular_plot": angular_path,
            "ring_radius": r_ring, "realized_times": times}


# ============================================================================
# Paper figures: heatmap comparison + mass-vs-time + driver
# ============================================================================
def _discrete_dl_mass(HM_DL, center, rg_param, ry_param, domain_radius=1.0):
    """Replicate numerical_tools.calc_mass_diff: DL + central-patch mass."""
    dRad = domain_radius / rg_param
    dThe = 2.0 * np.pi / ry_param
    m_weight = (np.arange(rg_param) + 1)[:, None]          # (m+1) ring weight
    dl = np.sum(HM_DL * m_weight) * dRad * dRad * dThe
    return dl + center * np.pi * dRad * dRad


def _discrete_al_mass(RvR, N, rg_param, domain_radius=1.0):
    """Replicate calc_mass_adv for N identical MTs from one ray's rho(r)."""
    dRad = domain_radius / rg_param
    return N * np.sum(RvR) * dRad


def _analytic_uncoupled_mass(t, rg_param, ry_param, coeffs, kappa, domain_radius=1.0):
    """Analytic uncoupled DL mass, using the SAME discrete quadrature as calc_mass."""
    r_centers, _, _ = A.build_eval_grid(rg_param, ry_param, domain_radius)
    prof = A.phi_uncoupled(r_centers, t, coeffs, kappa, domain_radius)   # (rg,)
    center = A.phi_uncoupled(np.array([0.0]), t, coeffs, kappa, domain_radius)[0]
    HM = np.repeat(prof[:, None], ry_param, axis=1)
    return _discrete_dl_mass(HM, center, rg_param, ry_param, domain_radius)


def _render_polar(ax, HM_DL, center, rg_param, ry_param, cmap, norm, title):
    """Polar pcolormesh matching data_visualization.produce_heatmap_tool_rect."""
    full = np.vstack([np.full((1, ry_param), center), HM_DL])    # (rg+1, ry)
    nr = full.shape[0]
    r = np.linspace(0, 1, nr + 1)
    theta = np.linspace(0, 2 * np.pi, ry_param + 1)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)
    pcm = ax.pcolormesh(X, Y, full.T, shading="flat", cmap=cmap, norm=norm)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)
    return pcm


def plot_heatmap_comparison(rg_param=32, ry_param=32, N_LIST=(0, 8, 16, 24),
                            v_param=1.0, w_param=100.0, t_stamp=0.1,
                            domain_radius=1.0, D=1.0, n_modes=80,
                            outdir=None, cmap="viridis"):
    """Numerical vs analytic phi(r,theta) heatmaps at one timestamp (+ fluctuation row).

    Top row: full phi field, numerical vs analytic, shared colour scale.
    Bottom row: angular fluctuation phi - <phi>_theta (highlights the N-fold MT
    structure), shared colour scale.  Analytic field = (scaled J_0 mean) +
    (LSQ-fit J_N) cos(N theta); amplitudes calibrated to the numerical field
    (the absolute modulation amplitude needs the coupled secular solve).
    """
    outdir = outdir or os.path.join(os.path.dirname(__file__), "..", "..",
                                    "summaries_reports", "figures")
    os.makedirs(outdir, exist_ok=True)
    N = len(N_LIST)
    r_centers, _, theta = A.build_eval_grid(rg_param, ry_param, domain_radius)
    coeffs, kappa = A.fourier_bessel_coeffs_uncoupled(rg_param, n_modes,
                                                      domain_radius, ic="patch")

    num = run_numerical_phi(rg_param, ry_param, list(N_LIST), v_param, w_param,
                            [t_stamp], domain_radius=domain_radius, D=D)
    HM_num = num["HM_DL"][0]; C_num = num["HM_C"][0]; t_real = num["realized_times"][0]

    # numerical channels
    g_num = HM_num.mean(axis=1)
    a_num = (2.0 / ry_param) * (HM_num @ np.cos(N * theta))
    # analytic channels, amplitude-calibrated to numerical
    g_ana_shape = A.phi_uncoupled(r_centers, t_real, coeffs, kappa, domain_radius)
    sg = _lsq_scale(g_num, g_ana_shape)
    jN = A.bessel_mode_shape(N, 1, r_centers, domain_radius)
    sa = _lsq_scale(a_num, jN)
    g_ana = sg * g_ana_shape
    a_ana = sa * jN
    center_ana = sg * A.phi_uncoupled(np.array([0.0]), t_real, coeffs, kappa, domain_radius)[0]
    HM_ana = g_ana[:, None] + np.outer(a_ana, np.cos(N * theta))

    # fluctuations
    fl_num = HM_num - g_num[:, None]
    fl_ana = HM_ana - g_ana[:, None]

    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(2, 2, figsize=(11, 12))
    vmax = max(HM_num.max(), HM_ana.max()); vmin = min(HM_num.min(), HM_ana.min())
    norm = Normalize(vmin=vmin, vmax=vmax)
    fmax = max(np.abs(fl_num).max(), np.abs(fl_ana).max())
    fnorm = Normalize(vmin=-fmax, vmax=fmax)

    p0 = _render_polar(ax[0, 0], HM_num, C_num, rg_param, ry_param, cmap, norm,
                       "Numerical  $\\phi(r,\\theta)$")
    p1 = _render_polar(ax[0, 1], HM_ana, center_ana, rg_param, ry_param, cmap, norm,
                       "Analytic  $g(r)+a(r)\\cos(N\\theta)$")
    fig.colorbar(p1, ax=ax[0, :].tolist(), location="bottom", pad=0.04, shrink=0.7)
    p2 = _render_polar(ax[1, 0], fl_num, 0.0, rg_param, ry_param, "coolwarm", fnorm,
                       "Numerical fluctuation  $\\phi-\\langle\\phi\\rangle_\\theta$")
    p3 = _render_polar(ax[1, 1], fl_ana, 0.0, rg_param, ry_param, "coolwarm", fnorm,
                       "Analytic fluctuation  $a(r)\\cos(N\\theta)$")
    fig.colorbar(p3, ax=ax[1, :].tolist(), location="bottom", pad=0.04, shrink=0.7)
    fig.suptitle(f"DL heatmap: numerical vs analytic at t={t_real:.3f}\n"
                 f"{rg_param}x{ry_param}, N={N}, v={v_param}, a=b=w={w_param}, "
                 f"R={domain_radius}, D={D}", fontsize=13)
    path = os.path.join(outdir, f"heatmap_compare_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}_t{t_real:.3f}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_mass_comparison(rg_param=32, ry_param=32, N_LIST=(0, 8, 16, 24),
                         v_coupled=1.0, w_coupled=100.0, t_max=0.35, n_t=40,
                         domain_radius=1.0, D=1.0, n_modes=120, outdir=None):
    """Mass(t): (A) uncoupled analytic-vs-numerical validation; (B) coupled run.

    Panel A (w=0): analytic uncoupled DL mass vs numerical mass -- the clean decay
    validation (should overlap).  Panel B (coupled): numerical total mass (DL+AL)
    vs the analytic uncoupled-DL mass baseline -- the gap is the coupling-induced
    loss (mass shuttled to the AL and advected/absorbed) that the leading
    uncoupled analytic does not capture.
    """
    outdir = outdir or os.path.join(os.path.dirname(__file__), "..", "..",
                                    "summaries_reports", "figures")
    os.makedirs(outdir, exist_ok=True)
    N = len(N_LIST)
    coeffs, kappa = A.fourier_bessel_coeffs_uncoupled(rg_param, n_modes,
                                                      domain_radius, ic="patch")
    chk = list(np.linspace(t_max / n_t, t_max, n_t))

    # ---- Panel A: uncoupled ----
    un = run_numerical_phi(rg_param, ry_param, [0], v_param=0.0, w_param=0.0,
                           checkpoints=chk, domain_radius=domain_radius, D=D)
    tA = un["realized_times"]
    m_num_A = np.array([_discrete_dl_mass(un["HM_DL"][i], un["HM_C"][i],
                                          rg_param, ry_param, domain_radius)
                        for i in range(len(chk))])
    m_ana_A = np.array([_analytic_uncoupled_mass(t, rg_param, ry_param, coeffs,
                                                 kappa, domain_radius) for t in tA])

    # ---- Panel B: coupled ----
    cp = run_numerical_phi(rg_param, ry_param, list(N_LIST), v_coupled, w_coupled,
                           checkpoints=chk, domain_radius=domain_radius, D=D)
    cr = run_numerical_radial(rg_param, ry_param, list(N_LIST), v_coupled, w_coupled,
                              checkpoints=chk, R_fixed_angle=N_LIST[0],
                              domain_radius=domain_radius, D=D)
    tB = cp["realized_times"]
    m_dl_B = np.array([_discrete_dl_mass(cp["HM_DL"][i], cp["HM_C"][i],
                                         rg_param, ry_param, domain_radius)
                       for i in range(len(chk))])
    m_al_B = np.array([_discrete_al_mass(cr["RvR"][i], N, rg_param, domain_radius)
                       for i in range(len(chk))])
    m_tot_B = m_dl_B + m_al_B
    m_ana_B = np.array([_analytic_uncoupled_mass(t, rg_param, ry_param, coeffs,
                                                 kappa, domain_radius) for t in tB])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(tA, m_num_A, "o", ms=4, label="numerical mass")
    ax[0].plot(tA, m_ana_A, "-", lw=2, label="analytic (uncoupled $J_0$ series)")
    ax[0].set_title(f"(A) Uncoupled validation (w=0): rel.err "
                    f"{np.max(np.abs(m_num_A-m_ana_A)/m_ana_A):.1e}")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("mass M(t)"); ax[0].legend(fontsize=9)
    ax[1].plot(tB, m_tot_B, "o", ms=4, label="numerical total (DL+AL)")
    ax[1].plot(tB, m_dl_B, "s", ms=3, alpha=0.5, label="numerical DL only")
    ax[1].plot(tB, m_ana_B, "-", lw=2, label="analytic uncoupled-DL baseline")
    ax[1].set_title(f"(B) Coupled v={v_coupled}, a=b=w={w_coupled}: gap = "
                    f"coupling loss")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("mass M(t)"); ax[1].legend(fontsize=9)
    fig.suptitle(f"Mass vs time  {rg_param}x{ry_param}, N={N}, R={domain_radius}, D={D}",
                 fontsize=13)
    path = os.path.join(outdir, f"mass_vs_time_{rg_param}x{ry_param}_N{N}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_first_principles_overlay(rg_param=32, ry_param=32, N_LIST=(0, 8, 16, 24),
                                  v_param=1.0, w_param=100.0,
                                  checkpoints=(0.05, 0.1, 0.2, 0.3),
                                  n_max=3, J_r=20, J_rho=20, coupling="scheme",
                                  domain_radius=1.0, D=1.0, outdir=None):
    """Overlay numerical vs FIRST-PRINCIPLES analytic (no least-squares scaling).

    The analytic decay rates s_k and amplitudes c_k come from
    ``solve_coupled_spectrum`` (eqs 24/25) -- not fitted.  ``coupling='scheme'``
    encodes the discrete solver's a->a*r*dThe coupling for a like-for-like
    comparison; ``'continuum'`` uses the equations exactly as written.
    """
    outdir = outdir or os.path.join(os.path.dirname(__file__), "..", "..",
                                    "summaries_reports", "figures")
    os.makedirs(outdir, exist_ok=True)
    N = len(N_LIST)
    chk = sorted(float(x) for x in checkpoints)
    r_centers, _, theta = A.build_eval_grid(rg_param, ry_param, domain_radius)
    dThe = 2.0 * np.pi / ry_param

    spec = A.solve_coupled_spectrum(a=w_param, b=w_param, v=v_param, N=N,
                                    n_max=n_max, J_r=J_r, J_rho=J_rho,
                                    domain_radius=domain_radius, ic_dRad=1.0 / rg_param,
                                    coupling=coupling, dThe=dThe)
    sigma_pred = np.real(A.dominant_decay_rate(spec))

    rad = run_numerical_radial(rg_param, ry_param, list(N_LIST), v_param, w_param,
                               chk, R_fixed_angle=N_LIST[0], domain_radius=domain_radius, D=D)
    times = rad["realized_times"]; PvR = rad["PvR"]; RvR = rad["RvR"]
    # numerical dominant sigma (late-window log-L2 slope of DL ray profile)
    dl_norms = np.array([np.linalg.norm(PvR[i, 1:]) for i in range(len(chk))])
    win = slice(max(0, len(chk) - 3), len(chk))
    sigma_num = -np.polyfit(times[win], np.log(dl_norms[win]), 1)[0]

    cmap = plt.get_cmap("viridis"); ncol = max(1, len(chk) - 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, t in enumerate(times):
        col = cmap(i / ncol)
        phi_ana = A.coupled_phi_field(spec, r_centers, np.array([0.0]), t)[:, 0]
        rho_ana = A.coupled_rho_field(spec, r_centers, t)
        ax[0].plot(r_centers, PvR[i, 1:], "o", ms=3, color=col, alpha=0.7)
        ax[0].plot(r_centers, phi_ana, "-", color=col, label=f"t={t:.3f}")
        ax[1].plot(r_centers, RvR[i], "o", ms=3, color=col, alpha=0.7)
        ax[1].plot(r_centers, rho_ana, "-", color=col, label=f"t={t:.3f}")
    ax[0].set_xlabel("r"); ax[0].set_ylabel(r"$\phi(r,\theta_{MT})$")
    ax[0].set_title("DL: numerical (o) vs first-principles (-)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("r"); ax[1].set_ylabel(r"$\rho(r)$")
    ax[1].set_title("AL: numerical (o) vs first-principles (-)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"First-principles overlay (NO fitting): $c_k$, $s_k$ from eqs (24)/(25), "
                 f"coupling='{coupling}'\n{rg_param}x{ry_param}, N={N}, v={v_param}, "
                 f"a=b=w={w_param}  |  $\\sigma$ predicted={sigma_pred:.3f}, "
                 f"numerical={sigma_num:.3f}", fontsize=12)
    path = os.path.join(outdir, f"first_principles_{coupling}_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"sigma: predicted={sigma_pred:.4f}  numerical={sigma_num:.4f}  -> {path}")
    return {"path": path, "sigma_pred": sigma_pred, "sigma_num": sigma_num, "spec": spec}


def first_principles_suite(rg_param=32, ry_param=32, N_LIST=(0, 8, 16, 24),
                           v_param=1.0, w_param=100.0,
                           checkpoints=(0.05, 0.1, 0.2, 0.3), t_heatmap=0.1,
                           n_mass=25, t_mass_max=0.35, n_max=3, J_r=20, J_rho=20,
                           T_fixed_ring_frac=0.5, coupling="scheme",
                           domain_radius=1.0, D=1.0, outdir=None):
    """All coupled verifications from the COMPUTED spectrum (s_k, c_k); NO fitting.

    Produces radial, angular, heatmap, and mass figures in which every analytic
    curve is reconstructed from ``solve_coupled_spectrum`` -- no least-squares
    scaling anywhere.  Returns a dict of figure paths plus sigma_pred/sigma_num.
    """
    outdir = outdir or os.path.join(os.path.dirname(__file__), "..", "..",
                                    "summaries_reports", "figures")
    os.makedirs(outdir, exist_ok=True)
    N = len(N_LIST)
    r_centers, _, theta = A.build_eval_grid(rg_param, ry_param, domain_radius)
    dThe = 2.0 * np.pi / ry_param
    from matplotlib.colors import Normalize

    spec = A.solve_coupled_spectrum(a=w_param, b=w_param, v=v_param, N=N,
                                    n_max=n_max, J_r=J_r, J_rho=J_rho,
                                    domain_radius=domain_radius, ic_dRad=1.0 / rg_param,
                                    coupling=coupling, dThe=dThe)
    sigma_pred = np.real(A.dominant_decay_rate(spec))

    # ---- numerical runs (shared) ----
    chk = sorted(float(x) for x in checkpoints)
    mass_chk = list(np.linspace(t_mass_max / n_mass, t_mass_max, n_mass))
    # union of times we need full fields / radial at
    fld = run_numerical_phi(rg_param, ry_param, list(N_LIST), v_param, w_param,
                            sorted(set(chk + mass_chk + [t_heatmap])),
                            domain_radius=domain_radius, D=D)
    rad = run_numerical_radial(rg_param, ry_param, list(N_LIST), v_param, w_param,
                               sorted(set(chk + mass_chk)), R_fixed_angle=N_LIST[0],
                               domain_radius=domain_radius, D=D)
    ang = run_numerical_angular(rg_param, ry_param, list(N_LIST), v_param, w_param,
                                chk, T_fixed_ring_frac=T_fixed_ring_frac,
                                domain_radius=domain_radius, D=D)
    fld_t = list(fld["realized_times"]); rad_t = list(rad["realized_times"])
    cmap = plt.get_cmap("viridis"); ncol = max(1, len(chk) - 1)
    paths = {}

    def _idx(times, t):
        return int(np.argmin(np.abs(np.asarray(times) - t)))

    # numerical dominant sigma from DL field L2 (late window)
    norms = np.array([field_l2_norm(fld["HM_DL"][_idx(fld_t, t)], rg_param, ry_param,
                                    domain_radius) for t in mass_chk])
    sigma_num = -np.polyfit(mass_chk[-4:], np.log(norms[-4:]), 1)[0]

    # ============ RADIAL ============
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, t in enumerate(chk):
        col = cmap(i / ncol); j = _idx(rad_t, t)
        phi_a = A.coupled_phi_field(spec, r_centers, np.array([0.0]), rad_t[j])[:, 0]
        rho_a = A.coupled_rho_field(spec, r_centers, rad_t[j])
        ax[0].plot(r_centers, rad["PvR"][j, 1:], "o", ms=3, color=col, alpha=0.7)
        ax[0].plot(r_centers, phi_a, "-", color=col, label=f"t={rad_t[j]:.3f}")
        ax[1].plot(r_centers, rad["RvR"][j], "o", ms=3, color=col, alpha=0.7)
        ax[1].plot(r_centers, rho_a, "-", color=col, label=f"t={rad_t[j]:.3f}")
    ax[0].set_xlabel("r"); ax[0].set_ylabel(r"$\phi(r,\theta_{MT})$")
    ax[0].set_title("DL radial: numerical (o) vs first-principles (-)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("r"); ax[1].set_ylabel(r"$\rho(r)$")
    ax[1].set_title("AL radial: numerical (o) vs first-principles (-)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"First-principles radial (no fitting), {rg_param}x{ry_param}, N={N}, "
                 f"v={v_param}, a=b=w={w_param}")
    paths["radial"] = os.path.join(outdir, f"fp_radial_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}.png")
    fig.tight_layout(); fig.savefig(paths["radial"], dpi=130); plt.close(fig)

    # ============ ANGULAR ============
    ring_m = ang["ring_index"]; r_ring = (ring_m + 1) * (domain_radius / rg_param)
    fig, axa = plt.subplots(figsize=(8, 5)); deg = np.degrees(theta)
    for i, t in enumerate(chk):
        col = cmap(i / ncol)
        prof_n = ang["PvT"][i]; fl_n = prof_n - prof_n.mean()
        phi_a = A.coupled_phi_field(spec, np.array([r_ring]), theta, ang["realized_times"][i])[0]
        fl_a = phi_a - phi_a.mean()
        axa.plot(deg, fl_n, "o", ms=3, color=col, alpha=0.7)
        axa.plot(deg, fl_a, "-", color=col, label=f"t={ang['realized_times'][i]:.3f}")
    for nl in N_LIST:
        axa.axvline(np.degrees(theta[nl]), color="grey", ls=":", lw=0.8, alpha=0.6)
    axa.set_xlabel(r"$\theta$ (degrees)")
    axa.set_ylabel(r"$\phi(\theta)-\langle\phi\rangle_\theta$")
    axa.set_title(f"First-principles angular (no fitting) at r={r_ring:.3f}\n"
                  f"numerical (o) vs spectrum reconstruction (-), harmonics up to {n_max}N")
    axa.legend(fontsize=8)
    paths["angular"] = os.path.join(outdir, f"fp_angular_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}.png")
    fig.tight_layout(); fig.savefig(paths["angular"], dpi=130); plt.close(fig)

    # ============ HEATMAP ============
    ih = _idx(fld_t, t_heatmap); th = fld_t[ih]
    HM_n = fld["HM_DL"][ih]; C_n = fld["HM_C"][ih]
    HM_a = A.coupled_phi_field(spec, r_centers, theta, th)
    C_a = A.coupled_phi_field(spec, np.array([0.0]), theta, th)[0, 0]
    fln = HM_n - HM_n.mean(axis=1, keepdims=True)
    fla = HM_a - HM_a.mean(axis=1, keepdims=True)
    fig, axh = plt.subplots(2, 2, figsize=(11, 12))
    norm = Normalize(vmin=min(HM_n.min(), HM_a.min()), vmax=max(HM_n.max(), HM_a.max()))
    fmax = max(np.abs(fln).max(), np.abs(fla).max()); fnorm = Normalize(-fmax, fmax)
    _render_polar(axh[0, 0], HM_n, C_n, rg_param, ry_param, "viridis", norm, "Numerical $\\phi$")
    p1 = _render_polar(axh[0, 1], HM_a, C_a, rg_param, ry_param, "viridis", norm, "First-principles $\\phi$")
    fig.colorbar(p1, ax=axh[0, :].tolist(), location="bottom", pad=0.04, shrink=0.7)
    _render_polar(axh[1, 0], fln, 0.0, rg_param, ry_param, "coolwarm", fnorm, "Numerical fluctuation")
    p3 = _render_polar(axh[1, 1], fla, 0.0, rg_param, ry_param, "coolwarm", fnorm, "First-principles fluctuation")
    fig.colorbar(p3, ax=axh[1, :].tolist(), location="bottom", pad=0.04, shrink=0.7)
    fig.suptitle(f"First-principles DL heatmap (no fitting) at t={th:.3f}\n"
                 f"{rg_param}x{ry_param}, N={N}, v={v_param}, a=b=w={w_param}", fontsize=13)
    paths["heatmap"] = os.path.join(outdir, f"fp_heatmap_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}_t{th:.3f}.png")
    fig.savefig(paths["heatmap"], dpi=130, bbox_inches="tight"); plt.close(fig)

    # ============ MASS ============
    m_num, m_ana = [], []
    for t in mass_chk:
        jf = _idx(fld_t, t); jr = _idx(rad_t, t)
        m_num.append(_discrete_dl_mass(fld["HM_DL"][jf], fld["HM_C"][jf], rg_param, ry_param, domain_radius)
                     + _discrete_al_mass(rad["RvR"][jr], N, rg_param, domain_radius))
        HM_a = A.coupled_phi_field(spec, r_centers, theta, t)
        C_a = A.coupled_phi_field(spec, np.array([0.0]), theta, t)[0, 0]
        rho_a = A.coupled_rho_field(spec, r_centers, t)
        m_ana.append(_discrete_dl_mass(HM_a, C_a, rg_param, ry_param, domain_radius)
                     + _discrete_al_mass(rho_a, N, rg_param, domain_radius))
    m_num = np.array(m_num); m_ana = np.array(m_ana)
    fig, axm = plt.subplots(figsize=(8, 5))
    axm.plot(mass_chk, m_num, "o", ms=4, label="numerical total mass (DL+AL)")
    axm.plot(mass_chk, m_ana, "-", lw=2, label="first-principles total mass")
    axm.set_xlabel("t"); axm.set_ylabel("mass M(t)")
    axm.set_title(f"First-principles total mass (no fitting)\n{rg_param}x{ry_param}, N={N}, "
                  f"v={v_param}, a=b=w={w_param}")
    axm.legend(fontsize=9)
    paths["mass"] = os.path.join(outdir, f"fp_mass_{rg_param}x{ry_param}_N{N}_v{v_param}_w{w_param}.png")
    fig.tight_layout(); fig.savefig(paths["mass"], dpi=130); plt.close(fig)

    print(f"sigma predicted={sigma_pred:.4f}  numerical={sigma_num:.4f}")
    for k, p in paths.items():
        print(f"  {k:8s}: {p}")
    return {"paths": paths, "sigma_pred": sigma_pred, "sigma_num": sigma_num}


def generate_paper_figures(rg_param=32, ry_param=32, N_LIST=(0, 8, 16, 24),
                           v_param=1.0, w_param=100.0,
                           checkpoints=(0.05, 0.1, 0.2, 0.3), t_heatmap=0.1,
                           domain_radius=1.0, D=1.0):
    """Produce all four validation figures for the paper, return their paths."""
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                           "summaries_reports", "figures")
    traj = plot_verification_trajectories(
        rg_param, ry_param, list(N_LIST), v_param, w_param,
        checkpoints=checkpoints, domain_radius=domain_radius, D=D, outdir=fig_dir)
    heat = plot_heatmap_comparison(
        rg_param, ry_param, N_LIST, v_param, w_param, t_stamp=t_heatmap,
        domain_radius=domain_radius, D=D)
    mass = plot_mass_comparison(
        rg_param, ry_param, N_LIST, v_coupled=v_param, w_coupled=w_param,
        domain_radius=domain_radius, D=D)
    paths = {"radial": traj["radial_plot"], "angular": traj["angular_plot"],
             "heatmap": heat, "mass": mass}
    for k, p in paths.items():
        print(f"{k:8s}: {p}")
    return paths


# ============================================================================
# Deterministic verification driver (general parameters, all first-principles)
# ============================================================================
def _mse(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    return float(np.mean((u - v) ** 2))


def _even_spacing_ok(microtubule_list, rays):
    n = len(microtubule_list)
    if n == 0 or rays % n != 0:
        return False
    step = rays // n
    expected = sorted((microtubule_list[0] + k * step) % rays for k in range(n))
    return sorted(int(x) for x in microtubule_list) == expected


def deterministic_verification(
        rings, rays, microtubule_list, a, b, v, timestamps,
        T=None, domain_radius=1.0, D=1.0,
        # analytic-scheme hyperparameters
        coupling="scheme", n_max=3, J_r=20, J_rho=20, n_quad=400,
        # numerical-scheme hyperparameters
        d_tube=0.0,
        # plotting hyperparameters
        fixed_angle_index=None, ring_frac=0.5, heatmap_time=None,
        n_mass=40, show_mse=False, cmap="viridis", dpi=130,
        show=False, save=True, out_root=None, label=None, sigma_window=0.4):
    """Deterministic numerical-vs-analytic (first-principles) verification suite.

    Parameters
    ----------
    rings (M), rays (N) : polar grid resolution (rg_param, ry_param).
    microtubule_list    : angular indices of microtubules in [0, rays-1]
                          (0-indexed like the numerical scheme; must be evenly
                          spaced for the analytic comb).
    a, b                : DL<->AL on-/off-rates (switch_param_a, switch_param_b).
    v                   : filament (advective) velocity.
    timestamps          : list of collection times; the final entry should equal T.
    T                   : total run time (defaults to max(timestamps)).
    domain_radius (r), D: disk radius and diffusion coefficient (defaults 1, 1).

    Analytic hyperparameters: ``coupling`` ('scheme' matches this solver's
    a->a*r*dThe coupling; 'continuum' uses the equations as written), ``n_max``
    angular orders, ``J_r``/``J_rho`` radial modes, ``n_quad`` quadrature nodes.
    Numerical: ``d_tube`` microtubule extraction width.
    Plotting: ``fixed_angle_index`` (radial ray; default first MT), ``ring_frac``
    (angular ring; default mid-disk), ``heatmap_time`` (default last timestamp),
    ``n_mass`` mass-curve samples, ``show_mse`` (shade error bands + add an
    MSE-vs-time panel), ``cmap``, ``dpi``, ``show``, ``save``.
    Output: ``out_root`` (default data_output/analytic_verification), ``label``
    (subfolder), ``sigma_window`` (late-time fraction used for the decay fit).

    Produces (numerical vs first-principles analytic, NO least-squares):
    radial DL+AL, angular DL, DL+AL heatmaps, mass(t), and -- if show_mse --
    an MSE-vs-time summary.  Returns a dict of paths and scalar diagnostics.
    """
    from matplotlib.colors import Normalize

    rings = int(rings); rays = int(rays)
    microtubule_list = sorted(int(x) for x in microtubule_list)
    num_mt = len(microtubule_list)
    ts = sorted(float(x) for x in timestamps)
    if T is None:
        T = ts[-1]
    if abs(ts[-1] - T) > 1e-12:
        print(f"[warn] final timestamp {ts[-1]} != T={T}; using T={T} as horizon.")
    if not _even_spacing_ok(microtubule_list, rays):
        print(f"[warn] microtubule_list {microtubule_list} is not evenly spaced "
              f"on {rays} rays; the analytic comb assumes even spacing.")
    if fixed_angle_index is None:
        fixed_angle_index = microtubule_list[0]
    if heatmap_time is None:
        heatmap_time = ts[-1]

    dThe = 2.0 * np.pi / rays
    R = domain_radius
    r_centers, _, theta = A.build_eval_grid(rings, rays, R)

    # ---- output directory ----
    if out_root is None:
        out_root = os.path.join(os.path.dirname(__file__), "..", "data_output",
                                "analytic_verification")
    if label is None:
        label = f"M{rings}_N{rays}_nmt{num_mt}_a{a}_b{b}_v{v}_T{T}"
    outdir = os.path.join(out_root, label)
    os.makedirs(outdir, exist_ok=True)

    # ---- analytic spectrum (first principles) ----
    spec = A.solve_coupled_spectrum(a=a, b=b, v=v, N=num_mt, n_max=n_max, J_r=J_r,
                                    J_rho=J_rho, domain_radius=R, n_quad=n_quad,
                                    ic_dRad=R / rings, coupling=coupling, dThe=dThe)
    # slowest *excited* eigenvalue (diagnostic only -- may be an AL-relaxation
    # mode with little DL content, so it need not match the DL field's decay)
    sigma_slowest = np.real(A.dominant_decay_rate(spec))

    # ---- numerical runs (shared over the needed times) ----
    # Dedup by integer time-stamp floor(t/dT): two near-equal floats mapping to
    # the same stamp would stall the snapshot collector (it advances its iterator
    # on the first and never matches the duplicate, zeroing all later snapshots).
    _dT = num.compute_dT(rings, rays, R, D)

    def _unique_by_stamp(times):
        seen, out = set(), []
        for t in sorted(times):
            s = int(np.floor(t / _dT))
            if s not in seen:
                seen.add(s); out.append(float(t))
        return out

    ts = _unique_by_stamp(ts)
    mass_ts = _unique_by_stamp(np.linspace(T / n_mass, T, n_mass))
    field_ts = _unique_by_stamp(ts + mass_ts + [heatmap_time])
    rad_ts = field_ts
    fld = run_numerical_phi(rings, rays, microtubule_list, v, a, field_ts,
                            domain_radius=R, D=D, d_tube=d_tube, b_param=b)
    rad = run_numerical_radial(rings, rays, microtubule_list, v, a, rad_ts,
                               R_fixed_angle=fixed_angle_index, domain_radius=R,
                               D=D, d_tube=d_tube, b_param=b)
    ang = run_numerical_angular(rings, rays, microtubule_list, v, a, ts,
                                T_fixed_ring_frac=ring_frac, domain_radius=R,
                                D=D, d_tube=d_tube, b_param=b)
    fld_t = list(fld["realized_times"]); rad_t = list(rad["realized_times"])
    _idx = lambda times, t: int(np.argmin(np.abs(np.asarray(times) - t)))

    # Dominant decay rate measured IDENTICALLY on both sides: the slope of the
    # DL-field L2 norm over the late time window.  (Measuring sigma the same way
    # for numerical and analytic is the only fair comparison; the slowest
    # eigenvalue above can differ if its mode carries little DL weight.)
    mass_arr = np.asarray(mass_ts)
    nrm = np.array([field_l2_norm(fld["HM_DL"][_idx(fld_t, t)], rings, rays, R)
                    for t in mass_ts])
    ana_nrm = np.array([field_l2_norm(A.coupled_phi_field(spec, r_centers, theta, t),
                                      rings, rays, R) for t in mass_ts])
    nwin = max(3, int(sigma_window * len(mass_ts)))

    def _slope_sigma(y):
        yy = y[-nwin:]; tt = mass_arr[-nwin:]; ok = yy > 0
        return (-np.polyfit(tt[ok], np.log(yy[ok]), 1)[0]
                if ok.sum() >= 2 else float("nan"))

    sigma_num = _slope_sigma(nrm)
    sigma_pred = _slope_sigma(ana_nrm)

    cmap_o = plt.get_cmap(cmap); ncol = max(1, len(ts) - 1)
    paths = {}; mse = {}

    def _finish(fig, name):
        p = os.path.join(outdir, name)
        if save:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        paths[name.split(".")[0]] = p

    # ===== RADIAL (DL + AL) =====
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    e_dl, e_al = [], []
    for i, t in enumerate(ts):
        col = cmap_o(i / ncol); j = _idx(rad_t, t)
        phi_a = A.coupled_phi_field(spec, r_centers, np.array([0.0]), rad_t[j])[:, 0]
        rho_a = A.coupled_rho_field(spec, r_centers, rad_t[j])
        phi_n = rad["PvR"][j, 1:]; rho_n = rad["RvR"][j]
        e_dl.append(_mse(phi_n, phi_a)); e_al.append(_mse(rho_n, rho_a))
        ax[0].plot(r_centers, phi_n, "o", ms=3, color=col, alpha=0.7)
        ax[0].plot(r_centers, phi_a, "-", color=col, label=f"t={rad_t[j]:.3f}")
        ax[1].plot(r_centers, rho_n, "o", ms=3, color=col, alpha=0.7)
        ax[1].plot(r_centers, rho_a, "-", color=col, label=f"t={rad_t[j]:.3f}")
        if show_mse:
            ax[0].fill_between(r_centers, phi_n, phi_a, color=col, alpha=0.15)
            ax[1].fill_between(r_centers, rho_n, rho_a, color=col, alpha=0.15)
    ax[0].set_xlabel("r"); ax[0].set_ylabel(r"$\phi(r,\theta_{MT})$")
    ax[0].set_title("DL radial: numerical (o) vs first-principles (-)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("r"); ax[1].set_ylabel(r"$\rho(r)$")
    ax[1].set_title("AL radial: numerical (o) vs first-principles (-)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Radial trajectories (no fitting) | M={rings}, N={rays}, "
                 f"#MT={num_mt}, a={a}, b={b}, v={v}, R={R}, D={D}")
    fig.tight_layout(); _finish(fig, "radial.png")
    mse["dl_radial"] = e_dl; mse["al_radial"] = e_al

    # ===== ANGULAR (DL) =====
    ring_m = ang["ring_index"]; r_ring = (ring_m + 1) * (R / rings)
    fig, axa = plt.subplots(figsize=(8.5, 5)); deg = np.degrees(theta); e_ang = []
    for i, t in enumerate(ts):
        col = cmap_o(i / ncol)
        prof_n = ang["PvT"][i]; fl_n = prof_n - prof_n.mean()
        phi_a = A.coupled_phi_field(spec, np.array([r_ring]), theta, ang["realized_times"][i])[0]
        fl_a = phi_a - phi_a.mean()
        e_ang.append(_mse(fl_n, fl_a))
        axa.plot(deg, fl_n, "o", ms=3, color=col, alpha=0.7)
        axa.plot(deg, fl_a, "-", color=col, label=f"t={ang['realized_times'][i]:.3f}")
        if show_mse:
            axa.fill_between(deg, fl_n, fl_a, color=col, alpha=0.15)
    for nl in microtubule_list:
        axa.axvline(np.degrees(theta[nl]), color="grey", ls=":", lw=0.8, alpha=0.6)
    axa.set_xlabel(r"$\theta$ (degrees)")
    axa.set_ylabel(r"$\phi(\theta)-\langle\phi\rangle_\theta$")
    axa.set_title(f"DL angular fluctuation at r={r_ring:.3f} (no fitting), "
                  f"harmonics up to {n_max}x#MT")
    axa.legend(fontsize=8)
    fig.tight_layout(); _finish(fig, "angular.png")
    mse["angular"] = e_ang

    # ===== HEATMAPS (DL + AL) =====
    ih = _idx(fld_t, heatmap_time); th = fld_t[ih]; jr = _idx(rad_t, heatmap_time)
    HM_n = fld["HM_DL"][ih]; C_n = fld["HM_C"][ih]
    HM_a = A.coupled_phi_field(spec, r_centers, theta, th)
    C_a = A.coupled_phi_field(spec, np.array([0.0]), theta, th)[0, 0]
    # AL fields: rho lives on MT rays only (all identical) -> spokes
    AL_n = np.zeros((rings, rays)); AL_a = np.zeros((rings, rays))
    rho_a_h = A.coupled_rho_field(spec, r_centers, th)
    for p in microtubule_list:
        AL_n[:, p] = rad["RvR"][jr]; AL_a[:, p] = rho_a_h
    fig, axh = plt.subplots(2, 2, figsize=(11, 12))
    dnorm = Normalize(vmin=min(HM_n.min(), HM_a.min()), vmax=max(HM_n.max(), HM_a.max()))
    almax = max(AL_n.max(), AL_a.max()); anorm = Normalize(0, almax if almax > 0 else 1)
    _render_polar(axh[0, 0], HM_n, C_n, rings, rays, cmap, dnorm, "Numerical DL $\\phi$")
    pD = _render_polar(axh[0, 1], HM_a, C_a, rings, rays, cmap, dnorm, "First-principles DL $\\phi$")
    fig.colorbar(pD, ax=axh[0, :].tolist(), location="bottom", pad=0.04, shrink=0.7)
    _render_polar(axh[1, 0], AL_n, 0.0, rings, rays, cmap, anorm, "Numerical AL $\\rho$ (spokes)")
    pA = _render_polar(axh[1, 1], AL_a, 0.0, rings, rays, cmap, anorm, "First-principles AL $\\rho$")
    fig.colorbar(pA, ax=axh[1, :].tolist(), location="bottom", pad=0.04, shrink=0.7)
    fig.suptitle(f"Heatmaps at t={th:.3f} (no fitting) | M={rings}, N={rays}, "
                 f"#MT={num_mt}, a={a}, b={b}, v={v}", fontsize=13)
    _finish(fig, "heatmap.png")

    # ===== MASS(t) =====
    m_num, m_ana = [], []
    for t in mass_ts:
        jf = _idx(fld_t, t); jrr = _idx(rad_t, t)
        m_num.append(_discrete_dl_mass(fld["HM_DL"][jf], fld["HM_C"][jf], rings, rays, R)
                     + _discrete_al_mass(rad["RvR"][jrr], num_mt, rings, R))
        HMa = A.coupled_phi_field(spec, r_centers, theta, t)
        Ca = A.coupled_phi_field(spec, np.array([0.0]), theta, t)[0, 0]
        rhoa = A.coupled_rho_field(spec, r_centers, t)
        m_ana.append(_discrete_dl_mass(HMa, Ca, rings, rays, R)
                     + _discrete_al_mass(rhoa, num_mt, rings, R))
    m_num = np.array(m_num); m_ana = np.array(m_ana)
    mse["mass"] = _mse(m_num, m_ana)
    fig, axm = plt.subplots(figsize=(8.5, 5))
    axm.plot(mass_ts, m_num, "o", ms=4, label="numerical total (DL+AL)")
    axm.plot(mass_ts, m_ana, "-", lw=2, label="first-principles total")
    if show_mse:
        axm.fill_between(mass_ts, m_num, m_ana, color="grey", alpha=0.25, label="error")
    axm.set_xlabel("t"); axm.set_ylabel("mass M(t)"); axm.legend(fontsize=9)
    axm.set_title(f"Total mass (no fitting) | $\\sigma$ pred={sigma_pred:.3f}, "
                  f"num={sigma_num:.3f}")
    fig.tight_layout(); _finish(fig, "mass.png")

    # ===== MSE-vs-time summary (optional) =====
    if show_mse:
        fig, axe = plt.subplots(figsize=(8.5, 5))
        axe.plot(ts, mse["dl_radial"], "o-", label="DL radial MSE")
        axe.plot(ts, mse["al_radial"], "s-", label="AL radial MSE")
        axe.plot(ts, mse["angular"], "^-", label="DL angular MSE")
        axe.set_xlabel("t"); axe.set_ylabel("MSE (numerical vs analytic)")
        axe.set_yscale("log"); axe.legend(fontsize=9)
        axe.set_title(f"MSE vs time | mass MSE={mse['mass']:.2e}")
        fig.tight_layout(); _finish(fig, "mse_vs_time.png")

    print(f"[deterministic_verification] {label}")
    print(f"  sigma: predicted={sigma_pred:.4f}  numerical={sigma_num:.4f}")
    print(f"  MSE  DL_radial(last)={mse['dl_radial'][-1]:.3e}  "
          f"AL_radial(last)={mse['al_radial'][-1]:.3e}  mass={mse['mass']:.3e}")
    for k, p in paths.items():
        print(f"  {k:12s}: {p}")
    return {"paths": paths, "sigma_pred": sigma_pred, "sigma_num": sigma_num,
            "sigma_slowest": sigma_slowest, "mse": mse, "outdir": outdir}


def grid_refinement_sigma(grids, microtubule_list, a, b, v, T,
                          domain_radius=1.0, D=1.0, coupling="scheme",
                          n_max=3, J_r=20, J_rho=20, n_mass=30, sigma_window=0.4):
    """#3: numerical dominant sigma vs grid (M=N from ``grids``) vs the predicted
    spectral sigma.  Returns a list of (grid, sigma_pred, sigma_num)."""
    rows = []
    for g in grids:
        dThe = 2.0 * np.pi / g
        spec = A.solve_coupled_spectrum(a=a, b=b, v=v, N=len(microtubule_list),
                                        n_max=n_max, J_r=J_r, J_rho=J_rho,
                                        domain_radius=domain_radius, ic_dRad=domain_radius / g,
                                        coupling=coupling, dThe=dThe)
        sp = np.real(A.dominant_decay_rate(spec))
        # evenly spaced MT positions on g rays with the same count
        nmt = len(microtubule_list); step = g // nmt
        mt = [(k * step) % g for k in range(nmt)]
        mass_ts = list(np.linspace(T / n_mass, T, n_mass))
        fld = run_numerical_phi(g, g, mt, v, a, mass_ts, domain_radius=domain_radius,
                                D=D, b_param=b)
        ft = list(fld["realized_times"])
        nrm = np.array([field_l2_norm(fld["HM_DL"][i], g, g, domain_radius)
                        for i in range(len(mass_ts))])
        nwin = max(3, int(sigma_window * len(mass_ts)))
        sn = -np.polyfit(mass_ts[-nwin:], np.log(nrm[-nwin:]), 1)[0]
        rows.append((g, sp, sn))
        print(f"  grid {g}x{g}: sigma_pred={sp:.4f}  sigma_num={sn:.4f}")
    return rows
