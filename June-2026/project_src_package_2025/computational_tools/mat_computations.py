import numpy as np
from scipy.linalg import svdvals

def normalize_columns(
    A,
    return_central_vector=False,
    central_patch=None
):
    """
    Normalize each column of A to unit Euclidean norm.

    Parameters
    ----------
    A : ndarray
        Input matrix.

    return_central_vector : bool, optional
        If True, also return normalized central patch values.

    central_patch : float, optional
        Central patch density value.

    Returns
    -------
    ndarray or None
        Normalized central patch values if requested.
    """

    central_vec = (
        np.empty(A.shape[1], dtype=np.float64)
        if return_central_vector else None
    )

    for idx in range(A.shape[1]):

        norm = np.linalg.norm(A[:, idx])

        A[:, idx] /= norm

        if return_central_vector:
            central_vec[idx] = central_patch / norm

    return central_vec

def rank1_energy_ratio(A):
    """
    Compute the fraction of matrix energy captured by the
    dominant singular value.

    Parameters
    ----------
    A : ndarray
        Input matrix.

    Returns
    -------
    float
        Rank-1 energy ratio in [0,1].
    """
    vals = svdvals(A)
    return (vals[0] ** 2) / np.dot(vals, vals)


def mean_alignment_error(C):
    """
    Compute the mean Frobenius deviation from perfect
    proportionality (|cos(theta)| = 1).

    Parameters
    ----------
    C : ndarray
        Cosine similarity matrix.

    Returns
    -------
    float
        Average proportionality error.
    """
    return np.linalg.norm(np.abs(C) - 1.0, 'fro') / np.sqrt(C.size)


def max_alignment_error(C):
    """
    Compute the maximum deviation from perfect
    proportionality (|cos(theta)| = 1).

    Parameters
    ----------
    C : ndarray
        Cosine similarity matrix.

    Returns
    -------
    float
        Worst-case proportionality error.
    """
    return np.max(np.abs(np.abs(C) - 1.0))


def affine_trajectory_overlap(M, reference=None, eps=1e-12):
    """
    Affine-overlap test for an additive term in the separability ansatz.

    Fit every column (trajectory) of ``M`` with an affine map onto a common
    reference shape,

        c_j  ~=  alpha_j * u  +  beta_j * 1,

    by least squares, WITHOUT forcing the trajectories to coincide. The fit is
    then inspected: a column-dependent offset ``beta_j`` that is genuinely
    *needed* -- i.e. that significantly lowers the residual relative to a
    scaling-only fit -- is the signature of an additive term in the field:

        phi(r, theta) = R(r) Theta(theta)            -> pure product, beta ~= 0
        phi(r, theta) = R(r) Theta(theta) + offset   -> additive, beta_j != 0

    For radial trajectories (pass ``M = D``) the recovered offsets estimate an
    additive term ``h(theta)`` that is constant in r; for angular trajectories
    (pass ``M = D.T``) they estimate ``g(r)``, constant in theta.

    Unlike the cosine-similarity pipeline, the columns are NOT normalized and
    ``M`` is not modified. Normalization is a pure scaling and would absorb the
    offset, which is exactly why the cosine diagnostics are blind to additivity.

    Decision guide
    --------------
    beta ~= 0, improvement ~= 0                  -> multiplicative (product)
    alpha ~= const, beta varies, improvement large -> additive
    both alpha and beta structured               -> product + additive

    Parameters
    ----------
    M : ndarray, shape (p, q)
        Trajectory matrix; each of the q columns is one trajectory sampled at
        p points.

    reference : ndarray, shape (p,), optional
        Common reference shape u. Defaults to the (sign-fixed) leading left
        singular vector of M.

    eps : float, optional
        Numerical guard for near-degenerate norms / determinant.

    Returns
    -------
    dict
        alpha : ndarray (q,)
            Per-trajectory scale coefficient.
        beta : ndarray (q,)
            Per-trajectory additive offset.
        offset_fraction : ndarray (q,)
            |beta_j| * sqrt(p) / ||c_j||, the offset expressed as a fraction of
            the trajectory's magnitude (scale-free; ~0 means no additive term).
        mult_residual : ndarray (q,)
            Relative residual of the scaling-only fit, S_mult / ||c_j||^2.
        affine_residual : ndarray (q,)
            Relative residual of the affine fit, S_aff / ||c_j||^2.
        improvement : ndarray (q,)
            (S_mult - S_aff) / S_mult in [0, 1]; fraction of the proportional
            residual explained by allowing an offset.
        reference : ndarray (p,)
            The reference shape u that was used.
        identifiable : bool
            False if u is ~collinear with the all-ones vector, in which case the
            offset cannot be resolved separately and a scaling-only fit is
            returned (beta = 0).
    """
    M = np.asarray(M, dtype=np.float64)
    p, q = M.shape

    if reference is None:
        # leading left singular vector serves as the common shape
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        u = U[:, 0].copy()
    else:
        u = np.asarray(reference, dtype=np.float64).reshape(-1)
        if u.shape[0] != p:
            raise ValueError(
                f"reference length {u.shape[0]} != trajectory length {p}"
            )

    # sign convention for reproducibility
    if u.sum() < 0:
        u = -u

    uu = float(u @ u)
    u1 = float(u.sum())
    det = uu * p - u1 * u1

    identifiable = det > eps * max(uu * p, 1.0)

    Uc = u @ M                              # (q,) <u, c_j>
    onec = M.sum(axis=0)                    # (q,) <1, c_j>
    norm_c2 = np.einsum('ij,ij->j', M, M)   # (q,) ||c_j||^2

    if identifiable:
        alpha = (p * Uc - u1 * onec) / det
        beta = (uu * onec - u1 * Uc) / det
    else:
        # u collinear with 1: offset not resolvable -> scaling-only fit
        alpha = Uc / max(uu, eps)
        beta = np.zeros(q, dtype=np.float64)

    recon = np.outer(u, alpha) + beta[np.newaxis, :]
    diff = M - recon
    S_aff = np.einsum('ij,ij->j', diff, diff)
    S_mult = norm_c2 - (Uc * Uc) / max(uu, eps)

    safe = norm_c2 > eps
    denom_c2 = np.where(safe, norm_c2, 1.0)
    mult_residual = np.where(safe, S_mult / denom_c2, 0.0)
    affine_residual = np.where(safe, S_aff / denom_c2, 0.0)

    safe_m = S_mult > eps
    denom_m = np.where(safe_m, S_mult, 1.0)
    improvement = np.clip(
        np.where(safe_m, (S_mult - S_aff) / denom_m, 0.0), 0.0, 1.0
    )

    offset_fraction = np.where(
        safe, np.abs(beta) * np.sqrt(p) / np.sqrt(denom_c2), 0.0
    )

    return {
        'alpha': alpha,
        'beta': beta,
        'offset_fraction': offset_fraction,
        'mult_residual': mult_residual,
        'affine_residual': affine_residual,
        'improvement': improvement,
        'reference': u,
        'identifiable': bool(identifiable),
    }


def pearson_correlation_matrix(M, eps=1e-12):
    """
    Pairwise Pearson correlation between the columns of M (centered cosine).

    This is the scale-free scalar summary of the affine fit. The raw cosine
    similarity tests proportionality through the origin (a purely multiplicative
    relationship); the Pearson correlation tests an affine relationship
    (scale + offset). The gap between the raw cosine matrix and this correlation
    matrix is therefore a direct additivity signal: if centering markedly raises
    the similarity, a constant additive offset is present.

    ``M`` is not modified.

    Parameters
    ----------
    M : ndarray, shape (p, q)
        Trajectory matrix; columns are trajectories.

    eps : float, optional
        Guard against zero-variance columns.

    Returns
    -------
    ndarray, shape (q, q)
        Correlation matrix with entries in [-1, 1].
    """
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Mc, axis=0)
    norms = np.where(norms > eps, norms, 1.0)
    Mn = Mc / norms
    return Mn.T @ Mn
