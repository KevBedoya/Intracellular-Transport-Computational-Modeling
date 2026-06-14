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
