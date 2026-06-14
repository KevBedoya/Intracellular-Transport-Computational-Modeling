from . import np, njit
# from numba.typed import Dict, ListType, List
from numba.typed import Dict
# from numba.types import ListType
from numba import int64
import math


def initialize_layers(rg_param, ry_param):
    diffusive_layer = np.zeros((2, rg_param, ry_param), dtype=np.float64)
    advective_layer = np.zeros((2, rg_param, ry_param), dtype=np.float64)
    return diffusive_layer, advective_layer

@njit
def solve_d_rect(r, rings, rays, j_max, m):
    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    return (j_max + 0.5) * (m + 1) * d_radius * d_theta


def solve_d_rect_no_JIT(r, rings, rays, j_max, m):
    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    return (j_max + 0.5) * (m + 1) * d_radius * d_theta


def j_max_domain_list(ry_param, rg_param, N_param, r=1, overlap=False, d_tube=-1, ceil=True):

    d_radius = r / rg_param
    d_theta = ((2 * math.pi) / ry_param)

    if not overlap:
        # This variable denotes the maximum j_max value with respect to microtubule configuration on the first ring
        j_max_lim = j_max_bef_overlap(ry_param, N_param)
        max_d_tube = solve_d_rect(r, ry_param, rg_param, j_max_lim, 0)

        while d_tube <= 0 or d_tube > max_d_tube:
            d_tube = float(input(f"Select d_tube within the range: (0, {max_d_tube}] to avoid DL extraction region overlap: "))

    else:
        max_d_tube = solve_d_rect(r, ry_param, rg_param, ry_param - 1, 0)
        while d_tube <= 0 or d_tube > max_d_tube:
            d_tube = float(input(f"Select d_tube within the range: (0, {max_d_tube}]"))

    # What should be the highest rect_dist?

    for m in range(rg_param):
        if ceil:
            j_max = math.ceil((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)
        else:
            j_max = math.floor((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)
        print("Ring: ", m, "j_max: ", j_max)


# Decides the max j_max to prevent overlap in the domain\
@njit
def j_max_bef_overlap(ry_param, Microtubules):
    min_range = ry_param
    r = len(Microtubules)
    for n in range(r):
        # current microtubule
        c_n = Microtubules[n]
        # right-most microtubule
        r_n = Microtubules[(n+1) % r]
        # left-most microtubule
        l_n = Microtubules[(n-1) % r]
        min_range = min(min_range, abs(c_n - r_n - 1) % ry_param)
        min_range = min(min_range, abs(c_n - l_n - 1) % ry_param)
    # print(min_range)
    return min_range // 2
    # note if this function returns 0, there exists at least one overlapping region


def j_max_bef_overlap_no_JIT(N, Microtubules):
    min_range = N
    r = len(Microtubules)
    for n in range(r):
        # current microtubule
        c_n = Microtubules[n]
        # right-most microtubule
        r_n = Microtubules[(n+1) % r]
        # left-most microtubule
        l_n = Microtubules[(n-1) % r]
        min_range = min(min_range, abs(c_n - r_n - 1) % N)
        min_range = min(min_range, abs(c_n - l_n - 1) % N)
    # print(min_range)
    return min_range // 2
    # note if this function returns 0, there exists at least one overlapping region


@njit
def dict_gen(keys, values):
    d = Dict.empty(
        key_type=int64,
        value_type=int64
    )
    m = len(keys)
    n = len(values)

    assert m % n == 0

    group_size = m // n
    j = 0

    for i in range(m):
        d[keys[i]] = values[j]
        if (i + 1) % group_size == 0:
            j += 1
    return d


# collect ranges of centers across a modular ring and flatten them into a single container
@njit
def mod_range_flat(centers, radius, ring_len, sorted=False):
    N = ring_len
    total_len = int(len(centers) * (2 * radius + 1))
    flat_result = np.empty(total_len, dtype=np.int64)
    idx = 0

    for i in range(len(centers)):
        center = centers[i]
        for offset in range(-radius, radius + 1):
            val = (center + offset) % N
            flat_result[idx] = val
            idx += 1

    if sorted:
        flat_result = np.sort(flat_result)

    return flat_result


# This function/algorithm is currently under construction
def MTOC_offset_bound(bound_arr, rg_param, ry_param, offset_theta, offset_radius, disable_vert=False, disable_horz=False):

    if disable_horz:
        a = 0
    else:
        a = 1
    if disable_vert:
        b = 0
    else:
        b = 1

    offset_vert = ((1 - offset_radius) * np.sin(offset_theta)) * a
    offset_horz = ((1 - offset_radius) * np.cos(offset_theta)) * b

    delta_theta = (2 * np.pi) / ry_param
    delta_radius = 1 / rg_param

    for n in range(ry_param):
        ang_phi = n * delta_theta
        x = offset_horz * np.cos(ang_phi) + offset_vert * np.sin(ang_phi)
        rho = np.sqrt((x ** 2) + (offset_radius ** 2) - (offset_vert ** 2) - (offset_horz ** 2)) + x
        bound_arr[n] = np.ceil(rho * delta_radius)
    return


def solve_k(alpha, beta, d):
    if alpha == 0 or beta == 0 or d == 0:
        raise ValueError("alpha, beta, and d must be non-zero")

    pi = math.pi
    A = beta * (alpha ** 2) * d
    B = pi * alpha
    C = pi * beta

    discriminant = B**2 + 4 * A * C

    if discriminant < 0:
        raise ValueError("No real solution for k (discriminant < 0)")

    sqrt_disc = math.sqrt(discriminant)
    k1 = (B + sqrt_disc) / (2 * A)
    k2 = (B - sqrt_disc) / (2 * A)

    # Return the positive real root
    if k1 > 0 and k2 > 0:
        return min(k1, k2)  # take the smaller positive root for stability
    elif k1 > 0:
        return k1
    elif k2 > 0:
        return k2
    else:
        raise ValueError("No positive real solution for k")


def compute_d_i(alpha, beta, k):
    """
    Compute d from the general formula:
    d = [ (alpha/beta) * (k/2) + 1/2 ] * [ 2π / (alpha^2 * k^2) ]

    Parameters:
        alpha (float): parameter alpha (must be non-zero)
        beta (float): parameter beta (must be non-zero)
        k (float): the variable for which d is evaluated (must be non-zero)

    Returns:
        float: computed value of d
    """
    if alpha == 0 or beta == 0 or k == 0:
        raise ValueError("alpha, beta, and k must be non-zero")

    term1 = (alpha / beta) * (k / 2) + 0.5
    term2 = (2 * math.pi) / (alpha ** 2 * k ** 2)

    d = term1 * term2
    return d


def compute_d_ii(N, M, mu):
    term1 = (N / mu) * 0.5 + 0.5
    term2 = (2 * math.pi) / (M * N)
    return term1 * term2


def compute_M(N, mu):
    term1 = math.pi * 100
    term2 = 1/mu + 1/N
    return term1 * term2
