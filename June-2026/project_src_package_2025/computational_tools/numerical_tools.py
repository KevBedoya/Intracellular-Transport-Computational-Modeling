from . import njit, np
import pandas as pd


# (****) Main numerical PDE solver implemented under the 2-step (time-step) method (****)
@njit
def comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                         dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube):
    m = 0
    net_current_out = 0
    while m < rg_param:

        # The advective angle index 'aIdx'
        aIdx = 0
        n = 0

        while n < ry_param:
            if m == rg_param - 1:
                D_LAYER[1][m][n] = 0
            else:
                if n in d_list[m]:
                    # n denotes a discrete position (an extraction region ray) within an extraction region centered at a microtubule (indices contained in N_LIST)
                    # if the iteration steps on an extraction region ray at ring m, then:
                    # int(d_list[m][n]) is the corresponding microtubule position of the extraction region ray (n) at ring (m)
                    corr_MT_pos = int(d_list[m][n])

                    D_LAYER[1][m][n] = u_density_rect(D_LAYER, 0, m, n,
                                                      dRad, dThe, dT, central_patch,
                                                      rg_param, A_LAYER, corr_MT_pos,
                                                      switch_param_a, switch_param_b, d_tube)
                else:
                    D_LAYER[1][m][n] = u_density(D_LAYER, 0, m, n,
                                                 dRad, dThe, dT, central_patch,
                                                 rg_param, A_LAYER, aIdx,
                                                 switch_param_a, switch_param_b, N_LIST)
                if n == N_LIST[aIdx]:

                    A_LAYER[1][m][n] = u_tube_rect(A_LAYER, D_LAYER, 0, m, n,
                                                   switch_param_a, switch_param_b,
                                                   v_param, dT, dRad, dThe, d_tube)

                    if aIdx < len(N_LIST) - 1:
                        aIdx += 1

                if m == rg_param - 2:
                    net_current_out += j_r_r(D_LAYER, 0, m, n, dRad, 0) * rg_param * dRad * dThe
            n += 1
        m += 1

    return net_current_out

def find_max_with_time(filepath):
    data = pd.read_csv(filepath)

    t = data.iloc[:, 0].values
    f = data.iloc[:, 1].values

    left, right = 0, len(f) - 1
    while left < right:
        mid = (left + right) // 2
        if f[mid] < f[mid + 1]:
            left = mid + 1
        else:
            right = mid

    return t[left], f[left]

@njit
def comp_DL_AL_kp1_2step_mass_loss_j_r_r(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                         dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube):
    m = 0
    while m < rg_param:

        # The advective angle index 'aIdx'
        aIdx = 0
        n = 0

        while n < ry_param:
            if m == rg_param - 1:
                D_LAYER[1][m][n] = 0

            else:
                if n in d_list[m]:
                    # n denotes a discrete position (an extraction region ray) within an extraction region centered at a microtubule (indices contained in N_LIST)
                    # if the iteration steps on an extraction region ray at ring m, then:
                    # int(d_list[m][n]) is the corresponding microtubule position of the extraction region ray (n) at ring (m)
                    corr_MT_pos = int(d_list[m][n])

                    D_LAYER[1][m][n] = u_density_rect(D_LAYER, 0, m, n,
                                                      dRad, dThe, dT, central_patch,
                                                      rg_param, A_LAYER, corr_MT_pos,
                                                      switch_param_a, switch_param_b, d_tube)
                else:
                    D_LAYER[1][m][n] = u_density(D_LAYER, 0, m, n,
                                                 dRad, dThe, dT, central_patch,
                                                 rg_param, A_LAYER, aIdx,
                                                 switch_param_a, switch_param_b, N_LIST)
                if n == N_LIST[aIdx]:

                    A_LAYER[1][m][n] = u_tube_rect(A_LAYER, D_LAYER, 0, m, n,
                                                   switch_param_a, switch_param_b,
                                                   v_param, dT, dRad, dThe, d_tube)

                    if aIdx < len(N_LIST) - 1:
                        aIdx += 1

                if m == rg_param - 2:
                    return calc_loss_mass_j_r_r(D_LAYER, 0, dRad, dThe, rg_param, ry_param)
            n += 1
        m += 1

@njit
def comp_verify_bc(A_layer, D_layer, m_idx, n_idx, w_param):
    M = len(D_layer[0])
    N = len(D_layer[0][0])

    delta_theta = compute_dThe(N)
    delta_radius = compute_dRad(M)
    r = (m_idx + 1) * delta_radius

    # LHS = w_param * r * delta_theta * D_layer[0][m_idx][n_idx] - w_param * A_layer[0][m_idx][n_idx]
    LHS = D_layer[0][m_idx][n_idx] - A_layer[0][m_idx][n_idx]
    FDQ = (D_layer[0][m_idx][(n_idx + 1) % N] - D_layer[0][m_idx][n_idx]) / delta_theta

    RHS = -(2.0/r) * FDQ

    # print("FDQ: ", FDQ)
    # print("LHS: ", LHS)
    # print("RHS: ", RHS)

    # abs_diff = abs(LHS - RHS)
    ratio = abs(RHS)/LHS

    return ratio

# (****) Update density (phi) at a position (m,n) for timestep k+1 on DL. [non-d-tube update] (****)
@njit
def u_density(phi, k, m, n, d_radius, d_theta, d_time, central, rings, rho, mt_pos, a, b, tube_placements):
    """

    Calculate particle density at a position (m,n) on the diffusive layer at a time-point k.
    (Positions are relative to patches across our domain for the diffusive layer)

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param m: (int) position of radial ring
    :param n: (int) position of angular ray
    :param d_radius: (float) delta_radius
    :param d_theta: (float) delta_theta
    :param d_time: (float) delta_time
    :param central: (float) particle density at the center
    :param rings: (int) # of radial rings in the domain
    :param rho: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param mt_pos: (int) indexed position from the 'tube_placements' container
    (to specify particle density on the advective layer relative to microtubule/filament)

    :param a: (float) switch rate onto the diffusive layer (switch-on rate)
    :param b: (float) switch rate onto the advective layer (switch-off rate)
    :param tube_placements: (list(int)) discrete microtubule/filament positions between [0, rays-1]
    :return: particle density at a position (m,n) on the diffusive layer
    """

    current_density = phi[k][m][n]

    component_a = ((m+2) * j_r_r(phi, k, m, n, d_radius, rings)) - ((m+1) * j_l_r(phi, k, m, n, d_radius, central))

    component_a *= d_time / ((m+1) * d_radius)

    component_b = (j_r_t(phi, k, m, n, d_radius, d_theta)) - (j_l_t(phi, k, m, n, d_radius, d_theta))

    component_b *= d_time / ((m+1) * d_radius * d_theta)

    if n == tube_placements[mt_pos]:
        component_c = (a * phi[k][m][n]) * d_time - (((b * rho[k][m][n]) * d_time) / ((m+1) * d_radius * d_theta))
    else:
        component_c = 0

    return current_density - component_a - component_b - component_c


# (****) Update density (phi) at a position (m,n) for timestep k+1 on DL. [d_tube update] (****)
@njit
def u_density_rect(phi, k, m, n, d_radius, d_theta, d_time, central, rings, rho, mt_pos, a, b, d_tube):
    """

    Calculate particle density at a position (m,n) on the diffusive layer at a time-point k.
    (Positions are relative to patches across our domain for the diffusive layer)

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param m: (int) position of radial ring
    :param n: (int) position of angular ray
    :param d_radius: (float) delta_radius
    :param d_theta: (float) delta_theta
    :param d_time: (float) delta_time
    :param central: (float) particle density at the center
    :param rings: (int) # of radial rings in the domain
    :param rho: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param mt_pos: (int) indexed position from the 'tube_placements' container
    (to specify particle density on the advective layer relative to microtubule/filament)

    :param a: (float) switch rate onto the diffusive layer (switch-on rate)
    :param b: (float) switch rate onto the advective layer (switch-off rate)
    :param d_tube (float)
    :return: particle density at a position (m,n) on the diffusive layer
    """

    j_max = np.ceil((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)

    current_density = phi[k][m][n]

    component_a = ((m+2) * j_r_r(phi, k, m, n, d_radius, rings)) - ((m+1) * j_l_r(phi, k, m, n, d_radius, central))

    component_a *= d_time / ((m+1) * d_radius)

    component_b = (j_r_t(phi, k, m, n, d_radius, d_theta)) - (j_l_t(phi, k, m, n, d_radius, d_theta))

    component_b *= d_time / ((m+1) * d_radius * d_theta)

    component_c = a * phi[k][m][n] * d_time - (b * rho[k][m][mt_pos] * d_time) / ((1 + 2 * j_max) * (m+1) * d_radius * d_theta)

    return current_density - component_a - component_b - component_c


# (****)  (****)
@njit
def u_center(phi, k, d_radius, d_theta, d_time, curr_central, rho, tube_placements, v):
    """
    Calculates the particle density in the central patch.

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param d_radius: (float) delta-radius
    :param d_theta: (float) delta-theta
    :param d_time: (float) delta-time
    :param curr_central: (float) the current particle density at the center
    :param rho: (float) particle density on the
    :param tube_placements: (list(int)) discrete microtubule/filament positions between [0, rays-1]
    :param v: (float) velocity across the advective layer
    :return: particle density for the center patch
    """

    total_sum = 0
    for n in range(len(phi[0][0])):
        total_sum += j_l_r(phi, k, 0, n, d_radius, curr_central)

    total_sum *= (d_theta * d_time) / (np.pi * d_radius)
    diffusive_sum = curr_central - total_sum

    advective_sum = 0

    for i in range(len(tube_placements)):
        angle = tube_placements[i]
        j_l = rho[k][0][angle] * v
        advective_sum += (abs(j_l) * d_time) / (np.pi * d_radius * d_radius)

    return diffusive_sum + advective_sum


# (****)  (****)
@njit
def u_tube(rho, phi, k, m, n, a, b, v, d_time, d_radius, d_theta):
    """

    Calculate particle density at a position (m,n) on the advective layer at a time-point k.

    :param rho: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) current time step
    :param m: (int) position of the radial ring
    :param n: (int) position of the angular ray
    :param a: (float) switch-on rate
    :param b: (float) switch-off rate
    :param v: (float) velocity across the advective layer
    :param d_time: (float) delta-time
    :param d_radius: (float) delta-radius
    :param d_theta: (float) delta-theta
    :return: particle density at position (m,n) on the advective layer.
    """

    # N = phi.shape[2]
    N = len(phi[k][m])
    j_l = v * rho[k][m][n]
    if m == N - 1:
        j_r = 0
    else:
        j_r = v * rho[k][m+1][n]

    component_a = (rho[k][m][n] - ((j_r - j_l) * (1/d_radius)) * d_time)
    component_b = phi[k][m][n]
    component_b *= a * (m+1) * (d_radius * d_theta * d_time)
    component_c = b * rho[k][m][n] * d_time
    #
    return component_a + component_b - component_c


# (****)  (****)
@njit
def u_tube_rect(rho, phi, k, m, n, a, b, v, d_time, d_radius, d_theta, d_tube):
    """

    Calculate particle density at a position (m,n) on the advective layer at a time-point k.

    :param rho: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) current time step
    :param m: (int) position of the radial ring
    :param n: (int) position of the angular ray
    :param a: (float) switch-on rate
    :param b: (float) switch-off rate
    :param v: (float) velocity across the advective layer
    :param d_time: (float) delta-time
    :param d_radius: (float) delta-radius
    :param d_theta: (float) delta-theta
    :param d_tube (int)
    :return: particle density at position (m,n) on the advective layer.
    """

    j_l = v * rho[k][m][n]
    if m == len(phi[k][m]) - 1:
        j_r = 0
    else:
        j_r = v * rho[k][m+1][n]

    component_a = (rho[k][m][n] - ((j_r - j_l) * (1/d_radius)) * d_time)

    N = len(phi[k][m])

    j_max = np.ceil((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)

    component_b = 0
    for j in range(-j_max, j_max + 1):
        component_b += phi[k][m][(n + j) % N]
    component_b *= a * (m+1) * (d_radius * d_theta * d_time)

    component_c = b * rho[k][m][n] * d_time

    return component_a + component_b - component_c


# (****)  (****)
@njit
def calc_mass_diff(phi, k, d_radius, d_theta, curr_central, rings, rays):
    diffusive_mass = 0
    for m in range(rings):
        for n in range(rays):
            diffusive_mass += phi[k][m][n] * (m + 1)
    diffusive_mass *= (d_radius * d_radius) * d_theta
    return (curr_central * np.pi * d_radius * d_radius) + diffusive_mass


# (****)  (****)
@njit
def calc_mass_adv(rho, k, d_radius, d_theta, rings, tube_placements):
    advective_mass = 0
    for i in range(len(tube_placements)):
        angle = tube_placements[i]
        for m in range(rings):
            advective_mass += rho[k][m][angle] * d_radius

    return advective_mass


# (****)  (****)
@njit
def calc_mass(phi, rho, k, d_radius, d_theta, curr_central, rings, rays, tube_placements):
    """
    Calculates mass across the whole domain.

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param rho: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param d_radius: (float) delta-radius
    :param d_theta: (float) delta-theta
    :param curr_central: (float) the current particle density at the center
    :param rings: (int) # of radial rings in the domain
    :param rays: (int) # of angular rays in the domain
    :param tube_placements: (list(int)) discrete microtubule/filament positions between [0, rays-1]
    :return: domain mass
    """
    #
    # diffusive_mass = 0
    # for m in range(rings):
    #     for n in range(rays):
    #         diffusive_mass += phi[k][m][n] * (m+1)
    # diffusive_mass *= (d_radius * d_radius) * d_theta
    #
    # advective_mass = 0
    #
    # for i in range(len(tube_placements)):
    #     angle = tube_placements[i]
    #     for m in range(rings):
    #         advective_mass += rho[k][m][angle] * d_radius

    return calc_mass_diff(phi, k, d_radius, d_theta, curr_central, rings, rays) + calc_mass_adv(rho, k, d_radius, d_theta, rings, tube_placements)

    # return (curr_central * np.pi * d_radius * d_radius) + diffusive_mass + advective_mass


# (****) Compute the initial condition of the central patch (****)
@njit
def compute_init_cond_cent(rg_param, domain_radius=1.0):
    return 1 / (np.pi * compute_dRad(rg_param, domain_radius) ** 2)


# (****)  (****)
@njit
def j_r_r(phi, k, m, n, d_radius, rings):
    """

    Calculates the rightwards radial current

    Refer to project_src_package_2025/computational_tools/visual-discretization-demos for
    more information on the implementation/graphics of radial/angular currents

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param m: (int) position of radial ring
    :param n: (int) position of angular ray
    :param d_radius: (float) delta-radius
    :param rings: (int) # of radial rings in the domain
    :return: rightwards radial current
    """
    curr_ring = phi[k][m][n]
    if m == rings - 1:
        next_ring = 0
    else:
        next_ring = phi[k][m+1][n]
    result = -1 * ((next_ring - curr_ring) / d_radius)
    return result


# (****)  (****)
@njit
def j_l_r(phi, k, m, n, d_radius, central):
    """

    Calculates the leftwards radial current

    Refer to project_src_package_2025/computational_tools/visual-discretization-demos for
    more information on the implementation/graphics of radial/angular currents

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param m: (int) position of radial ring
    :param n: (int) position of radial ring
    :param d_radius: (float) delta-radius
    :param central (float) the current particle density at the center
    :return: leftwards radial current
    """
    curr_ring = phi[k][m][n]
    if m == 0:
        prev_ring = central
    else:
        prev_ring = phi[k][m-1][n]
    return -1 * ((curr_ring - prev_ring) / d_radius)


# (****)  (****)
@njit
def j_r_t(phi, k, m, n, d_radius, d_theta):
    """
    Calculates the rightwards angular current

    Refer to project_src_package_2025/computational_tools/visual-discretization-demos for
    more information on the implementation/graphics of radial/angular currents

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param m: (int) position of radial ring
    :param n: (int) position of radial ring
    :param d_radius: (float) delta-radius
    :param d_theta: (float) delta-theta
    :return: rightwards angular current
    """
    b = len(phi[k][m])
    return -1 * (phi[k][m][(n+1) % b] - phi[k][m][n]) / ((m+1) * d_radius * d_theta)


# (****)  (****)
@njit
def j_l_t(phi, k, m, n, d_radius, d_theta):
    """
       Calculates the leftwards angular current

       Refer to project_src_package_2025/computational_tools/visual-discretization-demos for
       more information on the implementation/graphics of radial/angular currents

       :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
       :param k: (int) time-point
       :param m: (int) position of radial ring
       :param n: (int) position of radial ring
       :param d_radius: (float) delta-radius
       :param d_theta: (float) delta-theta
       :return: leftwards angular current
       """
    b = len(phi[k][m])
    return -1 * (phi[k][m][n] - phi[k][m][(n-1) % b]) / ((m+1) * d_radius * d_theta)


# (****) Compute delta radius (****)
@njit
def compute_dRad(rg_param, domain_radius=1.0):
    return domain_radius / rg_param


# (****) Compute delta theta (****)
@njit
def compute_dThe(ry_param):
    return (2 * np.pi) / ry_param


# (****) Compute delta time (****)
@njit
def compute_dT(rg_param, ry_param, domain_radius=1.0, D=1.0):
    dRad = compute_dRad(rg_param, domain_radius)
    dThe = compute_dThe(ry_param)
    dT = (0.1 * min(dRad ** 2, dThe ** 2 * dRad ** 2)) / (2 * D)
    return dT


# (****) Compute K; the number of discrete time-steps corresponding to T_param (****)
@njit
def compute_K(rg_param, ry_param, T_param, domain_radius=1.0, D=1.0):
    return int(np.floor(T_param / compute_dT(rg_param, ry_param, domain_radius, D)))


# (****) (****)
@njit
def convert_K_to_T(rg_param, ry_param, K_param, domain_radius=1.0, D=1.0):
    return K_param * compute_dT(rg_param, ry_param, domain_radius, D)


# (****)  (****)
@njit
def calc_loss_mass_j_r_r(phi, k, d_radius, d_theta, rings, rays):
    """
    Calculates the amount of mass exiting the last ring of patches in the domain using radial currents.

    :param phi: (3-D float array) With 2 time points, m rings (rows), and n rays (columns)
    :param k: (int) time-point
    :param d_radius: (float) delta_radius
    :param d_theta: (float) delta_theta
    :param position: (int)
    :param rings (int)
    :param rays: (int) # of angular rays in the domain
    :return: total mass exiting the final ring of patches
    """
    total_sum = 0.0
    for n in range(rays):
        J_n = j_r_r(phi, k, rings-2, n, d_radius, 0)
        total_sum += J_n

    total_sum *= rings * d_radius * d_theta

    return total_sum


# <***************************** EXPERIMENTAL/REQUIRES MORE TESTING *****************************>


# <****> In_place transfer from k+1 th frame to the kth (in PDE solver for DL & AL) <****>
@njit
def update_layer_inplace(layer_target, layer_source, N, M):
    for i in range(M):
        for j in range(N):
            layer_target[i][j] = layer_source[i][j]