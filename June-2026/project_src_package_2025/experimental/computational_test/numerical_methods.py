from project_src_package_2025.computational_tools import numerical_tools as num, struct_init
from project_src_package_2025.system_configuration import sys_config
from numba import njit

ENABLE_JIT = sys_config.ENABLE_NJIT
ENABLE_CACHING = sys_config.ENABLE_NUMBA_CACHING


# Compute positional density (m,n) across diffusive layer (DL) and advective layer (AL) for k+1-th step under the 2-step time approach
# D_LAYER and A_LAYER are passed by reference
# Returns the net_current_out for the computation of MFPT
@njit(nopython=ENABLE_JIT, cache=ENABLE_CACHING)
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
                    D_LAYER[1][m][n] = num.u_density_rect(D_LAYER, 0, m, n, dRad, dThe,
                                                          dT,
                                                          central_patch, rg_param, A_LAYER,
                                                          int(d_list[m][n]), switch_param_a, switch_param_b, d_tube)
                else:
                    D_LAYER[1][m][n] = num.u_density(D_LAYER, 0, m, n, dRad, dThe,
                                                     dT,
                                                     central_patch, rg_param, A_LAYER,
                                                     aIdx,
                                                     switch_param_a, switch_param_b,
                                                     N_LIST)
                if n == N_LIST[aIdx]:

                    A_LAYER[1][m][n] = num.u_tube_rect(A_LAYER, D_LAYER, 0, m,
                                                       n, switch_param_a, switch_param_b, v_param, dT, dRad, dThe,
                                                       d_tube)

                    if aIdx < len(N_LIST) - 1:
                        aIdx += 1

                if m == rg_param - 2:
                    net_current_out += num.j_r_r(D_LAYER, 0, m, n, dRad, 0) * rg_param * dRad * dThe
            n += 1
        m += 1

    return net_current_out


@njit(nopython=ENABLE_JIT, cache=ENABLE_CACHING)
def compute_MFPT_until_T(rg_param, ry_param, v_param, switch_param_a, switch_param_b, T_param, N_LIST, D_LAYER, A_LAYER,
                         domain_radius=1.0, D=1.0, d_tube=0.0):
    if ENABLE_JIT:
        print("Running optimized version.")

    # Initialize constants

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    v_param *= -1

    # Initialize the ring position (m) dependent extraction range dictionary

    d_list = struct_init.build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    # Initialize MFPT

    MFPT = 0

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0
    while k < K:

        net_current_out = comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                                               dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        # print(net_current_out)

        MFPT += net_current_out * k * dT ** 2

        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        # transfer updated density info from the next step to the current
        # num.update_layer_inplace(diffusive_layer[0], diffusive_layer[1], rays, rings)
        # num.update_layer_inplace(advective_layer[0], advective_layer[1], rays, rings)
        k += 1
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return MFPT
