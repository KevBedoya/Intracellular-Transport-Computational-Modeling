import pandas as pd

from . import njit, numerical_tools as num, np
from computational_tools import struct_init


# Methods used to collect numerical results (via PDE solver) to conduct analyses on, e.g, mass(t), Phi(theta), Phi(rad), Rho(rad), ect.
# All methods below implement the space efficient two-step approach (in terms of how numerical data is stored across updates), i.e, [0][m][n] refers to the current (step k), and [1][m][n] refers to next (step k+1).

# v======================================== Mass stamp dependent analysis tools ========================================v


@njit
# (****) (****)
def comp_diffusive_angle_snapshots_mass_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param, N_LIST, D_LAYER, A_LAYER, PvT_DL_snapshots,
                                            checkpoint_collect_container, mass_retention_threshold=0.01, T_fixed_ring_seg=0.5, d_tube=0,
                                            domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                                            center_init_cond=True, m_init=0, n_init=0):

    print("Running optimized version.")

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).')

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] > ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    dRad = num.compute_dRad(ry_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    v_param *= -1

    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    mass_retained = 0

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    checkpoint_iter = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k == 0 or mass_retained > mass_retention_threshold:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT,
                                 switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", mass_retained)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)
            print(
                "********************************************************************************************************")

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_mass_stamp = checkpoint_collect_container[checkpoint_iter]
            if curr_mass_stamp * 0.99 < mass_retained < curr_mass_stamp * 1.01:
                PvT_DL_snapshots[checkpoint_iter] = D_LAYER[0][int(np.floor(rg_param * T_fixed_ring_seg))]
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
@njit
def comp_diffusive_rad_snapshots_mass_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param,
                                          N_LIST, D_LAYER, A_LAYER, R_fixed_angle, PvR_DL_snapshots, RvR_AL_snapshots,
                                          checkpoint_collect_container, mass_retention_threshold=0.01, domain_radius=1.0,
                                          D=1.0, mass_checkpoint=10 ** 6, d_tube=0,
                                          center_init_cond=True, m_init=0, n_init=0):

    print("Running optimized version.")

    R_fixed_angle = int(R_fixed_angle)

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)

    v_param *= -1

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    checkpoint_iter = 0

    mass_retained = 0

    k = 0
    while k == 0 or mass_retained > mass_retention_threshold:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT,
                                 switch_param_a, switch_param_b, v_param, d_tube)

        print(
            "********************************************************************************************************")
        if k > 0 and k % mass_checkpoint == 0:
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)
        print(
            "********************************************************************************************************")

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_mass_stamp = checkpoint_collect_container[checkpoint_iter]
            if curr_mass_stamp * 0.99 < mass_retained < curr_mass_stamp * 1.01:
                PvR_DL_snapshots[checkpoint_iter][0] = central_patch
                for m in range(rg_param):
                    PvR_DL_snapshots[checkpoint_iter][m + 1] = D_LAYER[0][m][R_fixed_angle]
                    RvR_AL_snapshots[checkpoint_iter][m] = A_LAYER[0][m][R_fixed_angle]
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER,
                                     N_LIST, v_param)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
@njit
# Collecting DL, central-patch, and AL snapshots for static heat-plot visualization.
def comp_diffusive_snapshots_mass_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param, N_LIST, D_LAYER, A_LAYER,
                                      HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots, checkpoint_collect_container,
                                      domain_radius=1.0, D=1.0, mass_retention_threshold=0.01, mass_checkpoint=10 ** 6, d_tube=0.0,
                                      center_init_cond=True, m_init=0, n_init=0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, D)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    v_param *= -1

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    mass_retained = 0
    MFPT = 0
    checkpoint_iter = 0
    k = 0

    while k == 0 or mass_retained > mass_retention_threshold:

        net_current_out = 0

        net_current_out += num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                                                    dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)
            print(
                "********************************************************************************************************")

        MFPT += net_current_out * k * dT ** 2

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_mass_stamp = checkpoint_collect_container[checkpoint_iter]
            if curr_mass_stamp * 0.99 < mass_retained < curr_mass_stamp * 1.01:
                MFPT_snapshots[checkpoint_iter] = MFPT
                HM_DL_snapshots[checkpoint_iter] = D_LAYER[0]
                HM_C_snapshots[checkpoint_iter] = central_patch
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]

        k += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ^======================================== Mass stamp dependent analysis tools ========================================^


# v======================================== Time stamp dependent analysis tools ========================================v


# (****) (****)
@njit
def comp_diffusive_angle_snapshots_time_dep(rg_param, ry_param, switch_param_a, switch_param_b, T_param, v_param,
                                            N_LIST, D_LAYER, A_LAYER, PvT_DL_snapshots, checkpoint_collect_container,
                                            T_fixed_ring_seg=0.5, d_tube=0.0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                                            center_init_cond=True, m_init=0, n_init=0):
    print("Running optimized version.")

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).')

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] > ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    dRad = num.compute_dRad(ry_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    v_param *= -1

    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    mass_retained = 0

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    checkpoint_iter = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k < K:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT,
                                 switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", mass_retained)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)
            print(
                "********************************************************************************************************")

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_stamp = np.floor(checkpoint_collect_container[checkpoint_iter] / dT)
            if k == curr_stamp:
                PvT_DL_snapshots[checkpoint_iter] = D_LAYER[0][int(np.floor(rg_param * T_fixed_ring_seg))]

                # 1/30/26
                # print("Time: ", checkpoint_collect_container[checkpoint_iter])
                # m_idx = int(np.floor(rg_param * T_fixed_ring_seg))
                # n_idx = int(N_LIST[0])
                # abs_err = num.comp_verify_bc(A_LAYER, D_LAYER, m_idx, n_idx, switch_param_a)
                # print("position: ", m_idx, "x", n_idx, "abs_err: ", abs_err)
                # print()
                # 1/30/26

                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def comp_diffusive_angle_snapshots_time_dep_matrix(rg_param, ry_param, switch_param_a, switch_param_b, T_param, v_param,
                                            N_LIST, D_LAYER, A_LAYER, diffusion_matrix, central_vector, checkpoint_collect_container, d_tube=0.0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                                            center_init_cond=True, m_init=0, n_init=0):
    print("Running optimized version.")

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).')

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] > ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    dRad = num.compute_dRad(ry_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    v_param *= -1

    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    mass_retained = 0

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    checkpoint_iter = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k < K:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT,
                                 switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", mass_retained)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)
            print(
                "********************************************************************************************************")

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_stamp = np.floor(checkpoint_collect_container[checkpoint_iter] / dT)
            if k == curr_stamp:
                diffusion_matrix[checkpoint_iter] = D_LAYER[0][0:rg_param-1]
                central_vector[checkpoint_iter] = central_patch
                # Ang_traj_mat[checkpoint_iter] = D_LAYER[0][0:rg_param - 1][0:N_LIST[1]-1]
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

@njit
def comp_BC_analysis_snapshots_time_dep(rg_param, ry_param, switch_param_a, switch_param_b, T_param, v_param, N_LIST, D_LAYER, A_LAYER, checkpoint,
                                           T_fixed_ring_seg=0.5, d_tube=0.0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                                           center_init_cond=True, m_init=0, n_init=0):
    print("Running optimized version.")

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).')

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] > ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    dRad = num.compute_dRad(ry_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    v_param *= -1
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0
    dl_mass = 1
    al_mass = 0

    curr_stamp = np.floor(checkpoint / dT)

    while k < K:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT,
                                 switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", dl_mass)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)
            print(
                "********************************************************************************************************")

        if k == curr_stamp:
            fixed_ring_seg = int(np.floor(rg_param * T_fixed_ring_seg))
            BC_ratio_snapshot = num.comp_verify_bc(A_LAYER, D_LAYER, fixed_ring_seg, N_LIST[0], switch_param_b)
            mass_retained = dl_mass + al_mass
            return BC_ratio_snapshot, mass_retained

        # Update mass for the next step
        dl_mass = num.calc_mass_diff(D_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param)
        al_mass = num.calc_mass_adv(A_LAYER, 0, dRad, dThe, rg_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
# - - - - - - - - - - - - - - - - - - - - - -


@njit
def comp_peak_time_mass_loss(rg_param, ry_param, switch_param_a, switch_param_b, T_param, v_param, Jrr_sum_timeseries,
                             N_LIST, D_LAYER, A_LAYER, relative_k, collection_factor, d_tube=0.0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                             center_init_cond=True, m_init=0, n_init=0):
    print("Running optimized version.")

    if len(N_LIST) > ry_param:
        raise IndexError(
            f'Too many angular indices supplied for microtubule positions: {len(N_LIST)} > {ry_param} (number of angular positions in domain).')

    for i in range(len(N_LIST)):
        if N_LIST[i] < 0 or N_LIST[i] > ry_param:
            raise IndexError(
                f'Angular index: {N_LIST[i]} falls outside of the legal index range: [0,{ry_param - 1}) under ry_param={ry_param}')

    dRad = num.compute_dRad(ry_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    v_param *= -1

    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    mass_retained = 0

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)
    k_step = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k < K:

        num.comp_DL_AL_kp1_2step_mass_loss_j_r_r(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                                                 dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        if k_step < relative_k and k % collection_factor == 0:
            Jrr_sum_timeseries[k_step] = num.comp_DL_AL_kp1_2step_mass_loss_j_r_r(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)
            k_step += 1

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", mass_retained)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)
            print(
                "********************************************************************************************************")

        mass_retained = num.calc_mass_diff(D_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param) + num.calc_mass_adv(A_LAYER, 0, dRad, dThe, rg_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
# - - - - - - - - - - - - - - - - - - - - - -


def bc_secant_computation(data_filepath):
    data = pd.read_csv(data_filepath)
    w = data["w"].values
    bc_ratios = data["BC_ratios"].values

    n = len(w)-1
    secant_array = np.zeros(n, dtype=np.float64)

    for i in range(n):
        secant_array[i] = (bc_ratios[i+1] - bc_ratios[i])/(w[i+1] - w[i])

    return secant_array

# (****) (****)
@njit
def comp_diffusive_rad_snapshots_time_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param, T_param,
                                          N_LIST, D_LAYER, A_LAYER, R_fixed_angle, PvR_DL_snapshots, RvR_AL_snapshots,
                                          checkpoint_collect_container, domain_radius=1.0, D=1.0, mass_checkpoint=10**6, d_tube=0,
                                          center_init_cond=True, m_init=0, n_init=0):

    print("Running optimized version.")

    R_fixed_angle = int(R_fixed_angle)

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)

    v_param *= -1

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    checkpoint_iter = 0

    mass_retained = 0

    k = 0
    while k < K:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT,
                                 switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)
            print(
                "********************************************************************************************************")

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_stamp = np.floor(checkpoint_collect_container[checkpoint_iter] / dT)
            if k == curr_stamp:
                PvR_DL_snapshots[checkpoint_iter][0] = central_patch
                for m in range(rg_param):
                    PvR_DL_snapshots[checkpoint_iter][m + 1] = D_LAYER[0][m][R_fixed_angle]
                    RvR_AL_snapshots[checkpoint_iter][m] = A_LAYER[0][m][R_fixed_angle]
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER,
                                     N_LIST, v_param)
        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (****) (****)
@njit
# Collecting DL, central-patch, and AL snapshots for static heat-plot visualization.
def comp_diffusive_snapshots_time_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param, T_param, N_LIST,
                                      D_LAYER, A_LAYER, HM_DL_snapshots, HM_C_snapshots, MFPT_snapshots,
                                      checkpoint_collect_container, domain_radius=1.0, D=1.0, mass_checkpoint=10 ** 6, d_tube=0.0,
                                      center_init_cond=True, m_init=0, n_init=0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, D)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    K_param = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    v_param *= -1

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    mass_retained = 0
    MFPT = 0
    checkpoint_iter = 0
    k = 0

    while k < K_param:

        net_current_out = 0

        net_current_out += num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                                                    dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)
        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)
            print(
                "********************************************************************************************************")

        MFPT += net_current_out * k * dT ** 2

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_stamp = np.floor(checkpoint_collect_container[checkpoint_iter] / dT)
            if k == curr_stamp:
                MFPT_snapshots[checkpoint_iter] = MFPT
                HM_DL_snapshots[checkpoint_iter] = D_LAYER[0]
                HM_C_snapshots[checkpoint_iter] = central_patch
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]

        k += 1

# <***************************** Temporarily retired *****************************>
# @njit(nopython=ENABLE_JIT, cache=ENABLE_CACHE)
# def comp_mass_loss_glb_pk(rings, rays, a, b, v, tube_placements, diffusive_layer, advective_layer, r=1.0, d=1.0,
#                           mass_retention_threshold=0.01):
#     """
#
#     Prints biophysical metrics including MFPT and the dimensionless time taken to reach the global-maximum of the mass-loss-rate as a
#     function of time, J(t).
#
#     :param rings: (float) # of radial rings in the cellular domain
#     :param rays: (float) # of angular rays in the cellular domain
#     :param a: (float) the switch rate onto the diffusive layer
#     :param b: (float) the switch rate onto the advective layer
#     :param v: (float) particle velocity on microtubules/filaments
#     :param tube_placements: (list(int)) discrete microtubule/filament positions between [0, rays-1]
#
#     :param diffusive_layer: [np.zeros((2, rings, rays), dtype=np.float64)] a 2 * rings * ray container to collect density at the diffusive layer
#     :param advective_layer: [np.zeros((2, rings, rays), dtype=np.float64)] a 2 * rings * ray container to collect density at the advective layer
#
#     :param r: (float) cellular radius, by default r=1.
#     :param d: (float) diffusion constant, by default d=1
#     :param mass_retention_threshold: (float) the amount of mass remaining in the domain until termination
#     :param mixed_config (bool) toggling the mixed configuration
#     :return: void
#     """
#
#     if ENABLE_JIT:
#         print("Running optimized version.")
#
#     if len(tube_placements) > rays:
#         raise IndexError(
#             f'Too many microtubules requested: {len(tube_placements)}, within domain of {rays} angular rays.')
#
#     for i in range(len(tube_placements)):
#         if tube_placements[i] < 0 or tube_placements[i] > rays:
#             raise IndexError(f'Angle {tube_placements[i]} is out of bounds, your range should be [0, {rays - 1}]')
#
#     d_radius = r / rings
#     d_theta = ((2 * math.pi) / rays)
#     d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)
#
#     phi_center = 1 / (math.pi * (d_radius * d_radius))
#
#     mass_retained = 0
#     # Mean first passage time
#     m_f_p_t = 0
#
#     mass_loss_step_i = 0
#     mass_loss_step_i_plus = 0
#
#     diff_special_angles = np.empty(rays, dtype=np.int64)
#
#     # if mixed_config:
#     #     diff_special_angles = sup.mod_range_flat_sorted(tube_placements, 1, rays)
#
#     # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     k = 0
#
#     while k == 0 or mass_retained > mass_retention_threshold:
#
#         net_current_out = 0
#
#         m = 0
#
#         while m < rings:
#             angle_index = 0
#
#             if m == rings - 2:
#                 if k % 2 == 0:
#                     mass_loss_step_i = num.calc_loss_mass_j_r_r(diffusive_layer, 0, d_radius, d_theta, rings, rays)
#                 else:
#                     mass_loss_step_i_plus = num.calc_loss_mass_j_r_r(diffusive_layer, 0, d_radius, d_theta, rings, rays)
#                 if k > 0 and k % 2 != 0:
#                     if mass_loss_step_i > mass_loss_step_i_plus:
#                         print("Mass loss J(t) global peak time (in dimensional units): ", (k - 1) * d_time,
#                               "# time steps: ", k - 1, "MFPT: ", m_f_p_t)
#                         return
#             n = 0
#             while n < rays:
#                 if m == rings - 1:
#                     diffusive_layer[1][m][n] = 0
#                 else:
#                     diffusive_layer[1][m][n] = num.u_density(diffusive_layer, 0, m, n, d_radius, d_theta, d_time,
#                                                              phi_center, rings, advective_layer, angle_index, a,
#                                                              b,
#                                                              tube_placements)
#                     if n == tube_placements[angle_index]:
#                         advective_layer[1][m][n] = num.u_tube(advective_layer, diffusive_layer, 0, m, n, a, b, v,
#                                                               d_time, d_radius, d_theta)
#                         if angle_index < len(tube_placements) - 1:
#                             angle_index = angle_index + 1
#                 if m == rings - 2:
#                     net_current_out += num.j_r_r(diffusive_layer, 0, m, n, d_radius, 0) * rings * d_radius * d_theta
#                 n += 1
#             m += 1
#
#         m_f_p_t += net_current_out * k * d_time * d_time
#         k += 1
#
#         mass_retained = num.calc_mass(diffusive_layer, advective_layer, 0, d_radius, d_theta, phi_center, rings, rays,
#                                       tube_placements)
#         phi_center = num.u_center(diffusive_layer, 0, d_radius, d_theta, d_time, phi_center, advective_layer,
#                                   tube_placements, v)
#         diffusive_layer[0] = diffusive_layer[1]
#         advective_layer[0] = advective_layer[1]
#
#
        # <<<< ------------- Updating DL and AL for the K+1-th step ------------- >>>>
        # m = 0
        #
        # while m < rg_param:
        #
        #     # The advective angle index 'aIdx'
        #     aIdx = 0
        #     n = 0
        #
        #     while n < ry_param:
        #         if m == rg_param - 1:
        #             D_LAYER[1][m][n] = 0
        #         else:
        #             if n in d_list[m]:
        #                 D_LAYER[1][m][n] = num.u_density_rect(D_LAYER, 0, m, n, dRad,
        #                                                       dThe, dT, central_patch, rg_param,
        #                                                       A_LAYER, int(d_list[m][n]), switch_param_a,
        #                                                       switch_param_b, d_tube)
        #             else:
        #                 D_LAYER[1][m][n] = num.u_density(D_LAYER, 0, m, n, dRad, dThe,
        #                                                  dT, central_patch, rg_param, A_LAYER,
        #                                                  aIdx, switch_param_a, switch_param_b, N_LIST)
        #             if n == N_LIST[aIdx]:
        #
        #                 A_LAYER[1][m][n] = num.u_tube_rect(A_LAYER, D_LAYER, 0, m, n,
        #                                                    switch_param_a, switch_param_b,
        #                                                    v_param, dT, dRad, dThe, d_tube)
        #                 if aIdx < len(N_LIST) - 1:
        #                     aIdx += 1
        #         n += 1
        #     m += 1
        # <<<< ------------- Updating DL and AL for the K+1-th step ------------- >>>>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ^======================================== Time stamp dependent analysis tools ========================================^


# (****) (****)
@njit
def advance_mass_analysis(k_start, k_end, rg_param, ry_param, switch_param_a, switch_param_b,
                          v_param, N_LIST, D_LAYER, A_LAYER,
                          MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
                          MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor,
                          relative_k, d_list, d_tube, dRad, dThe, dT, D, mass_checkpoint,
                          central_patch, dl_mass, al_mass, MA_k_step):
    """Advance the solution from timestep ``k_start`` to ``k_end``.

    Split out of comp_mass_analysis_respect_to_time so the time loop can be
    driven in chunks. Everything the loop carries across one iteration is either
    mutated in place (D_LAYER, A_LAYER, the timeseries) or passed in and returned
    (the four scalars), so stopping at any k and resuming from it is exactly
    equivalent to running straight through -- which is what makes checkpointing
    possible, and what tests/test_checkpoint_resume.py asserts.

    ``v_param`` must already be negated by the caller, and ``d_list`` already
    built: both are set up once per run, not per chunk.

    Returns the carried scalars: (central_patch, dl_mass, al_mass, MA_k_step).
    """
    k = k_start
    while k < k_end:

        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print(
                "********************************************************************************************************")
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", dl_mass,
                  "Current AL mass: ", al_mass)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)
            print(
                "********************************************************************************************************")

        # Collect mass
        if MA_k_step < relative_k and k % MA_collection_factor == 0:
            MA_DL_timeseries[MA_k_step] = dl_mass
            MA_AL_timeseries[MA_k_step] = al_mass
            MA_TM_timeseries[MA_k_step] = dl_mass + al_mass
            MA_ALoT_timeseries[MA_k_step] = al_mass / MA_TM_timeseries[MA_k_step]
            MA_ALoI_timeseries[MA_k_step] = al_mass / D
            MA_k_step += 1

            # fixed_ring_seg = int(np.floor(rg_param * T_fixed_ring_seg))
            # PvT_DL_snapshots[snapshot_idx] = D_LAYER[ 0 ][ fixed_ring_seg ]
            # BC_ratio_snapshot = num.comp_verify_bc_ratio(A_LAYER, D_LAYER, fixed_ring_seg, N_LIST[0], switch_param_a)
            # print(BC_ratio_snapshot)

        # Update mass for the next step
        dl_mass = num.calc_mass_diff(D_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param)
        al_mass = num.calc_mass_adv(A_LAYER, 0, dRad, dThe, rg_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return central_patch, dl_mass, al_mass, MA_k_step


# (****) Main mass-analysis driver (****)
# Deliberately NOT @njit: it drives the njit kernel above in chunks, and a
# checkpointing hook has to do file I/O between chunks, which nopython mode
# cannot (np.save fails to compile there). The heavy work is all inside
# advance_mass_analysis, so leaving the driver in Python costs one function call
# per chunk -- nothing against a chunk of hundreds of thousands of timesteps.
def comp_mass_analysis_respect_to_time(rg_param, ry_param, switch_param_a, switch_param_b, v_param, T_param,
                                       N_LIST, D_LAYER, A_LAYER, MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
                                       MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor,
                                       relative_k, d_tube=0, domain_radius=1.0, D=1.0, mass_checkpoint=10**6,
                                       center_init_cond=True, m_init=0, n_init=0,
                                       chunk_steps=0, on_chunk=None):
    """Solve to ``T_param``, collecting layer masses as it goes.

    ``chunk_steps`` splits the time loop into pieces of that many steps; 0 (the
    default) runs it in a single chunk, which is the original behaviour.
    ``on_chunk``, if given, is called as ``on_chunk(k, state)`` after each chunk
    with the timestep reached and the carried scalars -- the hook a checkpoint
    writer uses. Neither affects the numerics: chunk boundaries are exactly
    equivalent to running straight through.
    """

    print("Running optimized version.")

    # Initialize constants
    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    v_param *= -1
    # Initialize the ring position (m) dependent extraction range dictionary
    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    # Initialize layer masses
    dl_mass = 1
    al_mass = 0

    # Mass data collection iterator
    MA_k_step = 0

    step = K if chunk_steps <= 0 else min(chunk_steps, K)
    k = 0
    while k < K:
        k_end = min(k + step, K)
        central_patch, dl_mass, al_mass, MA_k_step = advance_mass_analysis(
            k, k_end, rg_param, ry_param, switch_param_a, switch_param_b,
            v_param, N_LIST, D_LAYER, A_LAYER,
            MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries,
            MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor,
            relative_k, d_list, d_tube, dRad, dThe, dT, D, mass_checkpoint,
            central_patch, dl_mass, al_mass, MA_k_step)
        k = k_end
        if on_chunk is not None:
            on_chunk(k, (central_patch, dl_mass, al_mass, MA_k_step))

    return central_patch, dl_mass, al_mass, MA_k_step
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
@njit
def comp_until_mass_depletion(rg_param, ry_param, switch_param_a, switch_param_b, v_param, N_LIST, D_LAYER, A_LAYER,
                              domain_radius=1.0, D=1.0,
                              mass_retention_threshold=0.01, d_tube=0.0,
                              center_init_cond=True, m_init=0, n_init=0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, D)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    if center_init_cond:
        central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    else:
        central_patch = 0.0
        D_LAYER[0][m_init][n_init] = num.compute_init_cond_patch(rg_param, ry_param, m_init, domain_radius)
    v_param *= -1

    d_list = struct_init.build_d_tube_map_dense(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    mass_retained = 0

    k = 0

    while k == 0 or mass_retained > mass_retention_threshold:
        num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]

        k += 1

    return k * dT
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
