from . import njit, numerical_tools as num, np
from computational_tools import struct_init


# (****) (****)
@njit
def comp_mfpt_by_mass_loss(rg_param, ry_param, switch_param_a, switch_param_b, v_param,
                           N_LIST, D_LAYER, A_LAYER, mass_checkpoint=10**6,
                           domain_radius=1.0, D=1.0, mass_retention_threshold=0.01, d_tube=0.0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    d_list = struct_init.build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube, domain_radius)
    v_param *= -1
    k = 0
    mass_retained = 0
    MFPT = 0

    while k == 0 or mass_retained > mass_retention_threshold:

        net_current_out = num.comp_DL_AL_kp1_2step(ry_param, ry_param, d_list, D_LAYER, central_patch, A_LAYER,
                                                   N_LIST, dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        MFPT += net_current_out * k * dT ** 2
        # Implemented to provide occasional status checks/metrics during MFPT calculation
        if k > 0 and k % mass_checkpoint == 0:
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)

        # This block will need to be modified for performance related optimizations (the computation of mass(k) is independent from computing MFPT)
        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch,
                                      rg_param, ry_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        k += 1

    sim_time_duration = dT * k

    return MFPT, sim_time_duration


# (****) (****)
@njit
def comp_mfpt_by_time(rg_param, ry_param, switch_param_a, switch_param_b, v_param, N_LIST, D_LAYER, A_LAYER,
                      T_param, mass_checkpoint=10 ** 6, domain_radius=1.0, D=1.0, d_tube=0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    v_param *= -1

    d_list = struct_init.build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    MFPT = 0
    mass_retained = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k < K:

        net_current_out = 0

        net_current_out += num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        MFPT += net_current_out * k * dT ** 2
        # Implemented to provide occasional status checks/metrics during MFPT calculation
        if k > 0 and k % mass_checkpoint == 0:
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch,
                                      rg_param, ry_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch,
                                     A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        # transfer updated density info from the next step to the current
        # num.update_layer_inplace(diffusive_layer[0], diffusive_layer[1], rays, rings)
        # num.update_layer_inplace(advective_layer[0], advective_layer[1], rays, rings)
        k += 1
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return MFPT


# (****) (****)
@njit
def comp_mfpt_by_time_points_mass_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param,
                                      N_LIST, D_LAYER, A_LAYER, checkpoint_collect_container, MFPT_snapshots,
                                      mass_retention_threshold=0.01, mass_checkpoint=10**6, domain_radius=1.0, D=1.0, d_tube=0.0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    v_param *= -1

    d_list = struct_init.build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    MFPT = 0
    mass_retained = 0

    checkpoint_iter = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k == 0 or mass_retained > mass_retention_threshold:

        net_current_out = 0

        net_current_out += num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                                                    dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)
        if k > 0 and k % mass_checkpoint == 0:
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)

        MFPT += net_current_out * k * dT ** 2

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_mass_stamp = checkpoint_collect_container[checkpoint_iter]
            if curr_mass_stamp * 0.99 < mass_retained < curr_mass_stamp * 1.01:
                MFPT_snapshots[checkpoint_iter] = MFPT
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        # transfer updated density info from the next step to the current
        # num.update_layer_inplace(diffusive_layer[0], diffusive_layer[1], rays, rings)
        # num.update_layer_inplace(advective_layer[0], advective_layer[1], rays, rings)
        k += 1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# (****) (****)
@njit
def comp_mfpt_by_time_points_time_dep(rg_param, ry_param, switch_param_a, switch_param_b, v_param,
                                      N_LIST, D_LAYER, A_LAYER, checkpoint_collect_container, MFPT_snapshots,
                                      T_param, mass_checkpoint=10**6, domain_radius=1.0, D=1.0, d_tube=0.0):

    print("Running optimized version.")

    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    central_patch = num.compute_init_cond_cent(rg_param, domain_radius)
    v_param *= -1

    d_list = struct_init.build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube, domain_radius)

    MFPT = 0
    mass_retained = 0

    checkpoint_iter = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k < K:

        net_current_out = 0

        net_current_out += num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST,
                                                    dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)
        if k > 0 and k % mass_checkpoint == 0:
            print("Velocity (V)= ", v_param, "Time step: ", k, "Simulation time: ", k * dT, "Current mass: ",
                  mass_retained,
                  "a=", switch_param_a, "b=", switch_param_b)

        MFPT += net_current_out * k * dT ** 2

        if checkpoint_iter < len(checkpoint_collect_container):
            curr_stamp = np.floor(checkpoint_collect_container[checkpoint_iter] / dT)
            if k == curr_stamp:
                MFPT_snapshots[checkpoint_iter] = MFPT
                checkpoint_iter += 1
        else:
            return

        mass_retained = num.calc_mass(D_LAYER, A_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param,
                                      N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]
        # transfer updated density info from the next step to the current
        # num.update_layer_inplace(diffusive_layer[0], diffusive_layer[1], rays, rings)
        # num.update_layer_inplace(advective_layer[0], advective_layer[1], rays, rings)
        k += 1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
