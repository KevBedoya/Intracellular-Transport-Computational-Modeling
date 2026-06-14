from . import njit, numerical_tools as num, np
from computational_tools import struct_init


@njit
def super_comp_type_I(rg_param, ry_param, switch_param_a, switch_param_b, T_param,
                      v_param, N_LIST, D_LAYER, A_LAYER,
                      Timestamp_LIST,
                      HM_DL_snapshots, HM_C_snapshots,
                      PvT_DL_snapshots, T_fixed_ring_seg,
                      MA_DL_timeseries, MA_AL_timeseries, MA_ALoI_timeseries, MA_ALoT_timeseries, MA_TM_timeseries, MA_collection_factor, relative_k,
                      PvR_DL_snapshots, RvR_AL_snapshots, R_fixed_angle,
                      MFPT_snapshots,
                      d_tube=0, D=1.0, domain_radius=1.0,
                      mass_checkpoint=10**6):

    """
    Super computation routine type I

    Args:
        rg_param (int): Number of rings for discretized domain
        ry_param (int): Number of rays for discretized domain
        switch_param_a (float): Switch rate from diffusive to advective layer
        switch_param_b (float): Switch rate from advective to diffusive layer
        T_param (float): Dimensionless duration until termination
        v_param (float): Velocity of particles on microtubules
        N_LIST (List[int]): 1-d list for angular index positions over discretized domain denoting microtubule positions
        D_LAYER (numpy.3darray[float64]): 3-d array for diffusive layer (3rd dim for 2-step approach in PDE numerical method)
        A_LAYER (numpy.3darray[float64]): 3-d array for advective layer (3rd dim for 2-step approach in PDE numerical method)
        Timestamp_LIST (List[float]): 1-d list for dimensionless timestamps for data collection (see options in parmas below)
        HM_DL_snapshots: Heatmap diffusive layer snapshot container
        HM_C_snapshots: Heatmap central patch mass snapshot container
        PvT_DL_snapshots: Phi v. Theta diffusive layer snapshot container (for phi as a function of theta at a fixed ring)
        T_fixed_ring_seg (float): Fixed ring fraction between (0,1) to collect for PvT_DL_snapshots
        MA_DL_timeseries: Mass analysis diffusive layer timeseries container (for mass as a function of time)
        MA_AL_timeseries: Mass analysis advective layer timeseries container (for mass as a function of time)
        MA_ALoI_timeseries: Mass analysis advective layer over initial mass timeseries container (for mass as a function of time)
        MA_ALoT_timeseries: Mass analysis advective layer over running total mass timeseries container (for mass as a function of time)
        MA_TM_timeseries: Mass analysis running total timeseries container (for mass as a function of time)
        MA_collection_factor (int) : Collecting data according to the ratio:   1 collection : MA_collection_factor [1, 10^3]
        relative_k (int) :
        PvR_DL_snapshots: Phi v. Radius diffusive layer snapshot container (for phi as a function of radius at a fixed angle)
        RvR_AL_snapshots: Rho v. Radius advective layer snapshot container (for rho as a function of radius at a fixed angle)
        R_fixed_angle (int): Fixed angle index to collect for PvR or RvR snapshots
        MFPT_snapshots: Snapshots of MFPT at specified timestamps
        d_tube (float): Diffusive layer microtubule collection range (only supports non-overlapping boundaries)
        D (float): Diffusion coefficient from analytic form of PDE
        domain_radius (float): Radius of the discretized domain
        mass_checkpoint(int): A repeating timestep checkpoint to print a computational progress report.
        MA_collection_factor_limit (int) : adjustable limit on the MA_collection_factor

    **All data containers (for snapshots or timeseries data) are passed by reference in this function.**

    Data containers are also processed for analysis in a separate file.

    """

    print("Running optimized version")

    # Compute constants
    dRad = num.compute_dRad(rg_param, domain_radius)
    dThe = num.compute_dThe(ry_param)
    dT = num.compute_dT(rg_param, ry_param, domain_radius, D)
    K = num.compute_K(rg_param, ry_param, T_param, domain_radius, D)
    v_param *= -1

    d_list = struct_init.build_d_tube_mapping_no_overlap(rg_param, ry_param, N_LIST, d_tube, domain_radius)
    print(d_list)

    # initialize variables

    dl_mass = 1
    al_mass = 0
    MFPT = 0
    central_patch = num.compute_init_cond_cent(rg_param, domain_radius)

    # initialize collection width values
    # relative_k = np.floor(K / MA_collection_factor)
    MA_k_step = 0

    # initialize timestep counter k
    k = 0

    timestamp = 0

    while k < K:

        # Numerically solve the PDE for a time step k
        # ************ -------------------------------------------------------------------------- ************
        net_current_escape = 0
        net_current_escape += num.comp_DL_AL_kp1_2step(ry_param, rg_param, d_list, D_LAYER, central_patch, A_LAYER, N_LIST, dRad, dThe, dT, switch_param_a, switch_param_b, v_param, d_tube)

        if k > 0 and k % mass_checkpoint == 0:
            print("Current timestep: ", k, "Current simulation time: ", k * dT, "Current DL mass: ", dl_mass, "Current AL mass: ", al_mass)
            print("Velocity (v): ", v_param, "Diffusive to Advective switch rate (a): ", switch_param_a,
                  "Advective to Diffusive switch rate (b): ", switch_param_b)

        # ------------%%%%%%%%%%%% collect results and update parameters for the next time-step %%%%%%%%%%%%------------
        # Compute MFPT at the current step
        MFPT += net_current_escape * k * dT ** 2

        # Collection mass

        if MA_k_step < relative_k and k % MA_collection_factor == 0:
            MA_DL_timeseries[MA_k_step] = dl_mass
            MA_AL_timeseries[MA_k_step] = al_mass
            MA_TM_timeseries[MA_k_step] = dl_mass + al_mass
            MA_ALoT_timeseries[MA_k_step] = al_mass / MA_TM_timeseries[MA_k_step]
            MA_ALoI_timeseries[MA_k_step] = al_mass / D
            MA_k_step += 1

        # Collect snapshots

        if timestamp < len(Timestamp_LIST):

            curr_stamp = np.floor(Timestamp_LIST[timestamp] / dT)
            if k == curr_stamp:

                # MFPT snapshot collection
                MFPT_snapshots[timestamp] = MFPT

                # Heatmap snapshot collection
                HM_DL_snapshots[timestamp] = D_LAYER[0]
                HM_C_snapshots[timestamp] = central_patch

                # Phi v. Theta snapshot collection
                PvT_DL_snapshots[timestamp] = D_LAYER[0][int(np.floor(rg_param * T_fixed_ring_seg))]

                # Density V. Radius snapshot collection
                PvR_DL_snapshots[timestamp][0] = central_patch
                for m in range(rg_param):
                    PvR_DL_snapshots[timestamp][m+1] = D_LAYER[0][m][R_fixed_angle]
                    RvR_AL_snapshots[timestamp][m] = A_LAYER[0][m][R_fixed_angle]

                timestamp += 1

        # Update mass for the next step
        dl_mass = num.calc_mass_diff(D_LAYER, 0, dRad, dThe, central_patch, rg_param, ry_param)
        al_mass = num.calc_mass_adv(A_LAYER, 0, dRad, dThe, rg_param, N_LIST)
        central_patch = num.u_center(D_LAYER, 0, dRad, dThe, dT, central_patch, A_LAYER, N_LIST, v_param)

        # Prepare for next timestep
        # num.update_layer_inplace(D_LAYER[0], D_LAYER[1], ry_param, rg_param)
        # num.update_layer_inplace(A_LAYER[0], A_LAYER[1], ry_param, rg_param)

        D_LAYER[0] = D_LAYER[1]
        A_LAYER[0] = A_LAYER[1]

        k += 1
    # ------------%%%%%%%%%%%% collect results and update parameters for the next time-step %%%%%%%%%%%%------------
        # m = 0
        # while m < rg_param:
        #     # Current angular index in N_LIST
        #     aIdx = 0
        #     n = 0
        #     while n < ry_param:
        #         if m == rg_param - 1:
        #             D_LAYER[1][m][n] = 0
        #         else:
        #             if n in d_list[m]:
        #                 D_LAYER[1][m][n] = num.u_density_rect(
        #                     D_LAYER, 0, m, n, dRad, dThe, dT,
        #                     central_patch, rg_param, A_LAYER, int(d_list[m][n]),
        #                     switch_param_a, switch_param_b, d_tube
        #                 )
        #             else:
        #                 D_LAYER[1][m][n] = num.u_density(
        #                     D_LAYER, 0, m, n, dRad, dThe, dT,
        #                     central_patch, rg_param, A_LAYER,
        #                     aIdx, switch_param_a, switch_param_b,
        #                     N_LIST)
        #
        #             # update the advective layer
        #             if n == N_LIST[aIdx]:
        #                 A_LAYER[1][m][n] = num.u_tube_rect(
        #                     A_LAYER, D_LAYER, 0, m, n,
        #                     switch_param_a, switch_param_b, v_param, dT, dRad, dThe, d_tube)
        #                 if aIdx < len(N_LIST) - 1:
        #                     aIdx += 1
        #
        #             if m == rg_param - 2:
        #                 net_current_escape += num.j_r_r(D_LAYER, 0, m, n, dRad, 0) * rg_param * dRad * dThe
        #         n += 1
        #     m += 1





