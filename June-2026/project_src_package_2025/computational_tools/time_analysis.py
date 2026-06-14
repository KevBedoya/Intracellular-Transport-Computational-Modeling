from . import numerical_tools as num, njit, sys_config, np, supplements as sup, uni
from numba.typed import List
from time import perf_counter

ENABLE_JIT = True


def _launch_time_analysis_MFPT(RG, RY, V, W, T, N_List, dTUBE=0.0, SingleStep=False, MassCheckpoint=10 ** 6, iters=1,
                               unit_conversion=False, verbose=False):
    dR = 1 / RG
    dT = 2 * np.pi / RY
    dK = (0.1 * (min(dR ** 2, dT ** 2 * dR ** 2))) * 0.5

    if dTUBE < 0:
        dTUBE = sup.solve_d_rect(1, RG, RY, sup.j_max_bef_overlap(RY, N_List), 0)

    d_list = List()

    for m in range(RG):
        j_max = np.ceil((dTUBE / ((m + 1) * dR * dT)) - 0.5)
        keys = sup.mod_range_flat(N_List, j_max, RY, False)
        dict_ = sup.dict_gen(keys, N_List)
        d_list.append(dict_)

    diff_layer, adv_layer = sup.initialize_layers(RG, RY)

    CENTER = 1 / (np.pi * (dR ** 2))

    if SingleStep:
        K = 1
    else:
        K = np.floor(T / dK)

    # warm up the compiler
    # int_params = [0] * 3
    # float_params = [0.0] * 7
    print("Warming up JIT compiler")
    _time_analysis_MFPT_iter(RG, RY, 0, W, W, V, dK, dR, dT, CENTER, diff_layer, adv_layer, N_List,
                            d_list, dTUBE=dTUBE, MassCheckpoint=MassCheckpoint)

    time_list = []

    for i in range(iters):
        if verbose:
            print("Executing run (", i+1, ")")
        start = perf_counter()
        _time_analysis_MFPT_iter(RG, RY, K, W, W, V, dK, dR, dT, CENTER, diff_layer, adv_layer, N_List,
                                 d_list, dTUBE=dTUBE, MassCheckpoint=MassCheckpoint)
        end = perf_counter()
        time_list.append((end - start) / (RY * RG))
        # time_list.append((end - start))

    mean_time = np.mean(time_list)
    if unit_conversion:
        print("Mean time: ", uni.convert_seconds(mean_time))
    else:
        print("Mean time: ", mean_time)


@njit(nopython=ENABLE_JIT, fastmath=True, cache=True)
def _time_analysis_MFPT_iter(RG, RY, K, A, B, V, dK, dR, dT, CENTER, diff_layer, adv_layer, N_list, d_list, dTUBE=0.0,
                             MassCheckpoint=10 ** 6):
    # d_list = List()
    #
    # for m in range(RG):
    #     j_max = np.ceil((dTUBE / ((m + 1) * dR * dT)) - 0.5)
    #     keys = sup.mod_range_flat(N_list, j_max, RY, False)
    #     dict_ = sup.dict_gen(keys, N_list)
    #     d_list.append(dict_)

    MFPT = 0
    mass_retained = 0

    # **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    k = 0

    while k < K:

        net_current_out = 0
        m = 0

        # **************************************************************************************************
        while m < RG:

            # The advective angle index 'aIdx'
            aIdx = 0
            n = 0

            while n < RY:
                if m == RG - 1:
                    diff_layer[1][m][n] = 0
                else:

                    # Mixed configuration block 5/28/25
                    # **********************************************************************************************************************************************
                    if n in d_list[m]:

                        diff_layer[1][m][n] = num.u_density_rect(diff_layer, 0, m, n, dR, dT,
                                                                 dK,
                                                                 CENTER, RG, adv_layer,
                                                                 int(d_list[m][n]), A, B, dTUBE)

                    else:
                        diff_layer[1][m][n] = num.u_density(diff_layer, 0, m, n, dR, dT,
                                                            dK,
                                                            CENTER, RG, adv_layer,
                                                            aIdx,
                                                            A, B,
                                                            N_list)
                    if n == N_list[aIdx]:

                        adv_layer[1][m][n] = num.u_tube_rect(adv_layer, diff_layer, 0, m, n, A,
                                                             B, V, dK, dR, dT, dTUBE)
                        if aIdx < len(N_list) - 1:
                            aIdx += 1

                    if m == RG - 2:
                        net_current_out += num.j_r_r(diff_layer, 0, m, n, dR, 0) * RG * dR * dT
                n += 1
            m += 1
        # *****************************************************************************************************************

        MFPT += net_current_out * k * dK * dK

        k += 1
        # Implemented to provide occasional status checks/metrics during MFPT calculation
        if k > 0 and k % MassCheckpoint == 0:
            print("Velocity (V)= ", V, "Time step: ", k, "Simulation time: ", k * dK, "Current mass: ",
                  mass_retained,
                  "a=", A, "b=", B)

        mass_retained = num.calc_mass(diff_layer, adv_layer, 0, dR, dT, CENTER,
                                      RG, RY, N_list)
        CENTER = num.u_center(diff_layer, 0, dR, dT, dK, CENTER,
                              adv_layer, N_list, V)

        # transfer updated density info from the next step to the current
        diff_layer[0] = diff_layer[1]
        adv_layer[0] = adv_layer[1]
