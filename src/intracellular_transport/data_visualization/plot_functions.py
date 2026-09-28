from . import plt, os, pd, datetime, math
from intracellular_transport.computational_tools import mat_computations
import numpy as np

def plot_general(file_list, labels, xlab, ylab, title, filepath, xlog=False, ylog=False, ylims=None,
                 continuous=False, dynamic_pts=False, save_png=True, show_plt=True, transparent=False,
                 figsize=(12, 8), lab_fontsize=30, title_fontsize=40, legend_fontsize=22,
                 fontname='Times New Roman'):

    plt.figure(figsize=figsize)

    for i in range(len(file_list)):
        df = pd.read_csv(file_list[i])

        if continuous:
            plt.plot(df[xlab], df[ylab], label=labels[i])
        else:
            if dynamic_pts:
                plt.scatter(df[xlab], df[ylab], label=labels[i], linewidth=(10 / (i + 1)))
            else:
                    plt.scatter(df[xlab], df[ylab], label=labels[i], linewidth=(10 / (i + 1)))

        if xlog:
            plt.xscale('log')

        if ylog:
            plt.yscale('log')

    plt.xlabel(xlab, fontsize=lab_fontsize, fontname=fontname)
    plt.ylabel(ylab, fontsize=lab_fontsize, fontname=fontname)
    plt.title(title, fontsize=title_fontsize, fontname=fontname)

    plt.xticks(fontsize=30, fontname=fontname)
    plt.yticks(fontsize=25, fontname=fontname)

    plt.legend(fontsize=legend_fontsize, frameon=True, edgecolor='black', loc='best')

    if ylims is not None:
        if len(ylims) < 0 or len(ylims) > 2:
            raise ValueError("There can only be two values provided for the y-axis limits, "
                             "a lower bound and an upper bound.")
        else:
            plt.ylim(ylims[0], ylims[1])
    if save_png:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'{title}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight', transparent=transparent)
            print(f'Plot saved to {filepath}')

    if show_plt:
        plt.show()
    plt.close()


def plot_mfpt_v_checkpoints(data_filepath, x_label, rg_param, ry_param, w_param, v_param, N_LIST, file_path, save_png=True, show_plt=True):
    data = pd.read_csv(data_filepath)
    x = data[x_label]
    y = data['MFPT']
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel('MFPT')
    plt.grid(True)
    plt.tight_layout()
    plt.title(f'MFPT({x_label}) W={w_param:.2e}   V={v_param}   N={len(N_LIST)}    Domain={rg_param}x{ry_param}')

    if save_png:
        if file_path:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file = os.path.join(file_path, f"MFPT_v_{x_label}.png")
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {file_path}')
    if show_plt:
        plt.show()
    plt.close()

def _pi_frac_label_from_index(k: int, N: int, include_index=True):
    """
    Format theta = 2*pi*k/N as a reduced fraction of pi using mathtext.
    Returns something like r"$0$", r"$\pi$", r"$\frac{2}{3}\pi$", plus " (k)".
    """
    num = 2 * k
    den = N
    g = math.gcd(num, den)
    num //= g
    den //= g

    if num == 0:
        s = r"$0$"
    elif den == 1:
        # integer multiple of pi
        if num == 1:
            s = r"$\pi$"
        else:
            s = rf"${num}\pi$"
    else:
        # proper fraction
        s = rf"$\frac{{{num}}}{{{den}}}\pi$"

    if include_index:
        s = f"{s} ({k})"
    return s


def plot_bc_analysis(data_filepath, _LIST, v_param, checkpoint, T_fixed_ring_seg, file_path, save_png=True, show_plt=True, grid_analysis=False):
    data = pd.read_csv(data_filepath)
    x = _LIST
    y = data["BC_ratios"].values

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    label_expr = r"$\frac{\left|\frac{1}{r}\left(\frac{\partial \phi}{\partial \theta}\big|_{L} - \frac{\partial \phi}{\partial \theta}\big|_{R}\right)\right|}{\phi(r,\theta_i) - \rho(r)}$"

    ax.plot(x, y, linewidth=2, label=label_expr)
    ax.scatter(x, y, s=40, zorder=3)

    if grid_analysis:
        ax.set_xlabel(r"Grid-Size", fontsize=16)
    else:
        ax.set_xlabel(r"$W$", fontsize=16)

    ax.set_ylabel(
        label_expr,
        fontsize=14
    )

    if grid_analysis:
        ax.set_title(
            rf"\textbf{{Boundary condition (BC) ratio versus grid-size}}\\"
            rf"$V={v_param},\ T={checkpoint},\ r={T_fixed_ring_seg}$",
            fontsize=16
        )
    else:
        ax.set_title(
            rf"\textbf{{Boundary condition (BC) ratio versus $W$}}\\"
            rf"$V={v_param},\ T={checkpoint},\ r={T_fixed_ring_seg}$",
            fontsize=16
        )

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=12, frameon=True)

    plt.tight_layout()

    if save_png:
        if file_path:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file = os.path.join(file_path, "BC_analysis_plot.png")
            plt.savefig(file, bbox_inches='tight', dpi=300)
            print(f'Plot saved to {file}')

    if show_plt:
        plt.show()

def plot_phi_v_theta(data_filepath, v, w, N_LIST, approach, position, file_path, checkpoint_collect_container, save_png=True, show_plt=True, png_title="phi_v_theta.png"):

    data = pd.read_csv(data_filepath)

    # Extract x-axis from Radians column
    x = data["Radians"].values
    disc_pos = data["Disc-Pos"].values

    col_key = ""
    # Build legend labels from time or mass points
    label_container = []
    if approach == 1:
        for val in checkpoint_collect_container:
            label_container.append(f"Mass ~ {val}")
            col_key = "M="
    elif approach == 2:
        for val in checkpoint_collect_container:
            label_container.append(f"T ~ {val}")
            col_key = "T="
    else:
        raise ValueError(f"Invalid approach: {approach}. Use 1 (mass-point) or 2 (time-point).")

    # Identify only the snapshot columns (those starting with 'T_')
    snapshot_cols = [col for col in data.columns if col.startswith(col_key)]

    # Plot each time snapshot as a separate curve
    plt.figure(figsize=(10, 6))
    for col, label in zip(snapshot_cols, label_container):
        plt.plot(x, data[col].values, label=label)

    N = len(disc_pos)
    print(N)

    # Choose a readable cap on tick labels (tweak if you like)
    MAX_TICKS = 16

    # Step so we get at most ~MAX_TICKS ticks (always include 0 and ~pi if available)
    step = max(1, N // MAX_TICKS)

    # Base tick indices
    tick_idx = list(range(0, N, step))

    # Ensure we include 0 and the middle (pi) when N even
    if 0 not in tick_idx:
        tick_idx.insert(0, 0)
    mid = N // 2
    if N % 2 == 0 and mid not in tick_idx:
        tick_idx.append(mid)

    # Sort & unique
    tick_idx = sorted(set(tick_idx))

    # Map indices -> x positions and labels
    tick_positions = [x[i] for i in tick_idx ]
    tick_labels = [_pi_frac_label_from_index(i, N, include_index=True) for i in tick_idx]

    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

    # Labels and title
    plt.xlabel("Theta (radians, with Disc-Pos index)")
    # plt.yscale('log')
    plt.ylabel(r"$\phi$")
    # plt.ylim(0.3, 0.4)
    title = r"$\phi$" + f"  W={w:.2e}   V={v}   N={len(N_LIST)}  Position={position}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    if save_png:
        if file_path:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file = os.path.join(file_path, png_title)
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {file}')
    if show_plt:
        plt.show()
    plt.close()


def plot_dense_v_rad(y_lab, data_filepath, v, w, N, rings, rays, fixed_angle, checkpoint_collect_container, file_path, approach, save_png=True, show_plt=True):

    data = pd.read_csv(data_filepath)

    x = data['Radius']

    label_container = []

    col_key = ""

    if approach == 1:
        for i in range(len(checkpoint_collect_container)):
            label_container.append(f"Mass ~ {checkpoint_collect_container[i]}")
            col_key = "M="

    elif approach == 2:
        for i in range(len(checkpoint_collect_container)):
            label_container.append(f"T ~ {checkpoint_collect_container[i]}")
            col_key = "T="
    else:
        raise ValueError(f"Invalid approach: {approach}. Use approach 1 (mass-point collection) or approach 2 (time-point collection).")
    # Plot each row of data

    plt.figure(figsize=(10, 6))

    # Identify only the snapshot columns (those starting with 'T_')
    snapshot_cols = [col for col in data.columns if col.startswith(col_key)]

    # Plot each time snapshot as a separate curve
    plt.figure(figsize=(10, 6))
    for col, label in zip(snapshot_cols, label_container):
        plt.plot(x, data[col].values, label=label)

    # Add labels, legend, and title
    plt.xlabel("(r) Radius")

    title = ""

    if y_lab.lower() == "phi":
        print("Hello")
        plt.ylabel(r"$\phi$")
        title = r"$\phi$" + f"(r) W={w:.2e}  V={v}   N={N}  fixed_angle={fixed_angle} Domain={rings}x{rays}"
    elif y_lab.lower() == "rho":
        print("Hello")
        plt.ylabel(r"$\rho$")
        title = r"$\rho$" + f"(r) W={w:.2e}  V={v}   N={N}  fixed_angle={fixed_angle} Domain={rings}x{rays}"

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_png:
        if file_path:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            # file = os.path.join(file_path, f'{y_lab}_v_theta_V={v}_W={w}_N={N}_Angle={fixed_angle}_{current_time}.png')
            file = os.path.join(file_path, f'{y_lab.lower()}_v_radius_@DiscAng={fixed_angle}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {file_path}')

    if show_plt:
        plt.show()
    plt.close()


def plot_mass_analysis(data_filepath, v, a, b, N, T, rings, rays, mass_type, file_name_mass_type, file_path, save_png=True, show_plt=True):

    data = pd.read_csv(data_filepath)

    # Plot each row of data
    plt.figure(figsize=(10, 6))

    x = data['T']
    y = data['M']
    # discretization = len(data)
    # x = np.linspace(0, T, discretization)
    plt.plot(x, y)

    # Add labels, legend, and title
    plt.xlabel("(T) Time")
    plt.ylabel("(m) Mass")
    plt.yscale('log')

    # Equal switch rates keep reading as a single W, as every earlier figure did.
    rates = f"W={a:.2e}" if a == b else f"A={a:.2e}   B={b:.2e}"

    title = f" Mass ({mass_type}) v. Time   {rates}   V={v}   N={len(N)}  Domain={rings}x{rays}"

    plt.title(title)
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_png:
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if file_path:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            # file = os.path.join(file_path, f'{mass_type}_versus_T_W={w}_V={v}_N={len(N)}_{rings}x{rays}_{current_time}.png')

            filename_ = "MA_" + file_name_mass_type + ".png"
            file = os.path.join(file_path, filename_)
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {file_path}')
    if show_plt:
        plt.show()
    plt.close()


def plot_Jrr_sum_analysis(data_filepath, v, w, N, rings, rays, file_path, save_png=True, show_plt=True):

    data = pd.read_csv(data_filepath)

    plt.figure(figsize=(10, 6))

    x = data['T']
    y = data['Jrr_sum']

    plt.plot(x, y)

    plt.xlabel("(T) Time")
    plt.ylabel("(Jrr_sum) Mass")

    title = f"Jrr_sum v. Time  W={w:.2e}   V={v}   N={len(N)}  Domain={rings}x{rays}"

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_png:
        if file_path:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            filename_ = "Jrr_versus_t.png"
            file = os.path.join(file_path, filename_)
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {file_path}')
    if show_plt:
        plt.show()

def plot_normalized_separable_trajectories(
    D,
    v_param,
    w_param,
    T_param,
    N_param_count,
    central_patch_density,
    selected_rings=None,
    selected_rays=None,
    round_digits=2,
    include_central_patch=True
):
    """
    Plot normalized phi(theta) and phi(r) trajectories.

    Parameters
    ----------
    D : ndarray
        Diffusive layer matrix of shape
        (radial rings, angular rays).

    v_param : float
        Filament velocity.

    w_param : float
        Switching rate.

    T_param : float
        Dimensionless time.

    N_param_count : int
        Number of filaments.

    central_patch_density : float
        Density value at the central patch (r = 0).

    selected_rings : list

    selected_rays : list

    round_digits : int, optional
        Number of decimal digits for radius labels.

    include_central_patch : bool, optional
        If True, prepend the normalized central patch density
        to each radial trajectory. If False, plot only the
        radial rings contained in D.
    """

    m, n = D.shape

    ########################################################
    # Plot 1 : phi(theta)
    ########################################################

    N = D.T.copy()

    mat_computations.normalize_columns(N)

    theta = np.linspace(0, 2 * np.pi, n)

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for ring in selected_rings:

        ax1.plot(
            theta,
            N[:, ring],
            linewidth=1.5,
            label=rf"$r={ring}$"
        )

    ax1.set_xlabel(r"$\theta$", fontsize=14)
    ax1.set_ylabel(r"Normalized $\phi(\theta)$", fontsize=14)

    ax1.set_title(
        rf"Normalized Angular Trajectories "
        rf"($v={v_param}$, "
        rf"$w={w_param}$, "
        rf"$T={T_param}$, "
        rf"$N={N_param_count}$)",
        fontsize=14
    )

    ax1.set_xticks(
        [
            0,
            np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            2 * np.pi
        ]
    )

    ax1.set_xticklabels(
        [
            r"$0$",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
            r"$\frac{3\pi}{2}$",
            r"$2\pi$"
        ]
    )

    ax1.grid(True, alpha=0.3)

    ax1.legend(
        title="Fixed radial ring",
        fontsize=8,
        loc="best"
    )

    fig1.tight_layout()

    ########################################################
    # Plot 2 : phi(r)
    ########################################################

    R = D.copy()

    central_patch_normalized = (
        mat_computations.normalize_columns(
            R,
            return_central_vector=True,
            central_patch=central_patch_density
        )
    )

    dr = 1.0 / (m + 1)

    if include_central_patch:

        radius = np.concatenate(
            (
                [0.0],
                np.arange(1, m + 1) * dr
            )
        )

    else:

        radius = np.arange(1, m + 1) * dr

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for ray in selected_rays:

        # theta_value = (
        #     2 * np.pi * col /
        #     (R.shape[1] - 1)
        # )

        if include_central_patch:

            trajectory = np.concatenate(
                (
                    [central_patch_normalized[ray]],
                    R[:, ray]
                )
            )

        else:

            trajectory = R[:, ray]

        ax2.plot(
            radius,
            trajectory,
            linewidth=1.5,
            # label=rf"$\theta={theta_value / np.pi:.2f}\pi$"
            label=rf"$\theta_i={ray}$"
        )

    ax2.set_xlabel(r"$r$", fontsize=14)
    ax2.set_ylabel(r"Normalized $\phi(r)$", fontsize=14)

    ax2.set_title(
        rf"Normalized Radial Trajectories "
        rf"($v={v_param}$, "
        rf"$w={w_param}$, "
        rf"$T={T_param}$, "
        rf"$N={N_param_count}$)",
        fontsize=14
    )

    ax2.grid(True, alpha=0.3)

    ax2.legend(
        title="Fixed angular ray",
        fontsize=8,
        loc="best"
    )

    fig2.tight_layout()

    plt.show()

def plot_cosine_similarity_heatmap(
    C,
    title="Cosine Similarity Matrix",
    annotate=True,
    decimals=3,
    cmap="viridis",
    figsize=(8, 7)
):
    """
    Plot a gridded heatmap of a cosine similarity matrix.

    Parameters
    ----------
    C : ndarray
        Cosine similarity matrix with entries in [0, 1].

    title : str, optional
        Figure title.

    annotate : bool, optional
        If True, display numerical values in each cell.

    decimals : int, optional
        Number of displayed decimal places.

    cmap : str, optional
        Matplotlib colormap.

    figsize : tuple, optional
        Figure size.
    """



    C = np.asarray(C)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        C,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal"
    )

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=12)

    # major ticks
    ax.set_xticks(np.arange(C.shape[1]))
    ax.set_yticks(np.arange(C.shape[0]))

    ax.set_xlabel("Trajectory Index")
    ax.set_ylabel("Trajectory Index")

    ax.set_title(title)

    # grid lines
    ax.set_xticks(
        np.arange(-0.5, C.shape[1], 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, C.shape[0], 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.5
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # numerical annotations
    if annotate:

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                value = C[i, j]

                text_color = (
                    "white"
                    if value < 0.5
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{value:.3e}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7
                )

    fig.tight_layout()
    plt.show()

# ================================================= UNDER INSPECTION/REQUIRES UPDATES =================================================
# def plot_dense_v_rad_mul(y_lab, data_filepaths, v, w, N, rings, rays, fixed_angle, time_point_container, file_path,
#                      save_png=False, show_plt=True):
#     """
#     Plots the same indexed row (trajectory) across multiple CSV datasets for comparison.
#
#     Parameters:
#         y_lab (str): Label for y-axis.
#         data_filepaths (list): List of CSV file paths (same format expected).
#         v, w, N, rings, rays, fixed_angle: Parameters for title.
#         time_point_container (list): Time values corresponding to each row.
#         file_path (str): Directory to save PNG if enabled.
#         save_png (bool): If True, saves PNG to file_path.
#         show_plt (bool): If True, displays the plot.
#     """
#
#     # Load all datasets into a list of DataFrames
#     datasets = [pd.read_csv(fp, header=None) for fp in data_filepaths]
#
#     # Determine x-axis values
#     if y_lab.lower() == "phi":
#         x = np.linspace(0, 1, rings + 1)
#     elif y_lab.lower() == "rho":
#         x = np.linspace(1 / rings, 1, rings)
#     else:
#         raise ValueError("y_lab must be either 'phi' or 'rho'.")
#
#     converted_container = [f"T={T:.3f}" for T in time_point_container]
#
#     num_trajectories = len(converted_container)
#     num_datasets = len(datasets)
#
#     plt.figure(figsize=(12, 6 * num_trajectories))
#
#     for i in range(num_trajectories):
#         plt.subplot(num_trajectories, 1, i + 1)
#         for j, df in enumerate(datasets):
#             label = f"{converted_container[i]} - Dataset {j + 1}"
#             plt.plot(x, df.iloc[i], label=label)
#
#         plt.xlabel("(R) Radius")
#         plt.ylabel(y_lab)
#         plt.title(f"Trajectory {i + 1} at {converted_container[i]}")
#         plt.legend()
#         plt.grid(True)
#
#     full_title = f"{y_lab}_comparison_V={v}_W={w:.2e}_N={N}_Angle={fixed_angle}_Domain={rings}x{rays}"
#     plt.suptitle(full_title, fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#
#     if save_png:
#         current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         if file_path:
#             os.makedirs(file_path, exist_ok=True)
#             filename = f"{y_lab}_comparison_V={v}_W={w}_N={N}_Angle={fixed_angle}_{current_time}.png"
#             save_path = os.path.join(file_path, filename)
#             plt.savefig(save_path, bbox_inches='tight')
#             print(f'Plot saved to {save_path}')
#
#     if show_plt:
#         plt.show()
#     plt.close()
#
#
# def plot_phi_v_theta_mult(data_filepaths, v, w, N, approach, position, file_path, save_png=False, show_plt=True,
#                      time_point_container=None):
#     """
#     Plots corresponding phi-vs-theta rows across multiple CSV files.
#     Labeling is based on the approach used.
#
#     Parameters:
#         data_filepaths (list): List of file paths (all should have same row count).
#         v, w, N, approach, position: Plot parameters.
#         file_path (str): Directory to save figure if save_png=True.
#         save_png (bool): Save figure to file if True.
#         show_plt (bool): Show figure if True.
#         time_point_container (list): Used only if approach == 3.
#     """
#
#     # Load all datasets
#     datasets = [pd.read_csv(fp, header=None) for fp in data_filepaths]
#
#     # Assume all files have the same number of rows
#     num_curves = datasets[0].shape[0]
#     x = range(1, datasets[0].shape[1] + 1)  # theta values (1-based index)
#
#     # Determine label strategy based on approach
#     if approach == 2:
#         label_container = ["0.675 < m < 0.68", "0.45 < mass_retained < 0.46",
#                            "0.225 < mass_retained < 0.26", "0.015 < mass_retained < 0.02"]
#     elif approach == 1:
#         label_container = ["early time", "late time"]
#     elif approach == 3:
#         if time_point_container is None:
#             raise ValueError("time_point_container must be provided for approach == 3")
#         label_container = [f"T={T:.3f}" for T in time_point_container]
#     elif approach == 4:
#         label_container = [f"ring={i * 2}" for i in range(num_curves)]
#     else:
#         raise ValueError(f'{approach} is not a valid argument. Choose 1, 2, 3, or 4.')
#
#     # Plot each row (trajectory) as a subplot comparing all datasets
#     plt.figure(figsize=(12, 6 * num_curves))
#
#     for i in range(num_curves):
#         plt.subplot(num_curves, 1, i + 1)
#         for j, df in enumerate(datasets):
#             label = f"{label_container[i]} - Dataset {j + 1}"
#             plt.plot(x, df.iloc[i], label=label)
#
#         plt.xlabel("Theta")
#         plt.ylabel("Phi")
#         plt.title(f"Trajectory {i + 1}: {label_container[i]}")
#         plt.legend()
#         plt.grid(True)
#
#     # Overall title
#     if approach == 4:
#         suptitle = f'Phi_vs_Theta_V={v}_W={w}_N={N}_Approach{approach}'
#     else:
#         suptitle = f'Phi_vs_Theta_V={v}_W={w}_N={N}_Approach{approach}_Position={position}'
#
#     plt.suptitle(suptitle, fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#
#     if save_png:
#         current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         if file_path:
#             os.makedirs(file_path, exist_ok=True)
#             filename = f'phi_v_theta_v={v}_w={w}_app={approach}_pos={position}_{current_time}.png'
#             save_path = os.path.join(file_path, filename)
#             plt.savefig(save_path, bbox_inches='tight')
#             print(f'Plot saved to {save_path}')
#
#     if show_plt:
#         plt.show()
#     plt.close()