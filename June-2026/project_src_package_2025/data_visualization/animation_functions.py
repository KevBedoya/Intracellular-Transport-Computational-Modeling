from . import plt, os, np, cm, Normalize, sup, exc
from matplotlib.patches import Polygon
import math


# (****) (****)
def produce_heatmap_tool_rect(HM_DL_snapshot, HM_C_snapshot, w_param, v_param, N_LIST_count, approach, pane,
                              filepath, MFPT, check_point, extraction_angle_list=None, boundary_of_extraction_list=None, display_extraction=True,
                              save_png=True, show_plt=True, transparent=False, toggle_border=False, color_scheme='viridis'):

    # Setting up the layers.

    # transparent = True

    diffusive_layer_center = np.full((1, HM_DL_snapshot.shape[1]), HM_C_snapshot)
    full_diffusive_layer = np.vstack([diffusive_layer_center, HM_DL_snapshot])
    rg_param, ry_param = full_diffusive_layer.shape

    # Constructing the discretized polar plane.

    r = np.linspace(0, 1, rg_param + 1)
    theta = np.linspace(0, 2 * np.pi, ry_param + 1)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    # Constructing the plot.
    plt.figure(figsize=(8, 10))
    # plt.figure(figsize=(10.24, 10.24), dpi=100)
    cmap = cm.get_cmap(color_scheme, 512)

    # Normalize density across the DL between the min and max density value.
    norm = Normalize(vmin=full_diffusive_layer.min(), vmax=full_diffusive_layer.max())

    # norm = Normalize(vmin=0, vmax=1)

    if toggle_border:
        heatmap = plt.pcolormesh(X, Y, full_diffusive_layer.T, shading='flat', cmap=cmap, norm=norm, edgecolors='k',
                                 linewidth=0.01)
    else:
        heatmap = plt.pcolormesh(X, Y, full_diffusive_layer.T, shading='flat', cmap=cmap, norm=norm)

    # Constructing the tick bar to display onto the plot. (This provides a numerical key for the colors depicting across the heatmap)

    cbar = plt.colorbar(heatmap, location='bottom', pad=0.08)
    ticks = np.linspace(full_diffusive_layer.min(), full_diffusive_layer.max(), num=8)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])
    cbar.ax.tick_params(labelsize=12, labelcolor='black')

    title = f'N={N_LIST_count}, w={w_param:.2e}, v={v_param}'
    if MFPT >= 0:
        title += f', MFPT={MFPT:.3f}'

    filename = ''

    if approach == 1:
        title += f',Total-Mass={check_point}'
        filename += f'static_HM_mass_dependence_pane={pane}.png'
    elif approach == 2:
        title += f', T={check_point}'
        filename += f'static_HM_time_dependence_pane={pane}.png'
    else:
        raise ValueError(f"Approach: {approach} is undefined, please input approach=1 (domain mass dependent) or approach=2 (time dependent)")
    plt.title(title, fontdict={'weight': 'bold', 'font': 'Times New Roman', 'size': 20}, pad=20)
    plt.axis('off')

    # Constructing the AL extraction region if it is toggled by the user.

    if display_extraction:
        if (boundary_of_extraction_list is not None and
                extraction_angle_list is not None and
                len(boundary_of_extraction_list) == rg_param - 1):

            ax = plt.gca()
            overlay_color = exc.extraction_overlay_colors.get(color_scheme, (0.5, 0.5, 0.5, 0.15))
            faint_color = (overlay_color[0], overlay_color[1], overlay_color[2], 0.08)

            for ring_idx in range(1, rg_param):
                angular_spread = boundary_of_extraction_list[ring_idx - 1]
                for angle_center in extraction_angle_list:
                    for offset in range(-angular_spread, angular_spread + 1):
                        angle_idx = (angle_center + offset) % ry_param
                        r1, r2 = r[ring_idx], r[ring_idx + 1]
                        t1, t2 = theta[angle_idx], theta[angle_idx + 1]
                        verts = [
                            (r1 * np.cos(t1), r1 * np.sin(t1)),
                            (r2 * np.cos(t1), r2 * np.sin(t1)),
                            (r2 * np.cos(t2), r2 * np.sin(t2)),
                            (r1 * np.cos(t2), r1 * np.sin(t2))
                        ]
                        color = faint_color if angular_spread == 0 else overlay_color
                        patch = Polygon(verts, closed=True, facecolor=color, edgecolor='black', linewidth=1.0)
                        ax.add_patch(patch)
        else:
            print("[Warning] Extraction overlay skipped due to boundary list mismatch or missing input.")

    if save_png:
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, filename)
            plt.savefig(file, bbox_inches='tight', transparent=transparent)
    if show_plt:
        plt.show()
    plt.close()

# (****) (****)
def display_domain_grid(rings, rays, microtubules, d_tube, r=1, display_extract=True, toggle_border=True):

    j_max_list = []

    j_max_lim = sup.j_max_bef_overlap(rays, microtubules)
    max_d_tube = sup.solve_d_rect(r, rings, rays, j_max_lim, 0)

    microtubules = list(np.unique(microtubules))

    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)

    for m in range(rings):
        j_max = math.ceil((d_tube / ((m + 1) * d_radius * d_theta)) - 0.5)
        j_max_list.append(j_max)

    return build_domain_grid(rings, rays, j_max_list, microtubules, display_extract, toggle_border, max_d_tube)

# (****) (****)
def build_domain_grid(rings, rays, boundary_of_extraction_list=None,
                      extraction_angle_list=None, display_extraction=False, toggle_border=True, d_sup=0):

    assert isinstance(rings, int) and rings > 0
    assert isinstance(rays, int) and rays > 0
    assert (not display_extraction) or (
        boundary_of_extraction_list is not None
        and extraction_angle_list is not None
        and len(boundary_of_extraction_list) == rings)

    # +1 for including center, +1 for boundary edge
    r = np.linspace(0, 1, rings + 2)
    theta = np.linspace(0, 2 * np.pi, rays + 1)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'})

    ax.set_title(f"Discretized Polar Grid Preview: {rings}x{rays} N={len(extraction_angle_list)} d_sup={d_sup:.5f}")

    # Base greyscale grid (used only as placeholder shading)
    C = np.zeros((theta.shape[0]-1, r.shape[0]-1))

    if toggle_border:
        pcm = ax.pcolormesh(X, Y, C, shading='flat', cmap='Greys', edgecolors='k', linewidth=0.1)
    else:
        pcm = ax.pcolormesh(X, Y, C, shading='flat', cmap='Greys', edgecolors='k', linewidth=0)

    center_radius = r[1]
    # center_circle = plt.Circle((0, 0), radius=(center_radius - 0.0004), color='white', zorder=10)
    center_circle = plt.Circle(
        (0, 0),
        center_radius,
        facecolor='white',
        edgecolor='black',  # Border color
        linewidth=1.2,  # Border thickness
        zorder=10  # Ensure it overlays everything
    )
    ax.add_patch(center_circle)

    ax.axis('off')

    # Red outer boundary edges (absorbing boundary)
    for i in range(rays):
        t1, t2 = theta[i], theta[i + 1]
        r1, r2 = r[-2], r[-1]
        verts = [
            (r1 * np.cos(t1), r1 * np.sin(t1)),
            (r2 * np.cos(t1), r2 * np.sin(t1)),
            (r2 * np.cos(t2), r2 * np.sin(t2)),
            (r1 * np.cos(t2), r1 * np.sin(t2))
        ]
        patch = Polygon(verts, closed=True, facecolor='none', edgecolor='red', linewidth=1.5)
        ax.add_patch(patch)

    # Extraction overlays (only until before absorbing ring)
    if display_extraction:
        overlay_color = (0.6, 0.2, 0.2, 0.3)
        for ring_idx in range(rings - 1):  # exclude last ring (absorbing)
            angular_spread = boundary_of_extraction_list[ring_idx]
            r1, r2 = r[ring_idx + 1], r[ring_idx + 2]
            for angle_center in extraction_angle_list:
                for offset in range(-angular_spread, angular_spread + 1):
                    angle_idx = (angle_center + offset) % rays
                    t1 = theta[angle_idx]
                    t2 = theta[(angle_idx + 1) % rays]
                    verts = [
                        (r1 * np.cos(t1), r1 * np.sin(t1)),
                        (r2 * np.cos(t1), r2 * np.sin(t1)),
                        (r2 * np.cos(t2), r2 * np.sin(t2)),
                        (r1 * np.cos(t2), r1 * np.sin(t2))
                    ]
                    patch = Polygon(verts, closed=True, facecolor=overlay_color,
                                    edgecolor='black', linewidth=0.8)
                    ax.add_patch(patch)

    # Microtubules (green radial lines at specified angular indices)
    for angle_idx in extraction_angle_list or []:
        # angle = 0.5 * (theta[angle_idx] + theta[(angle_idx + 1) % rays])  # center of conic region

        r_start = r[1]
        r_end = r[-2]  # last valid ring before absorbing boundary

        angle = 0.5 * (theta[angle_idx] + theta[(angle_idx + 1) % rays])  # center of conic region
        x1, y1 = r_start * np.cos(angle), r_start * np.sin(angle)

        # Get angular endpoints of the conic wedge
        t1 = theta[angle_idx]
        t2 = theta[(angle_idx + 1) % rays]

        # Compute the *polygonal* endpoints of the absorbing ring arc
        xA, yA = r_end * np.cos(t1), r_end * np.sin(t1)
        xB, yB = r_end * np.cos(t2), r_end * np.sin(t2)

        # Take the midpoint between those endpoints — this lies *on the polygonal arc*
        x2, y2 = 0.5 * (xA + xB), 0.5 * (yA + yB)

        ax.plot([x1, x2], [y1, y2], color='green', linewidth=3)

    plt.tight_layout()
    # plt.show()
    return fig
