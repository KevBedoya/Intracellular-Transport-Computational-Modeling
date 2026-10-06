
PARAMETER_SCHEMAS = {
    "Compute MFPT until mass %": {
            "required": [
                ("rg_param", ""),
                ("ry_param", ""),
                ("N_LIST", ""),
                ("v_param", ""),
                ("w_param", "")
            ],
            "default": [
                ("domain_radius", 1.0),
                ("D", 1.0),
                ("mass_checkpoint", int(1e6)),
                ("mass_retention_threshold", 1e-2),
                ("d_tube", 0)
            ],
            "approach": ["mass-dependent"]
        },
    "MFPT point collection (Mass % dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("N_LIST", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("checkpoint_collect_container", "")
        ],
        "default": [
            ("mass_retention_threshold", 1e-2),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["mass-dependent"]
    },
    "Phi angular dependence (collection: Mass % dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("N_LIST", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("mass_retention_threshold", 0.01),
            ("T_fixed_ring_seg", 0.5),
            ("d_tube", 0.0),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["mass-dependent"]
    },
    "Phi/Rho radial dependence (collection: Mass % dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("N_LIST", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("R_fixed_angle", -1),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("d_tube", 0.0),
            ("mass_retention_threshold", 1e-2),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["mass-dependent"]
    },
    "Static Heatplot Analysis (collection: Mass % dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("N_LIST", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("mass_retention_threshold", 1e-2),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("heatplot_border", False),
            ("heatplot_colorscheme", 'viridis'),
            ("save_png", True),
            ("show_plt", False),
            ("display_extraction", True)
        ],
        "approach": ["mass-dependent"]
    },

    "Full Analysis (time t dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("T_param", ""),
            ("N_LIST", "")
        ],
        "default": [
            ("d_tube", 0.0),
            ("Timestamp_List", None),
            ("MA_collection_factor", int(5)),
            ("MA_collection_factor_limit", int(1e3)),
            ("D", 1.0),
            ("domain_radius", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("T_fixed_ring_seg", 0.5),
            ("R_fixed_angle", int(-1)),
            ("save_png", True),
            ("show_plt", False),
            ("heat_plot_border", False),
            ("heatplot_colorscheme", 'viridis'),
            ("display_extraction", True)
        ],
        "approach": ["time-dependent"]
    },
    "Compute MFPT until time T": {
            "required": [
                ("rg_param", ""),
                ("ry_param", ""),
                ("N_LIST", ""),
                ("v_param", ""),
                ("w_param", ""),
                ("T_param", "")
            ],
            "default": [
                ("domain_radius", 1.0),
                ("D", 1.0),
                ("mass_checkpoint", int(1e6)),
                ("d_tube", 0.0)
            ],
            "approach": ["time-dependent"]
        },
    "MFPT point collection (time T dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("N_LIST", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("T_param", ""),
            ("checkpoint_collect_container", "")
        ],
        "default": [
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["time-dependent"]
    },
    "Phi angular dependence (collection: time T dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("T_fixed_ring_seg", 0.5),
            ("d_tube", 0.0),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["time-dependent"]
    },
    "Phi/Rho radial dependence (collection: time T dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("R_fixed_angle", -1),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("d_tube", 0.0),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["time-dependent"]
    },
    "Static Heatplot Analysis (collection: time T dep.)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("N_LIST", ""),
            ("T_param", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("heatplot_border", False),
            ("heatplot_colorscheme", 'viridis'),
            ("save_png", True),
            ("show_plt", False),
            ("display_extraction", True)
        ],
        "approach": ["time-dependent"]
    },

    "Time Until Mass Depletion": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("N_LIST", ""),
            ("v_param", ""),
            ("w_param", "")
        ],
        "default": [
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("d_tube", 0),
            ("mass_threshold", 0.01)
        ]
    },
    "Mass Analysis": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("a_param", ""),
            ("b_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
        ],
        "default": [
            ("MA_collection_factor", int(5)),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0),
            ("MA_collection_factor_limit", int(1e3)),
            ("save_png", True),
            ("show_plt", False),
            ("device", "cpu")
        ]
    },

    "Characteristic Time (mass vs v)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_LIST", ""),
            ("w_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
        ],
        "default": [
            ("MA_collection_factor", int(5)),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("center_init_cond", True),
            ("m_init", 0),
            ("n_init", 0),
            ("show_plt", False),
            ("device", "cpu"),
            ("tau", 1e-3)
        ],
        "approach": ["time-dependent"]
    },

    "Characteristic Time (a,b grid)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("a_list", ""),
            ("b_list", ""),
            ("v_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
        ],
        "default": [
            ("MA_collection_factor", int(5)),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("center_init_cond", True),
            ("m_init", 0),
            ("n_init", 0),
            ("show_plt", False),
            ("device", "cpu"),
            ("workers", 0),
            ("tau", 1e-3)
        ],
        "approach": ["time-dependent"]
    },

    "Jrr Mass Sum Over Time": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
        ],
        "default": [
            ("collection_factor", int(5)),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("d_tube", 0.0),
            ("collection_factor_limit", int(1e3)),
            ("save_png", True),
            ("show_plt", False),
            ("center_init_cond", True),
            ("m_init", 0),
            ("n_init", 0)
        ],
        "approach": ["time-dependent"]
    },

    "BC Parameter Dependence (w sweep)": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("T_param", ""),
            ("N_LIST", ""),
            ("w_LIST", ""),
            ("checkpoint", ""),
        ],
        "default": [
            ("T_fixed_ring_seg", 0.5),
            ("d_tube", 0.0),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False),
            ("center_init_cond", True),
            ("m_init", 0),
            ("n_init", 0)
        ],
        "approach": ["time-dependent"]
    },

    "BC Parameter Dependence (grid-size sweep)": {
        "required": [
            ("v_param", ""),
            ("T_param", ""),
            ("w_param", ""),
            ("N_amount", ""),
            ("checkpoint", ""),
            ("grid_list", ""),
        ],
        "default": [
            ("T_fixed_ring_seg", 0.5),
            ("d_tube", 0.0),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False)
        ],
        "approach": ["time-dependent"]
    },

    "Angular Trajectory Matrix": {
        "required": [
            ("rg_param", ""),
            ("ry_param", ""),
            ("v_param", ""),
            ("w_param", ""),
            ("N_LIST", ""),
            ("checkpoint_collect_container", ""),
        ],
        "default": [
            ("mass_retention_threshold", 0.01),
            ("d_tube", 0.0),
            ("domain_radius", 1.0),
            ("D", 1.0),
            ("mass_checkpoint", int(1e6)),
            ("save_png", True),
            ("show_plt", False),
            ("center_init_cond", True),
            ("m_init", 0),
            ("n_init", 0)
        ],
        "approach": ["mass-dependent"]
    },
}

PARAMETER_HINTS = {
    "rg_param": "rg_param: Number of radial rings in domain (int)",
    "ry_param": "ry_param: Number of angular rays in domain (int)",
    "w_param": "w_param: Mutual switch-rate between DL and AL (float)",
    "a_param": ("a_param: Switch rate onto the diffusive layer (a). Set equal "
                "to b_param for the mutual rate w. (float)"),
    "b_param": ("b_param: Switch rate onto the advective layer (b). Set equal "
                "to a_param for the mutual rate w. (float)"),
    "v_param": "v_param: Particle velocity across AL (float)",
    "T_param": "T_param: Solution time (dimensionless units) (float)",
    "N_LIST": "N_LIST: Angular index positions of microtubules in domain. Provide as list of ints: []",
    "d_tube": "d_tube: AL-to-DL extraction range relative to microtubule positioning. (float)",
    "D": "D: Diffusion coefficient. (float)",
    "domain_radius": "domain_radius: Radius of cellular domain (float)",
    "Timestamp_List": "Timestamp_List: Provide as a list of floats: [], s.t each entry denotes a time-stamp for data collection.",
    "MA_collection_factor": "MA_collection_factor: # of time-steps in between mass data collection points. (int)",
    "MA_collection_factor_limit": "MA_collection_factor_limit: Limit on MA_collection_factor. (int)",
    "T_fixed_ring_seg": "T_fixed_ring_seg: Fixed radial ring position to collect angular dependent info from on DL. (int)",
    "R_fixed_angle": "R_fixed_angle: Fixed angular ray position to collect radially dependent info from on DL and AL. (int)",
    "save_png": "save_png: Toggle to save png outputs. (boolean)",
    "save_csv": "save_csv: Toggle to save csv outputs. (boolean)",
    "show_plt": "show_plt: Toggle to display plots on a separate wiindow (off of UI). (boolean)",
    "device": ("device: Where the time-stepping loop runs. 'cpu' (default, the "
               "reference implementation), 'gpu' (~9x faster at 96x96 rising to "
               "~16x at 160x160; fails if unsupported), or 'auto' (prefer gpu, "
               "fall back to cpu). GPU results agree with CPU to ~1e-14 rather "
               "than exactly, and gpu requires d_tube = 0 and a centred initial "
               "condition. (str)"),
    "heat_plot_border": "heat_plot_border: Toggle to display borders on static DL heatplot outputs. (boolean)",
    "heatplot_colorscheme": "heatplot_colorscheme: selection of available colorscheme for static DL heatplots. (str)",
    "display_extraction": "display_extraction: Toggle to display DL-to-AL extraction borders on static DL heatplots. (boolean)",
    "mass_checkpoint": "mass_checkpoint: # of time-steps per computational checkpoint (status-check). (int)",
    "mass_retention_threshold": "mass_retention_threshold: amount of mass until termination of method. (float)",
    "mass_threshold": "mass_threshold: amount of mass until termination of method. (float)",
    "checkpoint_collect_container": "checkpoint_collect_container: Provide as a list of floats: [], s.t each entry denotes a time-stamp (approach=2) OR a mass-stamp for data collection (approach=1).",
    "approach": "approach: (1) For mass dependent data collection, (2) For time dependent data collection. (int)",
    "v_LIST": "v_LIST: Advective velocities to sweep. Provide as a list of floats: []",
    "w_LIST": "w_LIST: Switch rates (a=b) to sweep. Provide as a list of floats: []",
    "a_list": ("a_list: Switch rates onto the diffusive layer (a) to sweep. "
               "Every pair in a_list x b_list is solved. Provide as a list of "
               "floats: []"),
    "b_list": ("b_list: Switch rates onto the advective layer (b) to sweep. "
               "Every pair in a_list x b_list is solved. Provide as a list of "
               "floats: []"),
    "workers": ("workers: How many (a, b) points to solve concurrently on the "
                "CPU. 0 (default) uses one per core, less one. Ignored on the "
                "GPU, where a single solve already fills the card. (int)"),
    "tau": ("tau: Steady-decay threshold for t*. t* is the time from which "
            "|y''| of y = ln(total mass) stays at or below tau to the end of "
            "the run. Smaller is stricter and gives a later t*. "
            "Default 1e-3. (float)"),
    "grid_list": "grid_list: Square grid sizes to sweep. Provide as a list of ints: []",
    "N_amount": "N_amount: Number of evenly spaced microtubules. (int)",
    "checkpoint": "checkpoint: Time- or mass-stamp at which data is collected. (float)",
    "collection_factor": "collection_factor: # of time-steps in between data collection points. (int)",
    "collection_factor_limit": "collection_factor_limit: Limit on collection_factor. (int)",
    "center_init_cond": "center_init_cond: Seed unit mass in the central patch. Set False to seed at (m_init, n_init). (boolean)",
    "m_init": "m_init: Ring index for an off-centred initial condition, in [0, rg_param-1]. (int)",
    "n_init": "n_init: Ray index for an off-centred initial condition, in [0, ry_param-1]. (int)"
}

