
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
            ("w_param", ""),
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
            ("show_plt", False)
        ]
    },
}

PARAMETER_HINTS = {
    "rg_param": "rg_param: Number of radial rings in domain (int)",
    "ry_param": "ry_param: Number of angular rays in domain (int)",
    "w_param": "w_param: Mutual switch-rate between DL and AL (float)",
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
    "heat_plot_border": "heat_plot_border: Toggle to display borders on static DL heatplot outputs. (boolean)",
    "heatplot_colorscheme": "heatplot_colorscheme: selection of available colorscheme for static DL heatplots. (str)",
    "display_extraction": "display_extraction: Toggle to display DL-to-AL extraction borders on static DL heatplots. (boolean)",
    "mass_checkpoint": "mass_checkpoint: # of time-steps per computational checkpoint (status-check). (int)",
    "mass_retention_threshold": "mass_retention_threshold: amount of mass until termination of method. (float)",
    "mass_threshold": "mass_threshold: amount of mass until termination of method. (float)",
    "checkpoint_collect_container": "checkpoint_collect_container: Provide as a list of floats: [], s.t each entry denotes a time-stamp (approach=2) OR a mass-stamp for data collection (approach=1).",
    "approach": "approach: (1) For mass dependent data collection, (2) For time dependent data collection. (int)"
}

