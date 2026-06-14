# Computational Workflow: MFPT Solver

This document outlines the execution flow of the `solve_mfpt()` routine, with explicit traceability of all called methods and their **scientific reliability status**.

---

## Top-Level Routine

### `solve_mfpt()` (in `launch_functions/launch.py`)
> Launches the main computation for the MFPT. This function is reliable for scientific computing.

**Upstream GUI Trigger**: `GUI Solve MFPT`

**Internal Calls:**
- `comp_mfpt_by_time` 
- `solve_d_rect`
- `initialize_layers` 

---

## Component Function Overview

###  `comp_mfpt_by_time` (in `computational_tools/mfpt_comp_functions.py`)  
> Returns the mean first passage time (MFPT) from time-evolved data.

- **Trust level**:  Reliable `(****)`
- **Inputs**: `diffusive_layer`, `advective_layer`, `d_radius`, `rings`
- **Returns**: `MFPT: float`

---

### `solve_d_rect` (in `computational_tools/supplements.py`)  
> Computes an upper bound on `d_tube` before overlap.

- **Trust level**: Reliable `(****)`
- **Calls**: `j_max_bef_overlap`

---

### `initialize_layers` (in `computational_tools/supplements.py`)
> Initializes the diffusive and advective domain layers.

- **Trust level**: Reliable `(****)`

---

##  Initialization Constants (all in `computational_tools/numerical_tools.py`)

| Function              | Purpose                      | Verified |
|-----------------------|------------------------------|----------|
| `compute_dRad`        | Initialize radial spacing    | `(****)` |
| `compute_dThe`        | Initialize angular spacing   | `(****)` |
| `compute_dT`          | Time-step initialization     | `(****)` |
| `compute_K`           | Initialize constants         | `(****)` |
| `compute_init_cond_cent` | Central patch initial condition | `(****)` |

---

## Iterative Update Functions

All reside in `computational_tools/numerical_tools.py` unless noted.

| Function           | Description | Verified |
|--------------------|--------|----------|
| `u_density_rect`   | Update `diffusive_layer` in extraction zone | `(****)` |
| `u_density`        | Update `diffusive_layer` outside extraction | `(****)` |
| `u_tube_rect`      | Update `advective_layer` using N_LIST | `(****)` |
| `j_r_r`            | Net current out | `(****)` |
| `calc_mass`        | Mass conservation check | `(****)` |
| `u_center`         | Central patch update | `(****)` |

---

## Structural Initialization

| File | Function | Purpose | Verified|
|------|----------|---------| --------|
| `computational_tools/struct_init.py` | `init_d_list()` | Generate dictionary list of extraction ranges, ring-based | `(****)`|
---
