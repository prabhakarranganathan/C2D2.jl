# Input file reference

A C2D2.jl run is configured by one TOML file. This page lists every key the
two loaders read, with its meaning, its unit and its default. Keys you leave
out take the default. Keys the loaders do not read are ignored without a
warning (the reference inputs carry a few such keys from the MATLAB version of
the code; they are listed at the end).

The two loaders are

- `load_styext_context` in `julia/models/front_end/StyextRunContext.jl`, for
  the steady solvers (steady extension, and planar mixed flow / steady shear);
- `load_usyext_context` in `julia/models/front_end/UsyextRunContext.jl`, for
  the transient solver (start-up of extension, capillary thinning).

Commented examples of the three flow families are in `julia/examples/inputs/`.
The physics behind each key is in the technical report
([10.5281/zenodo.18012676](https://doi.org/10.5281/zenodo.18012676), which
always resolves to the latest version). Section numbers below refer to
version 3 of the report
([10.5281/zenodo.24035542](https://doi.org/10.5281/zenodo.24035542)).

## Units and notation

All inputs are dimensionless. The length unit is the equilibrium coil size
E₀, the time unit is λ₁,₀ (the longest relaxation time of a chain at rest, at
the given concentration), and the stress unit is c k_BT (chain number density
times thermal energy). Symbols:

| symbol | meaning |
|---|---|
| N_K | number of Kuhn segments in the chain |
| N_m | number of modes (springs) in the bead–spring chain |
| h*_K | hydrodynamic-interaction (HI) parameter of a Kuhn segment |
| φ | overlap ratio c/c* = C_φ c E₀³ (φ = 0 dilute limit, φ = 1 the overlap concentration) |
| E | chain stretch, end-to-end distance over E₀ (E = 1 at rest) |
| λ₁,₀, λ_Z | equilibrium longest relaxation time at the given φ, and its dilute (Zimm) value |
| Wi | Weissenberg number, λ₁,₀ times the strain rate |
| De | Deborah number, λ₁,₀ over the macroscopic flow time (for capillary thinning, the viscocapillary or Rayleigh time of the initial filament) |
| Oh | Ohnesorge number η_s/√(ρ R₀ γ) of the solvent |

## Sections read by both loaders

### `[polymer]`

| key | meaning | default |
|---|---|---|
| `NK` | N_K, number of Kuhn segments (sets the finite extensibility, E_max = √N_K) | `5000` |
| `Nm` | N_m, number of modes | `1` |
| `hK_star` | h*_K, HI parameter of a Kuhn segment; must lie in (0, 0.5). Below about 0.18 the chain is partially draining (Sec. 2.1.2) | `0.25` |
| `Q0_sq` | Q₀², equilibrium mean-square spring length in the code's units; keep `1.0` | `1.0` |
| `kBT` | k_BT in the code's units; keep `1.0` | `1.0` |
| `lambda0` | λ₁,₀ in the code's units; keep `1.0` (the key name uses the older symbol λ₀) | `1.0` |

### `[concentration]`

| key | meaning | default |
|---|---|---|
| `phi` | φ = c/c*, the overlap ratio (≥ 0) | `1.0` |
| `phi_values` | list; its first entry is used when `phi` is absent | – |

Both loaders accept a `phi` keyword argument that overrides the file (for
sweeps). `load_usyext_context` also accepts the older spelling `phi_kw`;
giving both with different values is an error.

### `[model]`

| key | meaning | allowed values | default |
|---|---|---|---|
| `fe_model` | spring law (Sec. 2.3): `"hookean"` no finite extensibility; `"fenepme"` one Peterlin factor from the chain end-to-end distance; `"fenepm"` one factor from the mean modal trace; `"fenep"` a factor per mode. The transient solver does not accept `"hookean"` | as listed | `"fenepme"` |
| `drag_model` | `"constant"` Zimm drag independent of stretch and concentration; `"c1d2"` stretch-dependent only (dilute); `"c2d2"` the full conformation- and concentration-dependent drag (Sec. 2.1) | as listed | `"c2d2"` |
| `draining_model` | `"partialdraining"` uses the Kirkwood–Riseman draining function α(h*_K, N); `"nondraining"` sets α = α_∞ everywhere (Sec. 2.1.2) | as listed | `"partialdraining"` |
| `spectrum_model` | relaxation spectrum (Sec. 2.2): `"auto"` picks `"rouse-full"` for constant drag and `"blob"` for C2D2 drag; `"blob"` the C2D2 blob spectrum; `"rouse-full"` (also `"rouse-analytical"`) exact Rouse eigenvalues; `"rouse-scaling"` p⁻²; `"zimm-scaling"` p⁻³ᐟ²; `"zimm-thurston"` (steady solvers only) | as listed | `"auto"` |
| `drag_interp` | how the weakly and strongly screened drags are joined, C2D2 drag only (Secs. 2.1.5–2.1.8): `"RP"` (= `"RPI"`) the original interpolation; `"BAREI"`, `"BARECUI"` the bare-blob model with a switch at E = φ or a Churchill–Usagi blend; `"OESI"`, `"OESCUI"` the shielded-blob model (overlap-enhanced shielding), switched or blended. `"OESCUI"` is the form used in our current work. Archived, for reproducing old runs only: `"CU"`, `"RPD"`, `"RPDW"`, `"RPDP"` | as listed, case-insensitive | `"RP"` |
| `p_phi` | Churchill–Usagi blending exponent of the `…CUI` schemes (and of `"CU"`); > 0 | number | `2.0` (`1.86` for `"CU"`) |
| `q_oes` | stretch-delayed shielding exponent of `"OESI"`/`"OESCUI"`: the shielding factor is ψ = min(φ/E^q, 1), so q > 0 delays the onset of shielding as the chain stretches. Must lie in [0, 1]. Ignored (with a warning) for other schemes | number | `0.0` |
| `C_phi` | overlap constant C_φ in φ = C_φ c E₀³ (Sec. 2.1.3); divides the stress prefactor, all drag models. 1 is the historical value; 0.39 for polystyrene in diethyl phthalate | > 0 | `1.0` |
| `drag_lut_n_points` | number of points of the look-up table in E that the solver uses for the drag (steady solvers only; the transient solver uses 400). 400 limits the accuracy of the C2D2 drag to about 1e-5 near E = 1 | integer ≥ 4 | `4000` |
| `De_mode` | legacy: `"cu_fit"` replaces the drag model's own λ₁,₀/λ_Z by the Martin-type correlation (1 + φ^p)^(1/p), p = `p_phi`, wherever λ₁,₀/λ_Z enters (De, the capillary-thinning prefactor U₁, η_s). Accepted with any `drag_interp`; intended for `"CU"`. Not recommended for new work | `"input"`, `"cu_fit"` | `"input"` |
| `g0_1D`, `s_form` | legacy relaxation-time factor g(s) of the `"CU"` scheme only; `g0_1D = 1` switches it off | number; `"sum"`, `"sum_hard"` | `1.0`; `"sum"` |

## Steady solvers (`load_styext_context`)

### `[run]`

| key | meaning | default |
|---|---|---|
| `label` | run name, used by the drivers to name the output folder | `"styext"` |

### `[flow]`

| key | meaning | default |
|---|---|---|
| `kind` | `"styext"` shear-free extension with principal rates `alphas`; `"pmf"` planar mixed flow (Sec. 3.2.2), which gives simple shear at `r = 1`. Aliases: `"ext"`, `"extension"`; `"shear"`, `"mixed"` | `"styext"` |
| `alphas` | the three principal strain-rate ratios of a shear-free flow, κ = Wi·diag(α); uniaxial extension `[-0.5, -0.5, 1.0]`, planar `[1.0, -1.0, 0.0]`, biaxial `[1.0, 1.0, -2.0]`. The largest entry is the stretching axis. Ignored for `"pmf"` | `[-0.5, -0.5, 1.0]` |
| `r` | vorticity-to-strain-rate ratio of the planar mixed flow (`"pmf"` only): 0 planar extension, 1 simple shear, > 1 rotation-dominated | `1.0` |
| `Wi_convention` | `"pmf"` only: `"strain"` means every Wi in this file is λ₁,₀ times the principal strain rate (the report's convention); `"shear"` (simple shear only) means they are λ₁,₀ γ̇, the usual shear Weissenberg number, which is twice the former | `"strain"` |

### `[limits]`

| key | meaning | default |
|---|---|---|
| `Wi_max` | stop the continuation once Wi exceeds this (`"Inf"` for no limit). In the `"shear"` convention it is read as λ₁,₀ γ̇ | `3.0` |
| `max_frac_stretch_sq` | stop once the chain stretch E² reaches this fraction of its maximum N_K E₀² (and, for FENE-PM/P, once the spring closure reaches the same fraction of its pole); in (0, 1) | `0.999` |
| `max_steps` | used only if `[continuation] max_steps` is absent | `5000` |

### `[sweep]` (optional; the fixed-Wi sweep used for steady shear)

Give either an explicit list or a logarithmic grid. The values follow
`Wi_convention`.

| key | meaning | default |
|---|---|---|
| `Wi_values` | list of Wi values to solve at | – |
| `Wi_min`, `Wi_max` | ends of a log-spaced grid | – |
| `n_per_decade` | points per decade of the grid | `10` |

### `[continuation]` (pseudo-arclength continuation, Sec. 3.3)

| key | meaning | default |
|---|---|---|
| `ds0` | initial arclength step | `1e-3` |
| `ds_min` | smallest step before the run stops | `1e-8` |
| `ds_max` | largest step | `1e3` |
| `max_steps` | maximum number of points on the curve | `5000` |

### `[newton]` (Newton iteration with Armijo line search, Sec. 3.3.3)

| key | meaning | default |
|---|---|---|
| `tol_res` | residual tolerance for convergence | `1e-8` |
| `max_newton` | maximum Newton iterations per point | `15` |
| `min_denom_tol` | smallest allowed denominator in the modal tensors (0 = off) | `0.0` |
| `tol_J_cond` | reciprocal condition number below which the Jacobian is treated as singular | `1e-12` |
| `use_armijo` | enable the line search | `true` |
| `armijo_c1`, `armijo_beta`, `max_bt` | line-search sufficient-decrease constant, backtracking factor and maximum backtracks | `1e-4`, `0.5`, `30` |
| `target_newton` | Newton iterations per point the step-size control aims for | `10` |
| `target_offmanifold_ratio` | predictor–corrector distance ratio the step-size control aims for | `0.3` |

### `[newton_fixed_Y]` (fallback solver at fixed stretch, Sec. 3.3.4)

| key | meaning | default |
|---|---|---|
| `max_newton_fixed_Y` | maximum Newton iterations of the fallback | `[newton] max_newton` |
| `tol_FY_rel_hard` | relative residual above which the run stops | `100.0` |
| `tol_FY_rel_soft` | relative residual above which a point is marked as not converged | `1e-2` |

## Transient solver (`load_usyext_context`)

The transient loader does not read `[run]`; the example scripts choose where
the output goes.

### `[flow]`

| key | meaning | default |
|---|---|---|
| `kind` | `"caber"` capillary thinning, viscocapillary limit Oh ≫ 1 (Sec. 4.5.1); `"caber_inertial"` capillary thinning, inertiocapillary limit Oh ≪ 1; `"constant"` start-up of extension at fixed Wi | `"caber"` |
| `alphas` | principal strain-rate ratios, as for the steady solver | `[-0.5, -0.5, 1.0]` |
| `De` | Deborah number. For `"caber"` De = λ₁,₀/t_v with t_v = η_s R₀/γ the viscocapillary time; for `"caber_inertial"` De is the Rayleigh-time Deborah number De_R = λ₁,₀/t_R, t_R = √(ρ R₀³/γ); for `"constant"` it only converts strain to time in the output | `1.0` |
| `De_values` | list; its first entry is used when `De` is absent | – |
| `Wi` | the constant Weissenberg number (`"constant"` only) | `1.0` |
| `Oh` | Ohnesorge number (`"caber_inertial"` only). It is the second input of the inertiocapillary balance; the polymer-stress prefactor U_R = Oh U₁/De_R is derived from it. 0 is allowed (no polymer arrest) | `0.005` |
| `U_R` | legacy override for `"caber_inertial"`: if given, it is converted to the equivalent Oh = U_R De_R/U₁ for this De_R. Give `Oh` instead | – |
| `X_R` | Newtonian end-plate factor of the inertiocapillary balance | `0.5115` |
| `Xe` | elastic-limit end-plate factor (`"caber_inertial"`; the viscous path uses the fixed values X_v = 0.7127, X_e = 1.5) | `1.5` |
| `De_Z` | with `De_mode = "cu_fit"` only: De = De_Z (1 + φ^p)^(1/p) replaces `De` | – |

### `[limits]`

| key | meaning | default |
|---|---|---|
| `max_frac_stretch_sq` | the run stops (status `terminated`) once E² reaches this fraction of its maximum | `0.999` |

### `[integration]` (adaptive exponential time-differencing, Sec. 4.1.5)

| key | meaning | default |
|---|---|---|
| `independent` | integration variable: `"strain"` Hencky strain, `"time"` time in units of λ₁,₀, `"auto"` strain while Wi is large and time while it is small | `"strain"` |
| `strain_max` | final Hencky strain | `6.0` |
| `Wi_min_strain_independent` | Wi below which `"auto"` steps in time (with a factor-2 hysteresis band) | `0.0` |
| `dstrain_init`, `dt_init` | first trial step in strain and in time | `1e-3`, `1e-3` |
| `rtol`, `atol` | relative and absolute tolerance of the embedded error estimate | `1e-7`, `1e-9` |
| `safety`, `grow_max`, `shrink_min` | step-size controller: safety factor, largest growth and smallest shrink per step | `0.9`, `2.0`, `0.2` |
| `dstrain_min`, `dstrain_max` | bounds on the strain step | `1e-8`, `5e-2` |
| `dt_min`, `dt_max` | bounds on the time step | `1e-10`, `1.0` |
| `scale_floor`, `epsdot_floor`, `hit_eps`, `overshoot_eps` | numerical floors (error scale, strain rate, hitting and overshooting a sample strain); rarely changed | `1e-14`, `1e-16`, `1e-12`, `1e-10` |
| `max_attempts` | maximum number of accepted plus rejected steps before the run returns a partial trajectory | `50000000` |
| `max_wall_s` | wall-clock budget in seconds (Inf = none) | `Inf` |

### `[output]`

| key | meaning | default |
|---|---|---|
| `sample_mode` | `"strain_grid"` one row every `dstrain_out`; `"strain_samples"` rows at the strains listed in `strain_samples` | `"strain_grid"` |
| `dstrain_out` | strain spacing of the output rows | `1e-2` |
| `strain_samples` | list of strains (0 and `strain_max` are added if missing) | – |

### `[initial_condition]` (read by `build_initial_M`)

| key | meaning | default |
|---|---|---|
| `kind` | `"equilibrium"` chains at rest; `"uniaxial_affine"` chains pre-stretched affinely with the fluid to Hencky strain `micro_affine_strain` before the flow starts | `"equilibrium"` |
| `micro_affine_strain` | the pre-strain ε₀ (each principal component of M scales as exp(2 α_i ε₀)) | `0.0` |

## Keys that are accepted and ignored

The reference inputs under `julia/experiments/c2d2ref/inputs/` were written for
the MATLAB version of the code and carry keys the Julia loaders do not read:
`[run] overwrite`, `write_every`, `out_root`; `[model] weights_method`,
`n_interior`, `n_per_decade`, `write_drag_lut_csv`, `assess_drag_mode`,
`assess_drag_axis`, `assess_spec_mode`, `assess_spec_axis`,
`kink_blend_width`; `[continuation] tol_root`, `max_it_root`, `grow`,
`shrink`; `[newton_fixed_Y] tol_FY_abs`; `[initial_condition]
apply_to_modes`; the whole `[postprocess]` section. They do nothing and can be
deleted.
