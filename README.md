# C2D2 — Conformation- and Concentration-Dependent Drag

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23848153.svg)](https://doi.org/10.5281/zenodo.23848153)
[![Report DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18012677.svg)](https://doi.org/10.5281/zenodo.18012677)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

C2D2.jl computes the rheology of dilute and semi-dilute solutions of flexible
polymers with the **C2D2** constitutive model, a multimode bead–spring model
in which the hydrodynamic drag on a chain depends on both the chain's
*conformation* (how far it is stretched) and the solution's *concentration*
(how much the chains overlap). The physics and the numerical methods are
described in the accompanying technical report, archived on Zenodo:
[10.5281/zenodo.18012677](https://doi.org/10.5281/zenodo.18012677).

The code solves three homogeneous flow problems:

| Flow | What you get | Solver |
|---|---|---|
| **Steady extension** (uniaxial, planar or biaxial) | the curve of steady polymer stress and extensional viscosity against the Weissenberg number Wi, from the coiled chain to full extension | pseudo-arclength continuation (STYEXT) |
| **Transient extension and capillary thinning** | the chain conformation, Wi and the stresses against strain and time during start-up of extension at fixed Wi, or during the thinning of a filament (CaBER-type flow) in the viscous or the inertial regime | adaptive exponential time-differencing integrator (USYEXT) |
| **Steady simple shear** (and planar mixed flows) | the shear viscosity and the first and second normal-stress coefficients along a sweep in Wi | fixed-Wi sweep or continuation on the same model |

Everything is dimensionless: lengths in units of the equilibrium coil size,
times in units of the longest relaxation time of a chain at rest, stresses in
units of the chain number density times k_BT. The inputs are the number of
Kuhn segments N_K, the number of modes N_m, the hydrodynamic-interaction
parameter h*_K of a Kuhn segment, the overlap ratio φ = c/c*, and the flow
parameters (Wi, or the Deborah and Ohnesorge numbers for capillary thinning).

## Status and support

**This codebase is under active development and evolving.** If you encounter bugs
or other issues, please email <prabhakar.ranganathan@monash.edu>. We cannot offer
routine technical support, but will try to address serious issues as soon as we can.

## Requirements

- [Julia](https://julialang.org/) 1.9 or newer (tested on 1.12). The easiest
  installation is [juliaup](https://github.com/JuliaLang/juliaup).
- Nothing else for the solvers: they depend only on the Julia standard library
  (`LinearAlgebra`, `Printf`, `TOML`), so no packages have to be installed to
  run the examples or the tests.
- The plotting and data-harvesting scripts of the paper figures have extra
  dependencies listed in `julia/Project.toml`; see *Full environment* below.

## Quickstart

Get the code:

```bash
git clone https://github.com/prabhakarranganathan/C2D2.jl.git
cd C2D2.jl
```

Run the steady-extension example:

```bash
julia julia/examples/run_styext_example.jl
```

The first run takes about a minute because Julia compiles the code; later
runs take seconds. The script prints where it wrote its output:
`julia/examples/output/styext_summary.csv`, one row per steady state with the
stretch, Wi, the polymer stresses and the extensional viscosity, plus a copy
of the input file. The other two examples work the same way:

```bash
julia julia/examples/run_usyext_example.jl     # capillary thinning  -> usyext_summary.csv, usyext_caber_diag.csv
julia julia/examples/run_shear_example.jl      # steady simple shear -> shear_summary.csv
```

### Changing a parameter

A run is configured entirely by one TOML file. The three examples read

```
julia/examples/inputs/styext_example.toml    steady uniaxial extension
julia/examples/inputs/caber_example.toml     capillary thinning (and start-up of extension)
julia/examples/inputs/shear_example.toml     steady simple shear
```

in which every key is commented with its meaning and its default. To change
the concentration, for instance, edit `phi` under `[concentration]` and run
the script again; to change the drag scheme, set `drag_interp` under
`[model]`. The complete list of keys, with units and defaults, is in
[`docs/input_reference.md`](docs/input_reference.md); the columns of the
output files are explained in
[`docs/output_reference.md`](docs/output_reference.md).

### Which flow, which solver, which drag scheme?

- You want **steady extensional viscosity against strain rate**: steady
  extension, `[flow] kind = "styext"` with `alphas = [-0.5, -0.5, 1.0]` for
  uniaxial flow. The continuation solver traces the whole curve, including
  the coil–stretch transition where the curve bends back on itself.
- You want **a filament-thinning experiment** (CaBER or dripping-onto-substrate):
  capillary thinning, `[flow] kind = "caber"` for a viscous solvent (Ohnesorge
  number Oh ≫ 1, input De = λ₁,₀/t_v with t_v the viscocapillary time) or
  `"caber_inertial"` for a low-viscosity solvent (Oh ≪ 1, inputs the
  Rayleigh-time Deborah number De_R and `Oh`; the polymer-stress prefactor is
  derived from them). The run returns Wi, the filament radius and the chain
  stretch against time.
- You want **start-up of extension at fixed strain rate**: `[flow] kind =
  "constant"` with `Wi`.
- You want **shear viscosity and normal stresses**: steady shear, `[flow] kind
  = "pmf"` with `r = 1`; give the Wi grid under `[sweep]`.
- **Drag scheme** (`[model] drag_interp`, used when `drag_model = "c2d2"`):
  `"RP"` is the original interpolation of the report and the default;
  `"OESCUI"` (shielded-blob model, smooth join) is the form used in our current
  work, usually with `q_oes = 0.25`. See the table below.
- **Spring law** (`[model] fe_model`): `"fenepm"` is a good default; `"fenep"`
  is more faithful for many modes but costs more; `"hookean"` has no finite
  extensibility and only makes sense for small strains.

### Calling the solvers from your own script

The example scripts are the full pattern (include the modules, load a
context, run, write CSV). The public entry points are:

| Problem | Load context from TOML | Run the solver | Write the output |
|---|---|---|---|
| Steady extension | `load_styext_context(toml)` | `build_manifold(ctx, solver_opts, man_opts)` | `StyextManifold.write_summary_csv` |
| Transient / capillary thinning | `load_usyext_context(toml)` | `integrate_usyext(ctx, flow, plan, tol)` | `UsyextIntegrator.write_summary_csv`, `write_caber_diag_csv` |
| Steady shear | `load_styext_context(toml)` | `sweep_fixed_Wi(ctx, params.Wi_sweep, solver_opts)` | `write_pmf_summary_csv` |

Each function has a docstring with its arguments, return values and the
section of the report that defines the physics (`?load_styext_context` at
the Julia prompt after including the modules). The drag kernel itself can be
called directly: `drag_ratio_zimm` in `julia/models/kernel/Drag.jl` returns
ζ/ζ_Z for one stretch and concentration.

## Drag-interpolation schemes

The C2D2 drag ratio ζ/ζ_Z (`[model] drag_model = "c2d2"`) joins the drag of
the weakly screened limit (a stretched chain seen as a string of tension
blobs) to that of the strongly screened limit (an overlapping chain seen as a
string of correlation blobs). Which join is used, and whether overlapping
chains shield the blobs from the solvent, is selected by `[model] drag_interp`
(report Secs. 2.1.5–2.1.8):

| `drag_interp` | Model | Join | Notes |
|---|---|---|---|
| `"RP"` (= `"RPI"`) | original blob model of the technical report | switch at E = φ | **default**; results identical to v1.0.0 |
| `"BAREI"` | **bare-blob** model: blobs keep their Kirkwood–Riseman draining | switch at E = φ | |
| `"BARECUI"` | **bare-blob** model | Churchill–Usagi blend, exponent `p_phi` | smooth in φ and E |
| `"OESI"` | **shielded-blob** model: overlapping chains make the blobs non-draining (overlap-enhanced shielding) | switch at E = φ | |
| `"OESCUI"` | **shielded-blob** model | Churchill–Usagi blend, exponent `p_phi` | smooth; the form used in our current work |

Additional keys, all optional:

- `p_phi`: Churchill–Usagi exponent of the `…CUI` schemes (default `2.0`).
- `q_oes`: stretch-delayed shielding for `"OESI"`/`"OESCUI"`. The shielding
  factor is ψ = min(φ/E^q, 1): with the default `0` it is set by the overlap
  alone; `0.25` delays the onset of shielding as the chain stretches and is
  the value used in our current capillary-thinning work.
- `C_phi`: overlap constant in φ = C_phi c E₀³ (default `1.0`; about 0.39 for
  polystyrene in diethyl phthalate).

All defaults reproduce v1.0.0 exactly, so existing input files run unchanged;
the newer schemes are opt-in. A minimal shielded-blob `[model]` block:

```toml
[model]
fe_model       = "fenepm"
drag_model     = "c2d2"
draining_model = "partialdraining"
drag_interp    = "OESCUI"
p_phi          = 2.0
q_oes          = 0.25
```

The reference implementation behind the regression tables is
[`docs/validation/2026-10-22_oesi_v2/oesi.py`](docs/validation/2026-10-22_oesi_v2/oesi.py);
`julia/experiments/test/test_kernel.jl` checks all four schemes against it to
a relative tolerance of 1e-10.

**Legacy options** (kept so that earlier runs can be reproduced; not
recommended for new work): `drag_interp = "CU"` (the first Churchill–Usagi
blend of the two limits, with `p_phi = 1.86`) and its relaxation-time factor
keys `g0_1D`, `s_form`; the `"RPD"`/`"RPDW"`/`"RPDP"` variants of `"RP"`; and
`De_mode = "cu_fit"` with `De_Z`. The last is accepted with any
`drag_interp` (it was intended for `"CU"`): it replaces the drag model's own
ratio of the equilibrium relaxation time to its dilute value, λ₁,₀/λ_Z, by
the Martin-type correlation (1 + φ^p)^(1/p) wherever that ratio enters, that
is in the Deborah number and in the capillary-thinning prefactor U₁.

## Notation

The code and its documentation use the notation of the report:

| symbol | meaning |
|---|---|
| E₀, E | equilibrium coil size; chain stretch (end-to-end distance over E₀, so E = 1 at rest) |
| N_K, N_m, h*_K | Kuhn segments per chain; number of modes; hydrodynamic-interaction (HI) parameter of a Kuhn segment |
| φ = C_φ c E₀³ = c/c* | overlap ratio |
| λ₁, λ₁,₀, λ_Z | longest relaxation time of the stretched chain; its equilibrium value at the given concentration; its dilute (Zimm) value. Λ = λ₁/λ₁,₀ = ζ/ζ₀ |
| Wi, De, Oh | Weissenberg number λ₁,₀ ε̇ (or λ₁,₀ γ̇/2 in the planar-mixed-flow convention); Deborah number λ₁,₀ over the macroscopic flow time; Ohnesorge number of the solvent |
| R, R₀ | filament radius and its initial value |

Where a key or a variable name carries an older symbol (`lambda0` for λ₁,₀,
`cu_lambda0_ratio`), its docstring says so.

## CaBER toolkit

[`tools/c2d2_caber_toolkit/`](tools/c2d2_caber_toolkit) is a single-page browser
tool (`c2d2_caber_toolkit.html`) that sets up capillary-thinning sweeps over
N_K, φ, h*_K, De or the prestretch, writes a self-contained run folder with
inputs and job scripts (local or SLURM), and plots the finished runs against
their steady-extension attractors. It exposes the drag-interpolation schemes
above. Its [`README.md`](tools/c2d2_caber_toolkit/README.md) walks through
installation and use; `example_PSDEP_phi_sweep/` is a finished example sweep.

## Tests

```bash
julia julia/experiments/test/run_all_tests.jl
```

runs five suites (kernel, evals, STYEXT, USYEXT and steady shear) and prints a
table with the number of passed tests per suite. "Pass" means: the drag kernel
reproduces its reference tables to a relative tolerance of 1e-10 for every
scheme, and the three solvers reproduce frozen reference outputs
(`julia/experiments/test/reference/`, generated independently with the MATLAB
version of the code and with a full-tensor Python implementation) to a
relative tolerance of 0.5%. The run takes about half a minute after
compilation and ends with `All tests passed.` and exit code 0; a failing suite
prints the last lines of its output. v1.1.1 passes 19 941 tests.

Additional sweep-level validation is kept under
[`docs/validation/`](docs/validation); see its
[`README.md`](docs/validation/README.md).

### Full environment (plotting and experiments)

To reproduce the paper figures, instantiate the project environment first:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Repository layout

```
julia/
  models/
    kernel/        spring laws (Fene.jl), drag (Drag.jl), relaxation spectrum (Spectrum.jl),
                   modal coefficients (CoeffsRelax.jl)
      flows/       STYEXT / planar-mixed-flow and USYEXT flow kinematics
      modelpacks/  assembled steady-flow model pack
      evals/       stress, conformation-tensor and state observables
    numerics/      continuation + Newton (STYEXT); fixed-Wi sweep (shear); ETD integrator (USYEXT)
    front_end/     TOML -> run-context loaders
    utils/         shared plotting helpers
  examples/        the three minimal run scripts and their commented inputs
  experiments/
    c2d2ref/       reference cases (inputs for the regression tests)
    pmf/           steady simple-shear (planar mixed flow) driver + inputs
    test/          regression test suite
    tnwpaper/      capillary-thinning paper: run scripts, harvested data, figures
tools/
  c2d2_caber_toolkit/  browser page that sets up, runs and plots CaBER sweeps
docs/
  input_reference.md   every TOML key, with meaning, unit and default
  output_reference.md  every output column
  validation/          frozen validation data + README
```

## Citing

If you use this software, please cite both:

- the software — this version (v1.1.0) is archived at [10.5281/zenodo.23848153](https://doi.org/10.5281/zenodo.23848153); the concept DOI [10.5281/zenodo.21089959](https://doi.org/10.5281/zenodo.21089959) always resolves to the latest version (metadata in [`CITATION.cff`](CITATION.cff)); and
- the accompanying technical report — [10.5281/zenodo.18012677](https://doi.org/10.5281/zenodo.18012677).

## License

[MIT](LICENSE) © 2026 Prabhakar Ranganathan (Monash University).
