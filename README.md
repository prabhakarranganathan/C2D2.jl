# C2D2 — Conformation- and Concentration-Dependent Drag

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23848153.svg)](https://doi.org/10.5281/zenodo.23848153)
[![Report DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18012677.svg)](https://doi.org/10.5281/zenodo.18012677)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Julia implementation of the **C2D2** constitutive model for the rheology of
dilute and semi-dilute viscoelastic polymer solutions, in which the
hydrodynamic drag on each polymer segment depends on both the local chain
*conformation* and the solution *concentration*.

The code solves two homogeneous flow problems:

- **STYEXT** — steady uniaxial extension. The stress/Weissenberg-number
  response is traced with a pseudo-arclength continuation (PAC) solver.
- **USYEXT** — transient uniaxial step-strain and capillary-thinning
  (CaBER-type) flows, integrated with an adaptive exponential time-differencing
  (ETD2) scheme.
- **Steady simple shear** (since v1.1.0) — the steady planar mixed flow
  (`[flow] kind = "pmf"`, `r = 1`) shares the STYEXT model pack and gives
  the shear viscosity and normal-stress coefficients along a Weissenberg-number
  sweep; see `julia/experiments/pmf/`.

The accompanying technical report is archived on Zenodo:
[10.5281/zenodo.18012677](https://doi.org/10.5281/zenodo.18012677).

## Status and support

**This codebase is under active development and evolving.** If you encounter bugs
or other issues, please email <prabhakar.ranganathan@monash.edu>. We cannot offer
routine technical support, but will try to address serious issues as soon as we can.

## Requirements

- [Julia](https://julialang.org/) ≥ 1.9 (tested on 1.12).
- The STYEXT/USYEXT solver itself depends **only on the Julia standard
  library** (`LinearAlgebra`, `Printf`, `TOML`) — no package installation is
  needed to run the examples below.
- The plotting / data-harvesting code has extra dependencies listed in
  `julia/Project.toml`; install them with the `Pkg.instantiate()` step shown
  under *Full environment*.

## Quickstart

```bash
git clone https://github.com/prabhakarranganathan/C2D2.jl.git
cd C2D2.jl
```

Run a minimal steady-extension (STYEXT) case driven by a TOML input file:

```bash
julia julia/examples/run_styext_example.jl
```

…or a transient capillary-thinning (USYEXT/CaBER) case:

```bash
julia julia/examples/run_usyext_example.jl
```

Each writes a `summary.csv` (and, for CaBER, a stress-decomposition
`caber_diag.csv`) plus a copy of the input TOML into `julia/examples/output/`.

### Driving your own run

A run is configured entirely by a TOML file. Reference inputs live in
[`julia/experiments/c2d2ref/inputs/`](julia/experiments/c2d2ref/inputs). The
public entry points are:

| Problem | Load context from TOML            | Run the solver                              | Output |
|---------|-----------------------------------|---------------------------------------------|--------|
| STYEXT  | `load_styext_context(toml)`       | `build_manifold(ctx, solver_opts, man_opts)`| `write_summary_csv` |
| USYEXT  | `load_usyext_context(toml)`       | `integrate_usyext(ctx, flow, plan, tol)`    | `write_summary_csv`, `write_caber_diag_csv` |

See the two scripts in `julia/examples/` for the full, copy-pasteable pattern
(module includes + calls).

## Drag-interpolation schemes

The C2D2 drag ratio ζ/ζ_Z (`[model] drag_model = "c2d2"`) is assembled from the
drag of the weakly screened (tension-blob) and strongly screened
(correlation-blob) limits. How the two limits are joined, and whether the blobs
shield each other's hydrodynamic interactions, is selected by
`[model] drag_interp`:

| `drag_interp` | Model | Join | Notes |
|---|---|---|---|
| `"RP"` (= `"RPI"`) | original blob model of the technical report | switch at E = φ | **default**; bit-identical to v1.0.0 |
| `"BAREI"` | **bare-blob** model | piecewise (continuous, slope corners) | bare Kirkwood–Riseman draining of tension and correlation blobs |
| `"BARECUI"` | **bare-blob** model | Churchill–Usagi blend, exponent `p_phi` | smooth in φ and E |
| `"OESI"` | **shielded-blob** model | piecewise | overlap-enhanced shielding of the hydrodynamic interactions |
| `"OESCUI"` | **shielded-blob** model | Churchill–Usagi blend, exponent `p_phi` | smooth; the form used in our current work |

Additional keys, all optional:

- `p_phi` — Churchill–Usagi exponent for the `…CUI` schemes (default `2.0`;
  `1.86` for the legacy `"CU"` scheme below).
- `q_oes` — stretch-delayed shielding exponent for `"OESI"`/`"OESCUI"`: the
  overlap dressing is ψ = min(φ/E^q, 1). Default `0` (shielding set by the
  overlap alone); `0.25` delays the onset of shielding as the chain stretches
  and is the value used in our current capillary-thinning work.
- `C_phi` — overlap constant in c = φ/(C_phi E₀³) (default `1.0`).

All defaults reproduce v1.0.0 exactly, so existing input files run unchanged;
the new schemes are opt-in. A minimal shielded-blob `[model]` block:

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

**Experimental / legacy options** (kept for reproducibility of earlier runs,
not recommended for new work): `drag_interp = "CU"` (the first Churchill–Usagi
blend of the two limits, with `p_phi = 1.86`), the `"RPD"`/`"RPDW"`/`"RPDP"`
overlap-dressed variants of `"RP"`, and the relaxation-time factor options
`g0_1D`, `s_form` and `De_mode = "cu_fit"` with `De_Z` that apply only to `"CU"`.

## CaBER toolkit

[`tools/c2d2_caber_toolkit/`](tools/c2d2_caber_toolkit) is a single-page browser
tool (`c2d2_caber_toolkit.html`) that sets up capillary-thinning sweeps over
N_K, φ, h*_K, De or the prestretch, writes a self-contained run folder with
inputs and job scripts (local or SLURM), and plots the finished runs against
their steady-extension attractors. It exposes the drag-interpolation schemes
above. Its [`README.md`](tools/c2d2_caber_toolkit/README.md) walks through
installation and use; `example_PSDEP_phi_sweep/` is a finished example sweep.

### Full environment (plotting and experiments)

To reproduce the paper figures, instantiate the project environment first:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Validation

Reference cases with frozen expected output are exercised by the test suite:

```bash
julia julia/experiments/test/run_all_tests.jl
```

This runs the kernel, evals, STYEXT, USYEXT and steady-shear (PMF) suites. The
STYEXT/USYEXT suites compare the solver output against the CSVs in
`julia/experiments/test/reference/` to a relative tolerance of 0.5%; the kernel
suite checks every drag-interpolation scheme against its reference table. Additional
sweep-level validation is kept under [`docs/validation/`](docs/validation) — see
its [`README.md`](docs/validation/README.md).

## Repository layout

```
julia/
  models/
    kernel/        FENE springs, drag, spectrum, relaxation coefficients
      flows/       STYEXT / USYEXT flow kinematics
      modelpacks/  assembled model packs
      evals/       stress, conformation-tensor and state observables
    numerics/      PAC manifold + Newton (STYEXT); fixed-Wi sweep; ETD2 integrator (USYEXT)
    front_end/     TOML -> run-context loaders
    utils/         shared plotting helpers
  examples/        minimal STYEXT / USYEXT run scripts
  experiments/
    c2d2ref/       reference cases (inputs for the regression tests)
    pmf/           steady simple-shear (planar mixed flow) driver + inputs
    test/          regression test suite
    tnwpaper/      capillary-thinning paper: run scripts, harvested data, figures
tools/
  c2d2_caber_toolkit/  browser page that sets up, runs and plots CaBER sweeps
docs/
  validation/      frozen validation data + README
```

## Citing

If you use this software, please cite both:

- the software — this version (v1.1.0) is archived at [10.5281/zenodo.23848153](https://doi.org/10.5281/zenodo.23848153); the concept DOI [10.5281/zenodo.21089959](https://doi.org/10.5281/zenodo.21089959) always resolves to the latest version (metadata in [`CITATION.cff`](CITATION.cff)); and
- the accompanying technical report — [10.5281/zenodo.18012677](https://doi.org/10.5281/zenodo.18012677).

## License

[MIT](LICENSE) © 2026 Prabhakar Ranganathan (Monash University).
