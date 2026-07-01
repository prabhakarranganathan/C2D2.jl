# C2D2 — Conformation- and Concentration-Dependent Drag

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21090147.svg)](https://doi.org/10.5281/zenodo.21090147)
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

### Full environment (plotting and experiments)

To reproduce the paper figures, instantiate the project environment first:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Validation

Reference cases with frozen expected output are exercised by the test suite:

```bash
julia julia/experiments/test/test_styext.jl
julia julia/experiments/test/test_usyext.jl
```

These compare the solver output against the CSVs in
`julia/experiments/test/reference/` to a relative tolerance of 0.5%. Additional
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
    numerics/      PAC manifold + Newton (STYEXT); ETD2 integrator (USYEXT)
    front_end/     TOML -> run-context loaders
    utils/         shared plotting helpers
  examples/        minimal STYEXT / USYEXT run scripts
  experiments/
    c2d2ref/       reference cases (inputs for the regression tests)
    test/          regression test suite
    tnwpaper/      capillary-thinning paper: run scripts, harvested data, figures
docs/
  validation/      frozen validation data + README
```

## Citing

If you use this software, please cite both:

- the software — archived at [10.5281/zenodo.21090147](https://doi.org/10.5281/zenodo.21090147) (metadata in [`CITATION.cff`](CITATION.cff)); and
- the accompanying technical report — [10.5281/zenodo.18012677](https://doi.org/10.5281/zenodo.18012677).

## License

[MIT](LICENSE) © 2026 Prabhakar Ranganathan (Monash University).
