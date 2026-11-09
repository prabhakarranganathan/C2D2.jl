# Changelog

All notable changes to C2D2.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Documentation: the technical report is now linked by its concept DOI,
  [10.5281/zenodo.18012676](https://doi.org/10.5281/zenodo.18012676), which
  always resolves to the latest version, instead of the version-1 record
  (10.5281/zenodo.18012677). Affects the README badge and links,
  `CITATION.cff` (`preferred-citation` DOI, URL and year 2026),
  `docs/input_reference.md` and the `Drag.jl` header. Report section numbers
  cited in the docs and docstrings refer to report version 3
  ([10.5281/zenodo.24035542](https://doi.org/10.5281/zenodo.24035542)); they
  were checked against it and none needed changing.

## [1.1.1] — 2026-11-07

Documentation for human readers, plus one keyword alias. No computed value,
default or key name changed; every input file gives the same results as
v1.1.0.

### Added

- `load_usyext_context(toml; phi = …)`: the concentration override is now
  spelled `phi`, as in `load_styext_context`. The older `phi_kw` still works;
  giving both with different values is an error. (Previously `phi` raised a
  `MethodError`.)

- `docs/input_reference.md`: every TOML key of the two loaders with its
  meaning, unit and default, and the list of keys that are accepted and
  ignored.
- `docs/output_reference.md`: every column of `summary.csv`,
  `caber_diag.csv` and the steady-shear summary.
- `julia/examples/inputs/`: one fully commented input per flow family
  (steady extension, capillary thinning, steady shear), and a third example
  script `run_shear_example.jl`. The two existing example scripts now read
  these commented inputs (same parameter values as before, same output).

### Changed

- README rewritten for a first-time reader: install, run, find the output,
  change a parameter, and a guide to which flow, solver and drag scheme to
  use.
- Docstrings of the public entry points and comments in the kernel now use
  the report's notation (λ₁,₀ for the equilibrium longest relaxation time,
  "HI parameter" for h*_K, "bare-blob" and "shielded-blob" models), point to
  the report section that defines the physics, and no longer refer to
  internal development notes. Two statements were corrected: the
  non-draining limit of the draining function is α_∞ ≈ 5.76 (not 5.775), and
  `De_mode = "cu_fit"` is accepted with any `drag_interp` (it was documented
  as applying only to `"CU"`).
- The capillary-thinning documentation now states what the code does: in
  the inertiocapillary flow the Ohnesorge number is the input and the
  polymer-stress prefactor U_R = Oh U₁/De_R is derived from it; `[flow] U_R`
  is a legacy override that is converted to an equivalent Oh.

### Known issues

Found during the documentation review; behaviour is unchanged in this
release and documented as it is.

- `[model] fe_model = "hookean"` is accepted by the steady loader but rejected
  by the transient loader.
- For `[flow] kind = "caber_inertial"`, a missing `[flow] Oh` silently
  defaults to 0.005 (the aqueous-solution value of Gaillard et al.) with an
  `@info` message rather than an error.
- `[flow] De_Z` is read by the transient loader but ignored by the steady
  loader, where only `De_mode = "cu_fit"` changes the equilibrium drag.
- In `caber_diag.csv` the `viscstress` column is written as
  `capstress + polystress` for the inertiocapillary flow too, where the force
  balance is quadratic in Wi, so it is not 3·Wi there.
- The `K_til` field of `UsyextPoint` holds φ·Γ_eq·U_z for the viscocapillary
  flow and φ·U_R for the inertiocapillary flow.

Archived at [10.5281/zenodo.24033584](https://doi.org/10.5281/zenodo.24033584).

## [1.1.0] — 2026-11-01

**If you used v1.0.0:** nothing you have changes. Every default is the same,
so existing input files give identical results. The new features are
switched on by new optional keys, listed below, and one new flow kind.

### Added

- **Drag-interpolation schemes** (`[model] drag_interp`): the **bare-blob**
  model (`"BAREI"`, `"BARECUI"`) and the **shielded-blob** model (`"OESI"`,
  `"OESCUI"`; overlap-enhanced shielding, in which overlapping chains make the
  blobs non-draining), each with a hard switch between the weakly and strongly
  screened limits (`…I`) or a smooth Churchill–Usagi blend (`…CUI`, exponent
  `p_phi`, default 2). `"RPI"` is an alias of the original `"RP"` scheme, which
  remains the default.
- **Stretch-delayed shielding** `q_oes` for `"OESI"`/`"OESCUI"`: the shielding
  factor is ψ = min(φ/E^q, 1); default 0, 0.25 used in our current work.
- `C_phi`, the overlap constant in φ = C_phi c E₀³ (default 1), and a drag
  look-up table split at the regime boundaries of each scheme
  (`drag_lut_n_points`, default 4000 for the steady solvers).
- Legacy: the first Churchill–Usagi blend `"CU"` with `p_phi = 1.86` and its
  relaxation-time factor options (`g0_1D`, `s_form`), and
  `De_mode = "cu_fit"` with `De_Z`. Kept for reproducing earlier runs.
- **Steady simple shear** via the planar mixed flow (`[flow] kind = "pmf"`,
  `r`, `Wi_convention`): fixed-Wi sweep (`SteadyWiSweep.jl`), driver
  `julia/experiments/pmf/run_pmf.jl`, the King–McKinley grid of inputs,
  reference CSVs from an independent full-tensor implementation, and
  `test_pmf.jl`.
- Kernel tests for every drag scheme against 340-row reference tables
  (`docs/validation/2026-10-22_oesi_v2/oesi.py`), to a relative tolerance of
  1e-10; validation records under `docs/validation/2026-10-2*_oesi*/`.
- **CaBER toolkit** (`tools/c2d2_caber_toolkit/`): a browser page that writes
  self-contained capillary-thinning sweep run folders (local or SLURM) and
  plots the results, with a finished example sweep.

### Changed

- `run_all_tests.jl` also runs the steady-shear (`pmf`) suite.
- The capillary-thinning paper's harvesters prefer the Julia-side
  `parameter-sets.xlsx`.

### Unchanged

- All defaults (`drag_interp = "RP"`, etc.) are bit-identical to v1.0.0;
  existing input files run unchanged. The liquid-bridge (free-surface
  filament) solver remains outside this package.

## [1.0.0] — 2026-07-01

First public release: steady uniaxial extension (STYEXT, pseudo-arclength
continuation), transient extension and CaBER-type capillary thinning (USYEXT,
exponential time differencing), the C2D2 drag kernel, a regression suite with
frozen reference data, and the capillary-thinning paper's run scripts and
harvested data.
Archived at [10.5281/zenodo.21090147](https://doi.org/10.5281/zenodo.21090147).
