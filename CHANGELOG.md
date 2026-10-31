# Changelog

All notable changes to C2D2.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-11-01

### Added

- **Drag-interpolation schemes** (`[model] drag_interp`): the **bare-blob**
  model (`"BAREI"`, `"BARECUI"`) and the **shielded-blob** model (`"OESI"`,
  `"OESCUI"`, overlap-enhanced shielding of hydrodynamic interactions), each
  with a piecewise join (`…I`) or a Churchill–Usagi blend (`…CUI`, exponent
  `p_phi`, default 2). `"RPI"` is an alias of the original `"RP"` scheme.
- **Stretch-delayed shielding** `q_oes` for `"OESI"`/`"OESCUI"`: overlap
  dressing ψ = min(φ/E^q, 1); default 0, 0.25 used in current work.
- `C_phi` overlap constant; general multi-piece drag look-up table
  (`drag_lut_n_points`) that resolves the scheme boundaries.
- Churchill–Usagi blend `"CU"` of the original two limits with `p_phi = 1.86`,
  and its relaxation-time factor options (`g0_1D`, `s_form`,
  `De_mode = "cu_fit"`, `De_Z`) — experimental / legacy.
- **Steady simple shear** via the planar mixed flow (`[flow] kind = "pmf"`,
  `r`, `Wi_convention`): fixed-Wi sweep (`SteadyWiSweep.jl`), driver
  `julia/experiments/pmf/run_pmf.jl`, King–McKinley grid inputs, reference
  CSVs from an independent full-tensor implementation, and `test_pmf.jl`.
- Kernel tests for every drag scheme against 340-row reference tables
  (`docs/validation/2026-10-22_oesi_v2/oesi.py`), to a relative tolerance of
  1e-10; validation records under `docs/validation/2026-10-2*_oesi*/`.
- **CaBER toolkit** (`tools/c2d2_caber_toolkit/`): browser page that writes
  self-contained CaBER sweep run folders (local or SLURM) and plots the results,
  with a finished example sweep.

### Changed

- `run_all_tests.jl` also runs the `pmf` suite.
- `tnwpaper` harvesters prefer the Julia-side `parameter-sets.xlsx`.

### Unchanged

- All defaults (`drag_interp = "RP"`, etc.) are bit-identical to v1.0.0;
  existing input files run unchanged. The liquid-bridge (VELB/ALE) solver
  remains outside this package.

## [1.0.0] — 2026-07-01

First public release: STYEXT steady uniaxial extension (pseudo-arclength
continuation), USYEXT transient extension and CaBER-type capillary thinning
(ETD2), the C2D2 drag kernel, regression suite with frozen reference data, and
the capillary-thinning paper's run scripts and harvested data.
Archived at [10.5281/zenodo.21090147](https://doi.org/10.5281/zenodo.21090147).
