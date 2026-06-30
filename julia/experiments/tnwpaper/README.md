# Thin-n-Win Julia postprocessing

This folder holds Julia-side harvesting and plotting code for the Thin-n-Win paper.

## Purpose

- Harvest production run outputs from `julia/experiments/tnwpaper/outputs/...`
- Build canonical flat tables for plotting and analysis
- Generate reproducible paper figures with CairoMakie

The production simulations are produced by the Julia scripts under `run/`
(e.g. `run/run_sweep.jl`, `run/run_generic_validation.jl`,
`run/run_exptl_comparison.jl`), which drive the C2D2 STYEXT/USYEXT solver in
`julia/models/`. The scripts in this folder then harvest, filter, and plot
those outputs. (The pipeline was originally MATLAB-driven; that lineage now
survives only for a few parameter/renormalisation exports noted in
`PAPER_DATA.md`.)

## First-half figure spine

Current working figure sequence for the generic-behaviour runs:

1. **Intro / defining `Wi_e`**
   - representative CaBER case
   - `R(t)`, `Wi` vs strain, polymer viscosity vs strain
2. **FENE-P Pipkin plot**
   - `Wi_e` vs `De` for several concentrations
3. **FENE-P explanation figure**
   - transient CaBER `eta_p` vs `Wi` trajectories overlaid on steady `styext` curves
   - two rows: moderately dilute and highly dilute
   - four representative `De` values
4. **C2D2 counterpart**
   - same conceptual sequence, more compactly presented
5. **Effect of `N_K` and `h_K*`**
6. **`De^dagger` vs `phi` summary** for both FENE-P and C2D2

## File layout

- `src/TNW.jl` - main module; includes the files below
- `src/metadata.jl` - folder-name parsing and metadata extraction
- `src/harvest.jl` - table harvesting from run outputs
- `src/plotutils.jl` - figure helpers and plot-side filters
- `scripts/harvest_generic_features.jl` - build canonical harvest tables
- `scripts/plot_fig2_fenep_pipkin.jl` - first proof-of-life reduced plot

## Canonical harvested tables

The harvester writes these to `julia/experiments/tnwpaper/outputs/harvested/generic_features/`:

- `manifest.csv`
  - one row per top-level sweep folder
- `curves.csv`
  - all rows from all `summary.csv` files with sweep/run metadata attached
- `features_selected.csv`
  - all rows from all per-run `wi_features_selected.csv` files with metadata attached
- `sweep_features.csv`
  - all rows from top-level `wi_features_vs_phi.csv`, `wi_features_vs_De.csv`, `Wi_e_vs_De.csv`

## Default input paths

The scripts assume the run outputs live at:

- `julia/experiments/tnwpaper/outputs/generic_features`
- `julia/experiments/tnwpaper/parameter-sets.xlsx`

These can be overridden by command-line arguments.

## Shipped data and reproducing the experimental-comparison figures

The digitised experimental data and the harvested comparison tables for the four
datasets used in the paper (`anna`, `calabrese`, `clasen`, `gaillard`) are shipped
under [`data/`](data) — see [`data/README.md`](data/README.md) for per-dataset
provenance and attribution. The harvested tables live in
`data/harvested/experimental_<dataset>/` (`expt_points.csv`, `model_curves_*.csv`,
`panel_*.csv`, `renorm_lookup.csv`, …).

The experimental panels (Fig. 5) plot directly from these. After
`Pkg.instantiate()` (for CairoMakie etc.), pass the shipped harvested dir as the
first argument, e.g.:

```bash
julia --project=julia \
  julia/experiments/tnwpaper/scripts/fig5/plot_fig5_panel_A.jl \
  julia/experiments/tnwpaper/data/harvested
```

Note: the *generic-behaviour* figures (Figs. 2–4) consume a separate
`generic_features` harvest produced by the `run/*.jl` sweeps; that intermediate is
not shipped, so those figures require re-running the sweeps first.

## Notes

- The harvester currently treats the directory tree as the source of truth.
- The spreadsheet can be joined in later for notes or ordering, but the first pass does not depend on it.
- The code is designed to tolerate missing per-run `wi_features_selected.csv` files.
