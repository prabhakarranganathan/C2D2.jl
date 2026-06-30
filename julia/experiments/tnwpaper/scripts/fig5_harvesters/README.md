# Fig. 5 experimental harvesters — repo-tree-fixed

This version is patched for your actual split repo layout:

- MATLAB paper root:
  `matlab/experiments/tnwpaper`
- Julia paper root:
  `julia/experiments/tnwpaper`

It assumes you have already created the symlink:

```bash
ln -s ../../../../matlab/experiments/tnwpaper/outputs/exptl_comparison \
      julia/experiments/tnwpaper/outputs/exptl_comparison
```

## Important defaults

These scripts now default to:

- spreadsheet:
  `matlab/experiments/tnwpaper/parameter-sets.xlsx`
- processed model/expt CSV root:
  `julia/experiments/tnwpaper/outputs/exptl_comparison`
  (via the symlink above)
- harvested output root:
  `julia/experiments/tnwpaper/outputs/harvested`

## Robustness

The shared helper now tries both of these locations for `TNW.jl`:

- `../src/TNW.jl`
- `../../src/TNW.jl`

So the harvesters work whether you place them in:

- `julia/experiments/tnwpaper/scripts/`
- or `julia/experiments/tnwpaper/scripts/fig5_harvesters/`

## Dependencies

```julia
import Pkg
Pkg.add(["CSV", "DataFrames", "XLSX"])
```

## Recommended location

Put the three `.jl` files directly in:

`julia/experiments/tnwpaper/scripts/`

## Usage from the repo root

Single dataset:

```bash
julia --project=julia \
  julia/experiments/tnwpaper/scripts/harvest_experimental_dataset.jl \
  gaillard
```

All datasets:

```bash
julia --project=julia \
  julia/experiments/tnwpaper/scripts/harvest_experimental_all.jl
```

Explicit full-path version:

```bash
julia --project=julia \
  julia/experiments/tnwpaper/scripts/harvest_experimental_dataset.jl \
  gaillard \
  matlab/experiments/tnwpaper/parameter-sets.xlsx \
  Plan_Exptl \
  julia/experiments/tnwpaper/outputs/harvested
```
