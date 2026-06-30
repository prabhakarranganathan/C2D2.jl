# Fig. 4 true panel split

This version does the split the way you actually wanted:

- `fig4_style.jl` is only boring shared infrastructure:
  - data loading
  - renorm attachment
  - palette / style helpers
  - save helpers
  - generic run-selection helpers

- every panel file owns its own:
  - labels
  - annotations
  - axis limits
  - dataset selections
  - line/symbol choices

So if you want to change A2, you edit `plot_fig4_panel_A2.jl` and nowhere else.

## Files

- `fig4_style.jl`
- `plot_fig4_panel_A1.jl`
- `plot_fig4_panel_A2.jl`
- `plot_fig4_panel_A3.jl`
- `plot_fig4_panel_B1.jl`
- `plot_fig4_panel_B2.jl`
- `plot_fig4_panel_C.jl`

## Usage

```bash
julia --project=. fig4_true_split/plot_fig4_panel_A2.jl
julia --project=. fig4_true_split/plot_fig4_panel_B1.jl
julia --project=. fig4_true_split/plot_fig4_panel_C.jl
```

Optional arguments:

```bash
julia --project=. fig4_true_split/plot_fig4_panel_A2.jl <harvested_dir> <figdir> <renorm_csv>
```

## Next step

Once these independent panel scripts behave, the next file should be a dumb panel assembler.


## Patch note

This fixed version exports `pretty_num` from `fig4_style.jl`, so the panel scripts no longer refer to `PaperFig` directly from `Main`.


## Patch note v3

This version fixes panel geometry by making the axis-box size and gutters explicit in `panel_fig_top` and `panel_fig_square`. That keeps A1/A2/A3 aligned even when the y-labels and y-tick labels are hidden in A2/A3.
