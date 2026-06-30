#!/usr/bin/env julia
# Fig 5 panel E — unnormalized experimental data, all datasets.
# Scatter only, no curves. Full-size square panel.

using CairoMakie, DataFrames
include(joinpath(@__DIR__, "fig5_style.jl"))
using .Fig5Style

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    df  = load_expt_unnormalized_all(harvested_dir)

    # keep experiment rows only
    df = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]

    fig, ax = panel_fig_square(
        xlabel = L"\mathrm{De}",
        ylabel = L"\mathrm{Wi}_{\mathrm{e}}",
    )

    plot_expt_unnorm!(ax, df; sty = sty)

    De_vals  = [Float64(x) for x in df.De  if !ismissing(x) && isfinite(Float64(x)) && Float64(x) > 0]
    Wie_vals = [Float64(x) for x in df.Wi_e if !ismissing(x) && isfinite(Float64(x)) && Float64(x) > 0]
    xmin, xmax = positive_bounds(De_vals;  pad_low = 0.05, pad_high = 0.2)
    ymin, ymax = positive_bounds(Wie_vals; pad_low = 0.05, pad_high = 0.2)
    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    reltext!(ax, 0.03, 0.93, "E"; sty = sty, color = :black, fs_scale = 1.35)

    save_panel(fig, figdir, "fig5_panel_E")
end

main(ARGS)
