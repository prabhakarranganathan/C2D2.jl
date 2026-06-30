#!/usr/bin/env julia
# Fig 5 panel F — normalized master plot, all experimental datasets
# Only experimental data points (no model curves).
# Master curve y = (De/Debar)^(4/3) overlaid.

using CairoMakie, DataFrames
include(joinpath(@__DIR__, "fig5_style.jl"))
using .Fig5Style

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    df  = load_expt_normalized_all(harvested_dir)

    # keep only experiment rows (model rows excluded by design of panel F)
    df = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]

    # Wider left gutter to prevent y-label overline being clipped at figure edge;
    # right gutter reduced by same amount to keep total figure width unchanged.
    fig, ax = panel_fig_square(
        xlabel           = L"\mathrm{De}/\overline{\mathrm{De}}",
        ylabel           = L"\mathrm{Wi}_{\mathrm{e}}/\overline{\mathrm{Wi}}_\mathrm{e}",
        left_gutter_cm   = 0.85,
        right_gutter_cm  = 0.78,
    )

    plot_expt_normalized!(ax, df; sty = sty)

    # axis limits
    De_bar_vals  = [Float64(x) for x in df.De_bar  if !ismissing(x) && isfinite(Float64(x)) && Float64(x) > 0]
    Wie_bar_vals = [Float64(x) for x in df.Wi_e_bar if !ismissing(x) && isfinite(Float64(x)) && Float64(x) > 0]
    xmin, xmax = positive_bounds(De_bar_vals;  pad_low = 0.1, pad_high = 0.3)
    ymin, ymax = positive_bounds(Wie_bar_vals; pad_low = 0.1, pad_high = 0.3)
    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    reltext!(ax, 0.03, 0.93, "F"; sty = sty, color = :black, fs_scale = 1.35)

    save_panel(fig, figdir, "fig5_panel_F")
end

main(ARGS)
