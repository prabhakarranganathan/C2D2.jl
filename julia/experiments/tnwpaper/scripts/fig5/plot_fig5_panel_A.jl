#!/usr/bin/env julia
# Fig 5 panel A — Gaillard, unnormalized Pipkin diagram

using CairoMakie, DataFrames
include(joinpath(@__DIR__, "fig5_style.jl"))
using .Fig5Style

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    df  = load_panel_unnormalized(harvested_dir, "gaillard")

    fig, ax = panel_fig_square(
        xlabel = L"\mathrm{De}",
        ylabel = L"\mathrm{Wi}_{\mathrm{e}}",
    )

    plot_dataset_unnorm!(ax, df, "gaillard"; sty = sty)

    # 2/3 low-De plateau reference
    hlines!(ax, [2/3];
            color = sty.ref_dark, linestyle = :dash,
            linewidth = 0.85 * sty.line_lw / 2)

    # axis limits from data
    model = df[coalesce.(string.(df.source_kind) .== "model", false), :]
    expt  = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]
    xmin, xmax = positive_bounds(model.De, expt.De)
    ymin, ymax = positive_bounds(model.Wi_e, expt.Wi_e)
    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    col_g = dataset_color("gaillard"; sty = sty)
    reltext!(ax, 0.03, 0.93, "A";                          sty = sty, color = :black, fs_scale = 1.35)
    reltext!(ax, 0.06, 0.83, L"h^\ast_\mathrm{K} = 0.005"; sty = sty)          # grey (default)
    reltext!(ax, 0.06, 0.73, L"N_\mathrm{K} = 30000";      sty = sty, color = col_g)
    reltext!(ax, 0.06, 0.63, L"\phi = 0.02";               sty = sty, color = col_g)

    save_panel(fig, figdir, "fig5_panel_A")
end

main(ARGS)
