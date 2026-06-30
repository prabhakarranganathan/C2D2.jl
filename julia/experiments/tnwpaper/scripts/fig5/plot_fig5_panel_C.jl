#!/usr/bin/env julia
# Fig 5 panel C — Clasen, unnormalized Pipkin diagram
# Samples are phi-shaded: lowest phi → lightest orange, highest phi → darkest orange.
# All samples share NK = 11200 (single NK label in middle shade).

using CairoMakie, DataFrames
using Printf, LaTeXStrings
include(joinpath(@__DIR__, "fig5_style.jl"))
using .Fig5Style

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    df  = load_panel_unnormalized(harvested_dir, "clasen")

    # ---- build phi-shade color map (lightest = lowest phi, darkest = highest phi) ----
    model = df[coalesce.(string.(df.source_kind) .== "model", false), :]
    expt  = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]

    phi_by_sid = Dict{String,Float64}()
    for row in eachrow(model)
        sid = string(row.sample_id)
        haskey(phi_by_sid, sid) || (phi_by_sid[sid] = Float64(row.phi))
    end
    phi_sids      = sort(collect(keys(phi_by_sid)); by = s -> phi_by_sid[s])
    base_col      = dataset_color("clasen"; sty = sty)
    shades        = phi_shade_colors(base_col, length(phi_sids))
    sample_colors = Dict{String,RGBf}(sid => shades[i] for (i, sid) in enumerate(phi_sids))
    # Middle shade for the single NK label (for even n, pick the lower-middle index)
    mid_col       = shades[max(1, div(length(shades) + 1, 2))]

    # ---- figure ----
    fig, ax = panel_fig_square(
        xlabel = L"\mathrm{De}",
        ylabel = L"\mathrm{Wi}_{\mathrm{e}}",
    )

    plot_dataset_unnorm!(ax, df, "clasen"; sty = sty, sample_colors = sample_colors)

    hlines!(ax, [2/3];
            color = sty.ref_dark, linestyle = :dash,
            linewidth = 0.85 * sty.line_lw / 2)

    xmin, xmax = positive_bounds(model.De, expt.De)
    ymin, ymax = positive_bounds(model.Wi_e, expt.Wi_e)
    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    # panel label + h_K* (grey, topmost) + N_K (middle-shade orange)
    reltext!(ax, 0.03, 0.93, "C";                          sty = sty, color = :black, fs_scale = 1.35)
    reltext!(ax, 0.06, 0.83, L"h^\ast_\mathrm{K} = 0.04"; sty = sty)           # grey (default)
    reltext!(ax, 0.06, 0.73, L"N_\mathrm{K} = 11200";     sty = sty, color = mid_col)

    # ---- phi labels (phi_sids sorted ascending: C1..C6, shades[1..6] lightest→darkest) ----
    # First label: full "φ = value" format; rest: value only.
    reltext!(ax, 0.03, 0.63, latexstring("\\phi = 0.004"); sty = sty, color = shades[1], align = (:left, :bottom))
    reltext!(ax, 0.03, 0.53, latexstring("0.005");         sty = sty, color = shades[2], align = (:left, :bottom))
    reltext!(ax, 0.2,  0.5,  latexstring("0.016");         sty = sty, color = shades[3], align = (:left, :bottom))
    reltext!(ax, 0.4,  0.5,  latexstring("0.05");          sty = sty, color = shades[4], align = (:left, :bottom))
    reltext!(ax, 0.2,  0.33, latexstring("0.16");          sty = sty, color = shades[5], align = (:left, :bottom))
    reltext!(ax, 0.4,  0.33, latexstring("0.51");          sty = sty, color = shades[6], align = (:left, :bottom))

    # Arrow φ=0.05: tail (0.5, 0.5) → head (0.56, 0.44)
    arrows2d!(ax, [0.5f0], [0.5f0], [0.06f0], [-0.06f0];
              space = :relative, color = shades[4],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)
    # Arrow φ=0.16: tail (0.28, 0.38) → head (0.328, 0.415)
    arrows2d!(ax, [0.28f0], [0.38f0], [0.048f0], [0.035f0];
              space = :relative, color = shades[5],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)

    save_panel(fig, figdir, "fig5_panel_C")
end

main(ARGS)
