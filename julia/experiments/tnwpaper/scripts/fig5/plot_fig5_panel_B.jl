#!/usr/bin/env julia
# Fig 5 panel B — Anna, unnormalized Pipkin diagram
# Samples are phi-shaded: lowest phi → lightest blue, highest phi → darkest blue.

using CairoMakie, DataFrames
using Printf, LaTeXStrings
include(joinpath(@__DIR__, "fig5_style.jl"))
using .Fig5Style

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    df  = load_panel_unnormalized(harvested_dir, "anna")

    # ---- build phi-shade color map (lightest = lowest phi, darkest = highest phi) ----
    model = df[coalesce.(string.(df.source_kind) .== "model", false), :]
    expt  = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]

    phi_by_sid = Dict{String,Float64}()
    for row in eachrow(model)
        sid = string(row.sample_id)
        haskey(phi_by_sid, sid) || (phi_by_sid[sid] = Float64(row.phi))
    end
    phi_sids     = sort(collect(keys(phi_by_sid)); by = s -> phi_by_sid[s])
    shades       = phi_shade_colors(dataset_color("anna"; sty = sty), length(phi_sids))
    sample_colors = Dict{String,RGBf}(sid => shades[i] for (i, sid) in enumerate(phi_sids))

    # ---- figure ----
    fig, ax = panel_fig_square(
        xlabel = L"\mathrm{De}",
        ylabel = L"\mathrm{Wi}_{\mathrm{e}}",
    )

    plot_dataset_unnorm!(ax, df, "anna"; sty = sty, sample_colors = sample_colors)

    hlines!(ax, [2/3];
            color = sty.ref_dark, linestyle = :dash,
            linewidth = 0.85 * sty.line_lw / 2)

    xmin, xmax = positive_bounds(model.De, expt.De)
    ymin, ymax = positive_bounds(model.Wi_e, expt.Wi_e)
    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    # panel label + h_K* (grey, top-left)
    reltext!(ax, 0.03, 0.93, "B";              sty = sty, color = :black, fs_scale = 1.35)
    reltext!(ax, 0.06, 0.83, L"h^\ast_\mathrm{K} = 0.01"; sty = sty)

    # ---- phi labels (phi_sids sorted ascending: A1 < A2 < A3 by phi) ----
    # A1 (lowest phi = 0.13): auto y from Wi_e at lowest De
    let
        log_ymin = log10(ymin); log_ymax = log10(ymax)
        y_span   = log_ymax - log_ymin
        sub1     = sort(model[coalesce.(string.(model.sample_id) .== phi_sids[1], false), :], :De)
        if !isempty(sub1)
            wie_low  = Float64(sub1.Wi_e[1])
            y1       = clamp((log10(wie_low) - log_ymin) / y_span, 0.02, 0.94) + 0.04
            phi_str1 = @sprintf("%.3g", phi_by_sid[phi_sids[1]])
            reltext!(ax, 0.03, y1, latexstring("\\phi = $(phi_str1)");
                     sty = sty, color = sample_colors[phi_sids[1]], align = (:left, :bottom))
        end
    end
    # A2 (phi = 0.23): user-specified position
    reltext!(ax, 0.12, 0.36, latexstring("0.23");
             sty = sty, color = sample_colors[phi_sids[2]], align = (:left, :bottom))
    # A3 (phi = 0.4): user-specified position
    reltext!(ax, 0.12, 0.27, latexstring("0.4");
             sty = sty, color = sample_colors[phi_sids[3]], align = (:left, :bottom))

    # ---- N_K labels (phi_sids[1] = NK=2700 has highest Wi_e endpoint for Anna) ----
    # NK=2700: full format, automatic top position
    reltext!(ax, 0.97, 0.88, latexstring("N_\\mathrm{K} = 2700");
             sty = sty, color = sample_colors[phi_sids[1]], align = (:right, :bottom))
    # NK=8760: user-specified position
    reltext!(ax, 0.88, 0.69, latexstring("8760");
             sty = sty, color = sample_colors[phi_sids[2]], align = (:right, :bottom))
    # NK=27000: user-specified position
    reltext!(ax, 0.85, 0.36, latexstring("27000");
             sty = sty, color = sample_colors[phi_sids[3]], align = (:right, :bottom))

    save_panel(fig, figdir, "fig5_panel_B")
end

main(ARGS)
