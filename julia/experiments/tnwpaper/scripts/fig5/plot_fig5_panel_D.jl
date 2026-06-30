#!/usr/bin/env julia
# Fig 5 panel D — Calabrese, unnormalized Pipkin diagram
# Samples are phi-shaded: lowest phi → lightest plum, highest phi → darkest plum.
# Two distinct NK values (9300, 21800) — per-sample NK labels, phi-shade colored.

using CairoMakie, DataFrames
using Printf, LaTeXStrings
include(joinpath(@__DIR__, "fig5_style.jl"))
using .Fig5Style

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    df  = load_panel_unnormalized(harvested_dir, "calabrese")

    # ---- build phi-shade color map (lightest = lowest phi, darkest = highest phi) ----
    model = df[coalesce.(string.(df.source_kind) .== "model", false), :]
    expt  = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]

    phi_by_sid = Dict{String,Float64}()
    for row in eachrow(model)
        sid = string(row.sample_id)
        haskey(phi_by_sid, sid) || (phi_by_sid[sid] = Float64(row.phi))
    end
    phi_sids      = sort(collect(keys(phi_by_sid)); by = s -> phi_by_sid[s])
    base_col      = dataset_color("calabrese"; sty = sty)
    shades        = phi_shade_colors(base_col, length(phi_sids))
    sample_colors = Dict{String,RGBf}(sid => shades[i] for (i, sid) in enumerate(phi_sids))

    # ---- figure ----
    fig, ax = panel_fig_square(
        xlabel = L"\mathrm{De}",
        ylabel = L"\mathrm{Wi}_{\mathrm{e}}",
    )

    plot_dataset_unnorm!(ax, df, "calabrese"; sty = sty, sample_colors = sample_colors)

    hlines!(ax, [2/3];
            color = sty.ref_dark, linestyle = :dash,
            linewidth = 0.85 * sty.line_lw / 2)

    xmin, xmax = positive_bounds(model.De, expt.De)
    ymin, ymax = positive_bounds(model.Wi_e, expt.Wi_e)
    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    # panel label + h_K* (grey, topmost)
    reltext!(ax, 0.03, 0.93, "D";                           sty = sty, color = :black, fs_scale = 1.35)
    reltext!(ax, 0.06, 0.83, L"h^\ast_\mathrm{K} = 0.015"; sty = sty)   # grey (default)

    # ---- phi labels (phi_sids ascending: VC3=0.03, VC4=0.07, VC1=0.09, VC2=0.22) ----
    # First label: full "φ = value"; rest: value only.
    # "current" x = 0.03 (annotate_phi_left! default); y values specified by RP.
    reltext!(ax, 0.2,  0.2,  latexstring("\\phi = 0.03"); sty = sty, color = shades[1], align = (:left, :bottom))
    reltext!(ax, 0.03, 0.2,  latexstring("0.07");         sty = sty, color = shades[2], align = (:left, :bottom))
    reltext!(ax, 0.03, 0.35, latexstring("0.09");         sty = sty, color = shades[3], align = (:left, :bottom))
    reltext!(ax, 0.2,  0.35, latexstring("0.22");         sty = sty, color = shades[4], align = (:left, :bottom))

    # ---- NK labels (manual; x=0.97 right-aligned, y positions by RP) ----
    # NK=9300 → full format; NK=21800 → value only.
    # Representative sid per NK = sample with highest max Wi_e for that NK.
    let
        nk_by_sid   = Dict{String,Int}()
        nk_max_wie  = Dict{Int,Float64}()
        nk_rep_sid  = Dict{Int,String}()
        for row in eachrow(model)
            sid = string(row.sample_id)
            haskey(nk_by_sid, sid) || (nk_by_sid[sid] = Int(round(Float64(row.NK))))
        end
        for sid in keys(nk_by_sid)
            nk  = nk_by_sid[sid]
            sub = model[coalesce.(string.(model.sample_id) .== sid, false), :]
            wie = isempty(sub) ? -Inf : maximum(Float64.(sub.Wi_e))
            if !haskey(nk_max_wie, nk) || wie > nk_max_wie[nk]
                nk_max_wie[nk] = wie
                nk_rep_sid[nk] = sid
            end
        end
        col9300  = get(sample_colors, get(nk_rep_sid, 9300,  ""), base_col)
        col21800 = get(sample_colors, get(nk_rep_sid, 21800, ""), base_col)
        reltext!(ax, 0.589, 0.365, latexstring("N_\\mathrm{K} = 9300");
                 sty = sty, color = col9300,  align = (:right, :bottom))
        reltext!(ax, 0.604, 0.174, latexstring("21800");
                 sty = sty, color = col21800, align = (:right, :bottom))
    end

    # ---- arrows (one per phi, tail → head) ----
    # φ=0.09 (shades[3]): tail (0.601, 0.351) → head (0.637, 0.324)
    arrows2d!(ax, [0.601f0], [0.351f0], [0.036f0], [-0.027f0];
              space = :relative, color = shades[3],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)
    # φ=0.22 (shades[4]): tail (0.590, 0.346) → head (0.542, 0.299)
    arrows2d!(ax, [0.590f0], [0.346f0], [-0.048f0], [-0.047f0];
              space = :relative, color = shades[4],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)
    # φ=0.07 (shades[2]): tail (0.615, 0.185) → head (0.692, 0.251)
    arrows2d!(ax, [0.615f0], [0.185f0], [0.077f0], [0.066f0];
              space = :relative, color = shades[2],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)
    # φ=0.03 (shades[1]): tail (0.623, 0.201) → head (0.661, 0.295)
    arrows2d!(ax, [0.623f0], [0.201f0], [0.038f0], [0.094f0];
              space = :relative, color = shades[1],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)
    # φ=0.22 (shades[4]): tail (0.289, 0.351) → head (0.331, 0.298)
    arrows2d!(ax, [0.289f0], [0.351f0], [0.042f0], [-0.053f0];
              space = :relative, color = shades[4],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)
    # φ=0.03 (shades[1]): tail (0.195, 0.224) → head (0.155, 0.279)
    arrows2d!(ax, [0.195f0], [0.224f0], [-0.040f0], [0.055f0];
              space = :relative, color = shades[1],
              tipwidth = 0.012f0, tiplength = 0.020f0, shaftwidth = 0.003f0)

    save_panel(fig, figdir, "fig5_panel_D")
end

main(ARGS)
