#!/usr/bin/env julia

using CSV
using DataFrames
using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "TNW.jl"))
using .TNW

include(joinpath(TNW.julia_root(), "models", "utils", "paperfig.jl"))
using .PaperFig

function load_sweep_features(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "sweep_features.csv")
    isfile(csv) || error("Missing harvested sweep_features.csv: $csv. Run harvest_generic_features.jl first.")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir = length(args) >= 2 ? args[2] : joinpath(paper_root(), "figures")
    ensure_dir(figdir)

    df = load_sweep_features(harvested_dir)
    mask = trues(nrow(df))
    mask .&= df.sweep_feature_kind .== "Wi_e_vs_De"
    mask .&= df.flow_family .== "caber"
    mask .&= df.drag_model .== "constant"
    mask .&= df.NK .== 3000
    mask .&= isapprox.(df.hK_star, 0.05; atol=1e-12)

    d = df[mask, :]
    isempty(d) && error("No rows found for baseline FENE-P Pipkin plot.")

    wanted_phi = [0.01, 0.1, 0.5, 1.0]
    d = d[map(x -> any(isapprox(x, p; atol=1e-12) for p in wanted_phi), d.phi), :]
    sort!(d, [:phi, :De])
    phis = unique(d.phi)

    fig, sty = new_figure(:fig2_singlecol_square)
    ax = make_axis(fig[1, 1]; xlabel=L"\mathrm{De}", ylabel=L"\mathrm{Wi}_{\mathrm{e}}")

    DEBUG_GUIDES = false

    if DEBUG_GUIDES
        relative_guide_grid!(ax; color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
    end

    cols = reverse(fenep_gray_series(length(phis); sty=sty))
    phi_to_col = Dict(phi => cols[i] for (i, phi) in enumerate(phis))

    label_pos = Dict(
        0.01 => (5, 0.88, 1.25),
        0.1 => (6, 0.85, 1.25),
        0.5 => (8, 0.85, 1.30),
        1.0 => (9, 1.05, 0.85),
    )

    xmin, xmax = 1e-2, 1.8e2
    ymin, ymax = 1e-1, 1e1

    for phi in phis
        di = d[isapprox.(d.phi, phi; atol=1e-12), :]
        sort!(di, :De)
        col = phi_to_col[phi]
        lines!(ax, di.De, di.Wi_e; color=col, linewidth=sty.line_lw)

        scatter!(ax, di.De, di.Wi_e;
            color=col,
            markersize=sty.marker_size,
            strokecolor=darken(col, 0.15),
            strokewidth=0.6
        )

        if haskey(label_pos, phi)
            j, xfac, yfac = label_pos[phi]
            j = clamp(j, 1, nrow(di))
            text!(ax, [di.De[j] * xfac], [di.Wi_e[j] * yfac];
                text=[PaperFig.pretty_num(phi)], color=darken(col, 0.08),
                align=(:left, :center), fontsize=sty.annotation_scale * sty.font_base_pt,
                font=:regular)
        end
    end

    hlines!(ax, [2 / 3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)
    text!(ax, [1.35e-2], [2 / 3 * 0.96]; text=[L"2/3"], color=sty.ref_dark,
        align=(:left, :top), fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
    text!(ax, [0.3], [3.65]; text=[L"\phi \,=\,"], color=soft_gray(sty),
        align=(:left, :top), fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
    parameter_block!(ax, [
            "FENE-P",
            L"N_{\mathrm{K}} = %$(PaperFig.pretty_num(d.NK[1]))"
        ]; sty=sty, x=0.33, y0=0.22, dy=0.075, color=soft_gray(sty))

    apply_log10_x!(ax, xmin, xmax)
    apply_log10_y!(ax, ymin, ymax)

    out_pdf = joinpath(figdir, "fig2_fenep_pipkin.pdf")
    out_png = joinpath(figdir, "fig2_fenep_pipkin.png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)

    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
end

main(ARGS)
