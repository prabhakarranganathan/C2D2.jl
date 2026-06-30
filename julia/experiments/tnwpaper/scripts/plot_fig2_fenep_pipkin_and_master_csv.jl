#!/usr/bin/env julia

using CSV
using DataFrames
using CairoMakie
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "..", "src", "TNW.jl"))
using .TNW

include(joinpath(TNW.julia_root(), "models", "utils", "paperfig.jl"))
using .PaperFig

function load_sweep_features(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "sweep_features.csv")
    isfile(csv) || error("Missing harvested sweep_features.csv: $csv. Run harvest_generic_features.jl first.")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function load_renorm_params(harvested_dir::AbstractString; renorm_csv::AbstractString="")
    csv = isempty(renorm_csv) ? joinpath(harvested_dir, "pipkin_renorm_params.csv") : renorm_csv
    isfile(csv) || error("Missing harvested pipkin_renorm_params.csv: $csv. Copy the MATLAB export there or pass an explicit third argument.")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function rowkey(drag_model, NK, hK_star, phi, De)
    if any(ismissing, (drag_model, NK, hK_star, phi, De))
        return missing
    end
    @sprintf("%s|%d|%.12g|%.12g|%.12g", String(drag_model), Int(round(NK)), float(hK_star), float(phi), float(De))
end

function with_join_key(df::DataFrame)
    out = copy(df)
    out.join_key = rowkey.(out.drag_model, out.NK, out.hK_star, out.phi, out.De)
    return out
end

function attach_renorm(df::DataFrame, renorm::DataFrame)
    lhs = with_join_key(df)
    rhs = with_join_key(renorm)

    keep = [:join_key, :Wics, :Wisc, :theta, :tilde_eta_p0, :U, :Hv, :phibar, :Debar, :Wiebar]
    rhs2 = rhs[.!ismissing.(rhs.join_key), keep]
    rhs2 = unique(rhs2, :join_key)

    out = leftjoin(lhs, rhs2; on=:join_key, matchmissing=:notequal)
    return out
end

master_curve(x) = x <= 1 ? 1.0 : x^(4/3)

# ------------------------------------------------------------
# Manual relative text helper, matching fig3 style
# ------------------------------------------------------------

function reltext!(ax, x, y, txt;
    sty,
    color=soft_gray(sty),
    fs_scale=1.0,
    align=(:left, :bottom),
)
    text!(ax, [x], [y];
        text=[txt],
        space=:relative,
        color=color,
        align=align,
        fontsize=fs_scale * sty.annotation_scale * sty.font_base_pt,
        font=:regular
    )
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir = length(args) >= 2 ? args[2] : joinpath(paper_root(), "figures")
    renorm_csv = length(args) >= 3 ? args[3] : ""
    ensure_dir(figdir)

    df = load_sweep_features(harvested_dir)
    renorm = load_renorm_params(harvested_dir; renorm_csv=renorm_csv)
    df = attach_renorm(df, renorm)

    mask = trues(nrow(df))
    mask .&= coalesce.(df.sweep_feature_kind .== "Wi_e_vs_De", false)
    mask .&= coalesce.(df.flow_family .== "caber", false)
    mask .&= coalesce.(df.drag_model .== "constant", false)
    mask .&= coalesce.(df.NK .== 3000, false)
    mask .&= coalesce.(isapprox.(df.hK_star, 0.05; atol=1e-12), false)

    d = df[mask, :]
    isempty(d) && error("No rows found for baseline FENE-P Pipkin plot.")

    wanted_phi = [0.01, 0.1, 0.5, 1.0]
    d = d[map(x -> !ismissing(x) && any(isapprox(x, p; atol=1e-12) for p in wanted_phi), d.phi), :]
    sort!(d, [:phi, :De])

    required_cols = [:Wics, :Wisc, :theta, :tilde_eta_p0, :U, :Hv, :phibar, :Debar, :Wiebar]
    for c in required_cols
        any(ismissing, d[!, c]) && error("Merged data contain missing renorm values in column $(c). Check pipkin_renorm_params.csv coverage.")
    end

    phis = unique(d.phi)

    φbar_vals = unique(round.(d.phibar; digits=12))
    println("[TNW] Using renormalization from CSV:")
    println("       renorm rows matched = $(nrow(d))")
    println("       unique phibar values = $(collect(φbar_vals))")
    for phi in phis
        di = d[isapprox.(d.phi, phi; atol=1e-12), :]
        println("       phi = $(phi): Wiebar = $(di.Wiebar[1]), Debar = $(di.Debar[1])")
    end

    sty = style()
    PaperFig.activate_theme!(sty)

    fig_w_cm = 17.0
    fig_h_cm = 8.5
    fig_pad_pt = (1, 1, 1, 5)

    fig = Figure(
        size=(PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor=:white,
        figure_padding=fig_pad_pt,
    )

    axA = Axis(fig[1, 1], xlabel=L"\mathrm{De}", ylabel=L"\mathrm{Wi}_{\mathrm{e}}")
    axB = Axis(fig[1, 2], xlabel=L"\mathrm{De}/\overline{\mathrm{De}}", ylabel=L"\mathrm{Wi}_{\mathrm{e}}/\overline{\mathrm{Wi}}_{\mathrm{e}}")

    colsize!(fig.layout, 1, Fixed(PaperFig.cm_to_pt(6.5)))
    colsize!(fig.layout, 2, Fixed(PaperFig.cm_to_pt(6.5)))
    colgap!(fig.layout, PaperFig.cm_to_pt(0.45))

    DEBUG_GUIDES = false
    if DEBUG_GUIDES
        relative_guide_grid!(axA; color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
        relative_guide_grid!(axB; color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
    end

    cols = reverse(fenep_gray_series(length(phis); sty=sty))
    phi_to_col = Dict(phi => cols[i] for (i, phi) in enumerate(phis))

    label_pos = Dict(
        0.01 => (5, 0.88, 1.25),
        0.1  => (6, 0.85, 1.25),
        0.5  => (8, 0.85, 1.30),
        1.0  => (9, 1.05, 0.85),
    )

    xminA, xmaxA = 1e-2, 1.8e2
    yminA, ymaxA = 1e-1, 1e1

    xminB, xmaxB = 1e-3, 10.0
    yminB, ymaxB = 0.1, 10.0

    # --------------------------------------------------------
    # Panel A and Panel B data
    # --------------------------------------------------------
    for phi in phis
        di = d[isapprox.(d.phi, phi; atol=1e-12), :]
        sort!(di, :De)
        col = phi_to_col[phi]

        # Panel A: unnormalized Pipkin curve
        lines!(axA, di.De, di.Wi_e; color=col, linewidth=sty.line_lw)
        scatter!(axA, di.De, di.Wi_e;
            color=col,
            markersize=sty.marker_size,
            strokecolor=darken(col, 0.15),
            strokewidth=0.6
        )

        if haskey(label_pos, phi)
            j, xfac, yfac = label_pos[phi]
            j = clamp(j, 1, nrow(di))
            text!(axA, [di.De[j] * xfac], [di.Wi_e[j] * yfac];
                text=[PaperFig.pretty_num(phi)],
                color=darken(col, 0.08),
                align=(:left, :center),
                fontsize=sty.annotation_scale * sty.font_base_pt,
                font=:regular
            )
        end

        # Panel B: CSV-normalized master plot
        xnorm = di.De ./ di.Debar
        ynorm = di.Wi_e ./ di.Wiebar

        lines!(axB, xnorm, ynorm; color=col, linewidth=sty.line_lw)
        scatter!(axB, xnorm, ynorm;
            color=col,
            markersize=sty.marker_size,
            strokecolor=darken(col, 0.15),
            strokewidth=0.6
        )
    end

    # --------------------------------------------------------
    # Panel A guides/annotations
    # --------------------------------------------------------
    hlines!(axA, [2/3];
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    text!(axA, [1.35e-2], [2/3 * 0.96];
        text=[L"2/3"],
        color=sty.ref_dark,
        align=(:left, :top),
        fontsize=sty.annotation_scale * sty.font_base_pt,
        font=:regular
    )

    text!(axA, [0.3], [3.65];
        text=[L"\phi \,=\,"],
        color=soft_gray(sty),
        align=(:left, :top),
        fontsize=sty.annotation_scale * sty.font_base_pt,
        font=:regular
    )

    NK_label = Int(round(first(unique(d.NK))))
    parameter_block!(axA, [
            "FENE-P",
            L"N_{\mathrm{K}} = %$(PaperFig.pretty_num(NK_label))"
        ];
        sty=sty, x=0.33, y0=0.22, dy=0.075, color=soft_gray(sty)
    )

    # --------------------------------------------------------
    # Panel B guides/annotations
    # --------------------------------------------------------
    hlines!(axB, [1.0];
        color=sty.ref_dark,
        linestyle=:dot,
        linewidth=0.85 * sty.line_lw / 2
    )

    xmaster = 10 .^ range(0.0, log10(xmaxB); length=400)
    ymaster = [master_curve(x) for x in xmaster]

    lines!(axB, xmaster, ymaster;
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    reltext!(axA, 0.02, 0.92, "A"; sty=sty, color=:black, fs_scale=1.35)
    reltext!(axB, 0.02, 0.92, "B"; sty=sty, color=:black, fs_scale=1.35)

    apply_log10_x!(axA, xminA, xmaxA)
    apply_log10_y!(axA, yminA, ymaxA)

    apply_log10_x!(axB, xminB, xmaxB)
    apply_log10_y!(axB, yminB, ymaxB)

    out_pdf = joinpath(figdir, "fig2_fenep_pipkin.pdf")
    out_png = joinpath(figdir, "fig2_fenep_pipkin.png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)

    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
end

main(ARGS)
