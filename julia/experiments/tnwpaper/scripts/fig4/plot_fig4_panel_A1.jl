#!/usr/bin/env julia
using CairoMakie
using DataFrames
include(joinpath(@__DIR__, "fig4_style.jl"))
using .Fig4Style

function main(args)


    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir = length(args) >= 2 ? args[2] : default_figdir()

    sty = style_bundle()
    curves = load_curves(harvested_dir)
    feats = load_features_selected(harvested_dir)


    AXIS_H_CM = 7.3
    TOP_CM = 0.66     # enough for ylabel at top
    BOTTOM_CM = 0.52  # enough for ylabel + tick labels

    AXIS_W_CM = 4.85
    LEFT_CM = 0.7   # enough for xlabel + tick labels
    RIGHT_CM = 0.93

    fig, ax = panel_fig_top(
        axis_w_cm=AXIS_W_CM,
        axis_h_cm=AXIS_H_CM,
        left_gutter_cm=LEFT_CM,
        right_gutter_cm=RIGHT_CM,
        top_gutter_cm=TOP_CM,
        bottom_gutter_cm=BOTTOM_CM,
        show_ylabel=true,
        show_yticklabels=true,
    )
    wicol = curve_colsym(curves, [:Wi, :wi])
    etacol = curve_colsym(curves, [:etaE1, :etaP])

    phi = 0.01
    Devals = [0.1, 1.0, 3.0, 10.0]
    drag_model = "c2d2"
    NK = 3000
    hK = 0.05

    stycurve = pick_curve_run(curves; flow_family="styext", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi)
    Wi_sty = Vector{Float64}(stycurve[!, wicol])
    eta_sty = Vector{Float64}(stycurve[!, etacol])

    good = isfinite.(Wi_sty) .& (Wi_sty .> 0) .& isfinite.(eta_sty) .& (eta_sty .> 0)
    Wi_sty = Wi_sty[good]
    eta_sty = eta_sty[good]

    cols = c2d2_cols(length(Devals); sty=sty)
    wis = [Wi_sty]

    lines!(ax, Wi_sty, eta_sty; color=sty.ref_dark, linewidth=0.9 * sty.line_lw, linestyle=:dot)

    for (i, De) in enumerate(Devals)
        di = pick_curve_run(curves; flow_family="caber", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi, De=De)
        fr = pick_feature_row(feats, di.run_id[1])
        idx = clamp(Int(fr.feature_idx), 1, nrow(di))
        Wi = Vector{Float64}(di[!, wicol])
        eta_p = Vector{Float64}(di[!, etacol])
        push!(wis, Wi)
        lines!(ax, Wi, eta_p; color=cols[i], linewidth=sty.line_lw, label=L"\mathrm{De} = %$(pretty_num(De))")
        scatter!(ax, [Wi[idx]], [eta_p[idx]]; color=cols[i], markersize=sty.marker_size, strokecolor=darken(cols[i], 0.15), strokewidth=0.6)
    end

    xminX, xmaxX = positive_bounds(wis...; pad_low=0.08, pad_high=0.08)
    xminX = min(xminX, 0.1)
    apply_log10_x!(ax, xminX, xmaxX)
    apply_log10_y!(ax, 1.0, 1_000_000.0)
    vlines!(ax, [2 / 3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)

    reltext!(ax, 0.03, 0.93, "A (i)"; sty=sty, color=:black, fs_scale=1.35)
    reltext!(ax, 0.08, 0.40, L"\phi = 0.01"; sty=sty)
    reltext!(ax, 0.50, 0.90, L"2/3"; sty=sty, color=sty.ref_dark)
    reltext!(ax, 0.08, 0.30, L"N_{\mathrm{K}} = 3000"; sty=sty)
    reltext!(ax, 0.08, 0.20, L"h_{\mathrm{K}}^{\ast} = 0.05"; sty=sty)

    save_panel(fig, figdir, "fig4_panel_A1")
end

main(ARGS)
