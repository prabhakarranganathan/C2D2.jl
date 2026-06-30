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
    LEFT_CM = 0.33   # enough for xlabel + tick labels
    RIGHT_CM = 0.33

    fig, ax = panel_fig_top(
        axis_w_cm=AXIS_W_CM,
        axis_h_cm=AXIS_H_CM,
        left_gutter_cm=LEFT_CM,
        right_gutter_cm=RIGHT_CM,
        top_gutter_cm=TOP_CM,
        bottom_gutter_cm=BOTTOM_CM,
        show_ylabel=false,
        show_yticklabels=false,
    )
    wicol = curve_colsym(curves, [:Wi, :wi])
    etacol = curve_colsym(curves, [:etaE1, :etaP])

    phi = 0.1
    De = 1.0
    drag_model = "c2d2"
    combos = [(3000, 0.05), (3000, 0.25), (1000, 0.05), (10000, 0.05)]
    cols = c2d2_cols(length(combos); sty=sty)

    wis = Vector{Vector{Float64}}()
    etas = Vector{Vector{Float64}}()

    for (i, (NK, hK)) in enumerate(combos)
        di = pick_curve_run(curves; flow_family="caber", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi, De=De)
        fr = pick_feature_row(feats, di.run_id[1])
        idx = clamp(Int(fr.feature_idx), 1, nrow(di))
        Wi = Vector{Float64}(di[!, wicol])
        eta_p = Vector{Float64}(di[!, etacol])
        push!(wis, Wi)
        push!(etas, eta_p)

        st = combo_style(NK, hK)
        lines!(ax, Wi, eta_p; color=cols[i], linewidth=sty.line_lw, linestyle=st.linestyle, label=combo_label(NK, hK))
        scatter!(ax, [Wi[idx]], [eta_p[idx]]; color=cols[i], marker=st.marker, markersize=sty.marker_size, strokecolor=darken(cols[i], 0.15), strokewidth=0.6)
    end

    xminX, xmaxX = positive_bounds(wis...; pad_low=0.08, pad_high=0.08)
    xminX = min(xminX, 0.1)
    yminY, ymaxY = positive_bounds(etas...; pad_low=0.08, pad_high=0.08)
    apply_log10_x!(ax, xminX, xmaxX)
    apply_log10_y!(ax, yminY, ymaxY)
    vlines!(ax, [2 / 3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)

    reltext!(ax, 0.03, 0.93, "A (iii)"; sty=sty, color=:black, fs_scale=1.35)
    reltext!(ax, 0.08, 0.40, L"\phi = 0.1"; sty=sty)
    reltext!(ax, 0.08, 0.30, L"\mathrm{De} = 1"; sty=sty)
    reltext!(ax, 0.45, 0.90, L"2/3"; sty=sty, color=sty.ref_dark)

    save_panel(fig, figdir, "fig4_panel_A3")
end

main(ARGS)
