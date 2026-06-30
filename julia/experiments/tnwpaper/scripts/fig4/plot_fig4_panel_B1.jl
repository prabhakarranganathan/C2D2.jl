#!/usr/bin/env julia
using CairoMakie
using DataFrames
include(joinpath(@__DIR__, "fig4_style.jl"))
using .Fig4Style

function baseline_label_pos()
    Dict(
        0.01 => (5, 0.88, 1.25),
        0.1  => (6, 0.85, 1.25),
        0.5  => (8, 0.85, 1.30),
        1.0  => (9, 1.05, 0.85),
    )
end

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir        = length(args) >= 2 ? args[2] : default_figdir()
    renorm_csv    = length(args) >= 3 ? args[3] : ""

    sty = style_bundle()
    sf  = attach_renorm(load_sweep_features(harvested_dir), load_renorm_params(harvested_dir; renorm_csv=renorm_csv))

    AXIS_H_CM = 7.3
    TOP_CM = 0.66     # enough for ylabel at top
    BOTTOM_CM = 0.52  # enough for ylabel + tick labels

    AXIS_W_CM = 7.1
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

    d = subset_pipkin(sf; drag_model="c2d2", NK=3000, hK_star=0.05)
    wanted_phi = [0.01, 0.1, 0.5, 1.0]
    d = d[map(x -> !ismissing(x) && any(isapprox(x, p; atol=1e-12) for p in wanted_phi), d.phi), :]
    sort!(d, [:phi, :De])

    phis = unique(d.phi)
    cols = c2d2_cols(length(phis); sty=sty)
    phi_to_col = Dict(phi => cols[i] for (i, phi) in enumerate(phis))
    label_pos = baseline_label_pos()

    for phi in phis
        di = d[approx_mask(d.phi, phi), :]
        sort!(di, :De)
        col = phi_to_col[phi]
        lines!(ax, di.De, di.Wi_e; color=col, linewidth=sty.line_lw)
        scatter!(ax, di.De, di.Wi_e; color=col, markersize=sty.marker_size, strokecolor=darken(col, 0.15), strokewidth=0.6)

        if haskey(label_pos, phi)
            j, xfac, yfac = label_pos[phi]
            j = clamp(j, 1, nrow(di))
            text!(ax, [di.De[j] * xfac], [di.Wi_e[j] * yfac];
                text=[pretty_num(phi)],
                color=darken(col, 0.08), align=(:left, :center),
                fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
        end
    end

    hlines!(ax, [2/3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)
    text!(ax, [1.35e-2], [2/3 * 0.96]; text=[L"2/3"], color=sty.ref_dark, align=(:left, :top), fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
    reltext!(ax, 0.02, 0.92, "B (i)"; sty=sty, color=:black, fs_scale=1.35)
    reltext!(ax, 0.40, 0.50, L"\phi ="; sty=sty)
    reltext!(ax, 0.08, 0.80, L"N_{\mathrm{K}} = 3000"; sty=sty)
    reltext!(ax, 0.08, 0.70, L"h_{\mathrm{K}}^{\ast} = 0.05"; sty=sty)

    apply_log10_x!(ax, 1e-2, 1.8e2)
    apply_log10_y!(ax, 1e-1, 1e1)

    save_panel(fig, figdir, "fig4_panel_B1")
end

main(ARGS)
