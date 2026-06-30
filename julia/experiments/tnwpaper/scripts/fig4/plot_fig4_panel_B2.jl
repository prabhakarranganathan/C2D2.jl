#!/usr/bin/env julia
using CairoMakie
using DataFrames
include(joinpath(@__DIR__, "fig4_style.jl"))
using .Fig4Style

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

    combos = [(3000, 0.05), (3000, 0.25), (1000, 0.05), (10000, 0.05)]
    d = subset_pipkin(sf; drag_model="c2d2", phi=0.1)
    d = d[map(eachrow(d)) do r
        any((Int(round(r.NK)) == NK && isapprox(r.hK_star, hK; atol=1e-12)) for (NK, hK) in combos)
    end, :]
    sort!(d, [:NK, :hK_star, :De])

    cols = c2d2_cols(length(combos); sty=sty)

    for (i, (NK, hK)) in enumerate(combos)
        di = d[(coalesce.(d.NK .== NK, false)) .& (coalesce.(isapprox.(d.hK_star, hK; atol=1e-12), false)), :]
        sort!(di, :De)
        isempty(di) && continue
        st = combo_style(NK, hK)
        lines!(ax, di.De, di.Wi_e; color=cols[i], linewidth=sty.line_lw, linestyle=st.linestyle, label=combo_label(NK, hK))
        scatter!(ax, di.De, di.Wi_e; color=cols[i], marker=st.marker, markersize=sty.marker_size, strokecolor=darken(cols[i], 0.15), strokewidth=0.6)
    end

    hlines!(ax, [2/3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)
    reltext!(ax, 0.02, 0.92, "B (ii)"; sty=sty, color=:black, fs_scale=1.35)
    reltext!(ax, 0.06, 0.80, L"\phi = 0.1"; sty=sty)
    reltext!(ax, 0.30, 0.50, L"(N_{\mathrm{K}}, h_{\mathrm{K}}^{\ast}) ="; sty=sty)
    reltext!(ax, 0.50, 0.50, L"(1000, 0.05)"; sty=sty)
    reltext!(ax, 0.40, 0.25, L"(3000, 0.05)"; sty=sty)
    reltext!(ax, 0.30, 0.15, L"(10000, 0.05)"; sty=sty)
    reltext!(ax, 0.30, 0.05, L"(3000, 0.25)"; sty=sty)

    apply_log10_x!(ax, 1e-2, 1.8e2)
    apply_log10_y!(ax, 1e-1, 1e1)

    save_panel(fig, figdir, "fig4_panel_B2")
end

main(ARGS)
