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

    fig, ax = panel_fig_square(
        axis_w_cm=AXIS_W_CM,
        axis_h_cm=AXIS_H_CM,
        left_gutter_cm=LEFT_CM,
        right_gutter_cm=RIGHT_CM,
        top_gutter_cm=TOP_CM,
        bottom_gutter_cm=BOTTOM_CM,
        show_ylabel=true,
        show_yticklabels=true,
        xlabel=L"\mathrm{De}/\overline{\mathrm{De}}", 
        ylabel=L"\mathrm{Wi}_{\mathrm{e}}/\overline{\mathrm{Wi}}_{\mathrm{e}}"
    )


    dbase = subset_pipkin(sf; drag_model="c2d2", NK=3000, hK_star=0.05)
    wanted_phi = [0.01, 0.1, 0.5, 1.0]
    dbase = dbase[map(x -> !ismissing(x) && any(isapprox(x, p; atol=1e-12) for p in wanted_phi), dbase.phi), :]
    sort!(dbase, [:phi, :De])
    phis = unique(dbase.phi)
    cols = c2d2_cols(length(phis); sty=sty)
    phi_to_col = Dict(phi => cols[i] for (i, phi) in enumerate(phis))
    label_pos = baseline_label_pos()

    for phi in phis
        di = dbase[approx_mask(dbase.phi, phi), :]
        sort!(di, :De)
        x = di.De ./ di.Debar
        y = di.Wi_e ./ di.Wiebar
        col = phi_to_col[phi]
        lines!(ax, x, y; color=col, linewidth=sty.line_lw)
        scatter!(ax, x, y; color=col, markersize=sty.marker_size, strokecolor=darken(col, 0.15), strokewidth=0.6)

        if haskey(label_pos, phi)
            j, xfac, yfac = label_pos[phi]
            j = clamp(j, 1, nrow(di))
            text!(ax, [x[j] * xfac], [y[j] * yfac];
                text=[pretty_num(phi)],
                color=darken(col, 0.08), align=(:left, :center),
                fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
        end
    end

    combos = [(3000, 0.05), (3000, 0.25), (1000, 0.05), (10000, 0.05)]
    dvar = subset_pipkin(sf; drag_model="c2d2", phi=0.1)
    dvar = dvar[map(eachrow(dvar)) do r
        any((Int(round(r.NK)) == NK && isapprox(r.hK_star, hK; atol=1e-12)) for (NK, hK) in combos)
    end, :]
    sort!(dvar, [:NK, :hK_star, :De])
    cols2 = c2d2_cols(length(combos); sty=sty)

    for (i, (NK, hK)) in enumerate(combos)
        di = dvar[(coalesce.(dvar.NK .== NK, false)) .& (coalesce.(isapprox.(dvar.hK_star, hK; atol=1e-12), false)), :]
        sort!(di, :De)
        isempty(di) && continue
        x = di.De ./ di.Debar
        y = di.Wi_e ./ di.Wiebar
        st = combo_style(NK, hK)
        lines!(ax, x, y; color=cols2[i], linewidth=sty.line_lw, linestyle=st.linestyle, label=combo_label(NK, hK))
        scatter!(ax, x, y; color=cols2[i], marker=st.marker, markersize=sty.marker_size, strokecolor=darken(cols2[i], 0.15), strokewidth=0.6)
    end

    hlines!(ax, [1.0]; color=sty.ref_dark, linestyle=:dot, linewidth=0.85 * sty.line_lw / 2)
    xmaster = 10 .^ range(0.0, log10(30.0); length=400)
    lines!(ax, xmaster, [master_curve(x) for x in xmaster]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)
    reltext!(ax, 0.02, 0.92, "C"; sty=sty, color=:black, fs_scale=1.35)

    apply_log10_x!(ax, 1e-3, 10.0)
    apply_log10_y!(ax, 0.1, 10.0)

    save_panel(fig, figdir, "fig4_panel_C")
end

main(ARGS)
