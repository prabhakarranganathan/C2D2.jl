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

# ------------------------------------------------------------
# data loading
# ------------------------------------------------------------

function load_curves(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "curves.csv")
    isfile(csv) || error("Missing harvested curves.csv: $csv. Run harvest_generic_features.jl first.")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function load_features_selected(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "features_selected.csv")
    isfile(csv) || error("Missing harvested features_selected.csv: $csv. Run harvest_generic_features.jl first.")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

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

# ------------------------------------------------------------
# generic helpers
# ------------------------------------------------------------

function approx_mask(v, x; atol=1e-12)
    map(v) do y
        if ismissing(y)
            false
        elseif y isa Real
            isfinite(y) && isapprox(Float64(y), x; atol=atol)
        else
            false
        end
    end
end

function curve_colsym(df::DataFrame, candidates::Vector{Symbol})
    for c in candidates
        c in propertynames(df) && return c
    end
    error("None of the candidate columns $(candidates) found in DataFrame.")
end

function reltext!(ax, x, y, txt; sty, color=soft_gray(sty), fs_scale=1.0, align=(:left, :bottom))
    text!(ax, [x], [y];
        text=[txt],
        space=:relative,
        color=color,
        align=align,
        fontsize=fs_scale * sty.annotation_scale * sty.font_base_pt,
        font=:regular,
    )
end

function positive_bounds(vectors...; pad_low=0.08, pad_high=0.08)
    vals = Float64[]
    for v in vectors
        append!(vals, [x for x in v if isfinite(x) && x > 0])
    end
    isempty(vals) && error("No positive finite values found.")
    vmin = minimum(vals)
    vmax = maximum(vals)
    ymin = 10.0^(floor(log10(vmin)) - pad_low)
    ymax = 10.0^(ceil(log10(vmax)) + pad_high)
    return ymin, ymax
end

function pick_curve_run(curves::DataFrame; flow_family::String, drag_model::String, NK::Int, hK_star::Float64, phi::Float64, De::Union{Nothing,Float64}=nothing)
    d = curves
    mask = trues(nrow(d))
    hasproperty(d, :flow_family) && (mask .&= coalesce.(d.flow_family .== flow_family, false))
    hasproperty(d, :drag_model) && (mask .&= coalesce.(d.drag_model .== drag_model, false))
    hasproperty(d, :NK) && (mask .&= coalesce.(d.NK .== NK, false))
    hasproperty(d, :hK_star) && (mask .&= coalesce.(isapprox.(d.hK_star, hK_star; atol=1e-12), false))

    if flow_family == "styext"
        if hasproperty(d, :setting_var) && hasproperty(d, :setting_value)
            mask .&= coalesce.(string.(d.setting_var) .== "phi", false)
            mask .&= approx_mask(d.setting_value, phi)
        elseif hasproperty(d, :phi)
            mask .&= approx_mask(coalesce.(d.phi, NaN), phi)
        elseif hasproperty(d, :phi_nominal)
            mask .&= approx_mask(coalesce.(d.phi_nominal, NaN), phi)
        else
            error("No concentration metadata found for styext rows.")
        end
    else
        if hasproperty(d, :phi)
            mask .&= approx_mask(coalesce.(d.phi, NaN), phi)
        elseif hasproperty(d, :phi_nominal)
            mask .&= approx_mask(coalesce.(d.phi_nominal, NaN), phi)
        elseif hasproperty(d, :setting_var) && hasproperty(d, :setting_value)
            mask .&= coalesce.(string.(d.setting_var) .== "phi", false)
            mask .&= approx_mask(d.setting_value, phi)
        else
            error("No concentration metadata found for caber rows.")
        end
    end

    if !isnothing(De)
        if hasproperty(d, :De_run)
            mask .&= approx_mask(coalesce.(d.De_run, NaN), De)
        elseif hasproperty(d, :De)
            mask .&= approx_mask(coalesce.(d.De, NaN), De)
        else
            error("No De or De_run column found.")
        end
    end

    dd = d[mask, :]
    isempty(dd) && error("No run found for flow_family=$flow_family, drag_model=$drag_model, NK=$NK, hK_star=$hK_star, phi=$phi, De=$(De).")

    rid = unique(dd.run_id)[1]
    di = dd[dd.run_id.==rid, :]
    if :t in propertynames(di)
        sort!(di, :t)
    end
    return di
end

function pick_feature_row(features::DataFrame, run_id::AbstractString)
    f = features[features.run_id.==run_id, :]
    isempty(f) && error("No selected-feature rows found for run_id=$run_id")

    if hasproperty(f, :selection_reason)
        g = f[f.selection_reason.=="deepest_min", :]
        !isempty(g) && return g[1, :]
        g = f[f.selection_reason.=="fallback_inflexion", :]
        !isempty(g) && return g[1, :]
    end
    if hasproperty(f, :feature_type)
        g = f[f.feature_type.=="min", :]
        !isempty(g) && return g[1, :]
    end
    return f[1, :]
end

function rowkey(drag_model, NK, hK_star, phi, De)
    if any(ismissing, (drag_model, NK, hK_star, phi, De))
        return missing
    end
    @sprintf("%s|%d|%.12g|%.12g|%.12g",
        String(drag_model), Int(round(NK)), float(hK_star), float(phi), float(De))
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
    leftjoin(lhs, rhs2; on=:join_key, matchmissing=:notequal)
end

master_curve(x) = x <= 1 ? 1.0 : x^(4 / 3)

function c2d2_cols(n; sty)
    if isdefined(PaperFig, :c2d2_green_series)
        return reverse(getfield(PaperFig, :c2d2_green_series)(n; sty=sty))
    end
    fallback = [
        RGBf(0.80, 0.92, 0.82),
        RGBf(0.64, 0.84, 0.67),
        RGBf(0.46, 0.72, 0.51),
        RGBf(0.27, 0.58, 0.36),
        RGBf(0.14, 0.42, 0.24),
        RGBf(0.09, 0.31, 0.18),
        RGBf(0.06, 0.24, 0.14),
        RGBf(0.04, 0.18, 0.10),
    ]
    n <= length(fallback) || error("Fallback C2D2 palette only supports up to $(length(fallback)) entries.")
    return fallback[1:n]
end

pretty_hK(h) = @sprintf("%.2f", h)
combo_label(NK, hK) = L"(N_{\mathrm{K}},\,h_{\mathrm{K}}^{\ast}) = (%$(Int(round(NK))),\,%$(pretty_hK(hK)))"

function combo_style(NK, hK)
    nk = Int(round(Float64(NK)))
    hk = Float64(hK)

    if nk == 3000 && isapprox(hk, 0.05; atol=1e-12)
        return (linestyle=:solid,   marker=:circle)
    elseif nk == 10000 && isapprox(hk, 0.05; atol=1e-12)
        return (linestyle=:dash,    marker=:rect)
    elseif nk == 1000 && isapprox(hk, 0.05; atol=1e-12)
        return (linestyle=:dot,     marker=:utriangle)
    elseif nk == 3000 && isapprox(hk, 0.25; atol=1e-12)
        return (linestyle=:dashdot, marker=:diamond)
    else
        return (linestyle=:solid,   marker=:circle)
    end
end

# ------------------------------------------------------------
# top-row panels
# ------------------------------------------------------------

function plot_combined_panel!(ax; curves, feats, phi, Devals, sty, drag_model, NK, hK, panel_tag)
    wicol = curve_colsym(curves, [:Wi, :wi])
    etacol = curve_colsym(curves, [:etaE1, :etaP])
    Mcol = curve_colsym(curves, [:M, :E2, :Y])

    stycurve = pick_curve_run(curves; flow_family="styext", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi)
    Wi_sty = Vector{Float64}(stycurve[!, wicol])
    M_sty = Vector{Float64}(stycurve[!, Mcol])
    eta_sty = Vector{Float64}(stycurve[!, etacol])

    good = isfinite.(Wi_sty) .& (Wi_sty .> 0) .&
           isfinite.(M_sty) .& (M_sty .> 0) .&
           isfinite.(eta_sty) .& (eta_sty .> 0)

    Wi_sty = Wi_sty[good]
    M_sty = M_sty[good]
    eta_sty = eta_sty[good]

    # IMPORTANT: do NOT sort by Wi for C2D2 steady curves.
    # Preserve the original continuation/path order from curves.csv.

    cols = c2d2_cols(length(Devals); sty=sty)
    rows = NamedTuple[]
    for (i, De) in enumerate(Devals)
        di = pick_curve_run(curves; flow_family="caber", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi, De=De)
        fr = pick_feature_row(feats, di.run_id[1])
        idx = clamp(Int(fr.feature_idx), 1, nrow(di))
        Wi = Vector{Float64}(di[!, wicol])
        M = Vector{Float64}(di[!, Mcol])
        eta_p = Vector{Float64}(di[!, etacol])
        push!(rows, (De=De, col=cols[i], Wi=Wi, M=M, eta_p=eta_p, Wi_e=Wi[idx], M_e=M[idx], eta_p_e=eta_p[idx]))
    end

    lines!(ax, Wi_sty, eta_sty; color=sty.ref_dark, linewidth=0.9 * sty.line_lw, linestyle=:dot)

    for row in rows
        lines!(ax, row.Wi, row.eta_p; color=row.col, linewidth=sty.line_lw, label=L"\mathrm{De} = %$(PaperFig.pretty_num(row.De))")
        scatter!(ax, [row.Wi_e], [row.eta_p_e]; color=row.col, markersize=sty.marker_size, strokecolor=darken(row.col, 0.15), strokewidth=0.6)
    end

    xminX, xmaxX = positive_bounds(Wi_sty, [r.Wi for r in rows]...; pad_low=0.08, pad_high=0.08)
    xminX = min(xminX, 0.1)
    yminEta = 1.0
    ymaxEta = 1000000.0

    apply_log10_x!(ax, xminX, xmaxX)
    apply_log10_y!(ax, yminEta, ymaxEta)

    vlines!(ax, [2 / 3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)

    reltext!(ax, 0.03, 0.93, panel_tag; sty=sty, color=:black, fs_scale=1.35)
    reltext!(ax, 0.08, 0.40, L"\phi = %$(PaperFig.pretty_num(phi))"; sty=sty, color=soft_gray(sty))
    reltext!(ax, 0.5, 0.9, L"2/3"; sty=sty, color=sty.ref_dark)
    reltext!(ax, 0.08, 0.3, L"N_{\mathrm{K}} = %$(Int(round(NK)))"; sty=sty, color=soft_gray(sty))
    reltext!(ax, 0.08, 0.2, L"h_{\mathrm{K}}^{\ast} = %$(pretty_hK(hK))"; sty=sty, color=soft_gray(sty))

    #axislegend(ax; position=:lt, framevisible=false, labelsize=sty.annotation_scale * sty.font_base_pt)
    return nothing
end

function plot_eta_variation_panel!(ax; curves, feats, phi, De, combos, sty, drag_model, panel_tag)
    wicol = curve_colsym(curves, [:Wi, :wi])
    etacol = curve_colsym(curves, [:etaE1, :etaP])
    cols = c2d2_cols(length(combos); sty=sty)

    etas = Vector{Vector{Float64}}()
    wis = Vector{Vector{Float64}}()

    for (i, (NK, hK)) in enumerate(combos)
        di = pick_curve_run(curves; flow_family="caber", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi, De=De)
        fr = pick_feature_row(feats, di.run_id[1])
        idx = clamp(Int(fr.feature_idx), 1, nrow(di))

        Wi = Vector{Float64}(di[!, wicol])
        eta_p = Vector{Float64}(di[!, etacol])

        push!(wis, Wi)
        push!(etas, eta_p)

        col = cols[i]
        styc = combo_style(NK, hK)

        lines!(ax, Wi, eta_p;
            color=col,
            linewidth=sty.line_lw,
            linestyle=styc.linestyle,
            label=combo_label(NK, hK)
        )

        scatter!(ax, [Wi[idx]], [eta_p[idx]];
            color=col,
            marker=styc.marker,
            markersize=sty.marker_size,
            strokecolor=darken(col, 0.15),
            strokewidth=0.6
        )
    end

    xminX, xmaxX = positive_bounds(wis...; pad_low=0.08, pad_high=0.08)
    xminX = min(xminX, 0.1)
    yminY, ymaxY = positive_bounds(etas...; pad_low=0.08, pad_high=0.08)

    apply_log10_x!(ax, xminX, xmaxX)
    apply_log10_y!(ax, yminY, ymaxY)

    vlines!(ax, [2 / 3];
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    reltext!(ax, 0.03, 0.93, panel_tag; sty=sty, color=:black, fs_scale=1.35)
    reltext!(ax, 0.08, 0.40, L"\phi = 0.1"; sty=sty, color=soft_gray(sty))
    reltext!(ax, 0.08, 0.30, L"\mathrm{De} = 1"; sty=sty, color=soft_gray(sty))
    reltext!(ax, 0.45, 0.90, L"2/3"; sty=sty, color=sty.ref_dark)

    return nothing
end

# ------------------------------------------------------------
# Pipkin panels
# ------------------------------------------------------------

function baseline_label_pos()
    Dict(
        0.01 => (5, 0.88, 1.25),
        0.1 => (6, 0.85, 1.25),
        0.5 => (8, 0.85, 1.30),
        1.0 => (9, 1.05, 0.85),
    )
end

function subset_pipkin(df; drag_model, NK=nothing, hK_star=nothing, phi=nothing)
    mask = trues(nrow(df))
    mask .&= coalesce.(df.sweep_feature_kind .== "Wi_e_vs_De", false)
    mask .&= coalesce.(df.flow_family .== "caber", false)
    mask .&= coalesce.(df.drag_model .== drag_model, false)
    if NK !== nothing
        mask .&= coalesce.(df.NK .== NK, false)
    end
    if hK_star !== nothing
        mask .&= coalesce.(isapprox.(df.hK_star, hK_star; atol=1e-12), false)
    end
    if phi !== nothing
        mask .&= approx_mask(coalesce.(df.phi, NaN), phi)
    end
    return df[mask, :]
end

function plot_pipkin_baseline!(axU, axN; d, sty, panel_tag_left="B (i)", panel_tag_right="C")
    wanted_phi = [0.01, 0.1, 0.5, 1.0]
    dd = d[map(x -> !ismissing(x) && any(isapprox(x, p; atol=1e-12) for p in wanted_phi), d.phi), :]
    sort!(dd, [:phi, :De])
    phis = unique(dd.phi)
    cols = c2d2_cols(length(phis); sty=sty)
    phi_to_col = Dict(phi => cols[i] for (i, phi) in enumerate(phis))
    label_pos = baseline_label_pos()

    for phi in phis
        di = dd[approx_mask(dd.phi, phi), :]
        sort!(di, :De)
        col = phi_to_col[phi]

        lines!(axU, di.De, di.Wi_e; color=col, linewidth=sty.line_lw)
        scatter!(axU, di.De, di.Wi_e; color=col, markersize=sty.marker_size, strokecolor=darken(col, 0.15), strokewidth=0.6)

        xnorm = di.De ./ di.Debar
        ynorm = di.Wi_e ./ di.Wiebar
        lines!(axN, xnorm, ynorm; color=col, linewidth=sty.line_lw)
        scatter!(axN, xnorm, ynorm; color=col, markersize=sty.marker_size, strokecolor=darken(col, 0.15), strokewidth=0.6)

        if haskey(label_pos, phi)
            j, xfac, yfac = label_pos[phi]
            j = clamp(j, 1, nrow(di))
            text!(axU, [di.De[j] * xfac], [di.Wi_e[j] * yfac];
                text=[PaperFig.pretty_num(phi)],
                color=darken(col, 0.08), align=(:left, :center),
                fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)

            xn = xnorm[j] * min(xfac, 1.2)
            yn = ynorm[j] * yfac
            #text!(axN, [xn], [yn];
               # text=[PaperFig.pretty_num(phi)],
                #color=darken(col, 0.08), align=(:left, :center),
                #fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
        end
    end

    hlines!(axU, [2 / 3]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)
    text!(axU, [1.35e-2], [2 / 3 * 0.96]; text=[L"2/3"], color=sty.ref_dark, align=(:left, :top), fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)
    #text!(axU, [0.3], [3.65]; text=[L"\phi \,=\,"], color=soft_gray(sty), align=(:left, :top), fontsize=sty.annotation_scale * sty.font_base_pt, font=:regular)

    #parameter_block!(axU, [L"N_{\mathrm{K}} = 3000", L"h_{\mathrm{K}}^{\ast} = 0.05"];
    #    sty=sty, x=0.33, y0=0.24, dy=0.072, color=soft_gray(sty))

    hlines!(axN, [1.0]; color=sty.ref_dark, linestyle=:dot, linewidth=0.85 * sty.line_lw / 2)
    xmaster = 10 .^ range(0.0, log10(30.0); length=400)
    lines!(axN, xmaster, [master_curve(x) for x in xmaster]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)

    reltext!(axU, 0.02, 0.92, panel_tag_left; sty=sty, color=:black, fs_scale=1.35)
    reltext!(axU, 0.4, 0.5, L"\phi ="; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.08, 0.8, L"N_{\mathrm{K}} = 3000"; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.08, 0.7, L"h_{\mathrm{K}}^{\ast} = 0.05"; sty=sty, color=soft_gray(sty))   
    reltext!(axN, 0.02, 0.92, panel_tag_right; sty=sty, color=:black, fs_scale=1.35)


    apply_log10_x!(axU, 1e-2, 1.8e2)
    apply_log10_y!(axU, 1e-1, 1e1)
    apply_log10_x!(axN, 1e-3, 10.0)
    apply_log10_y!(axN, 0.1, 10.0)
    return nothing
end

function combo_key(NK, hK)
    return (Int(round(Float64(NK))), Float64(hK))
end

function plot_pipkin_variation!(axU, axN; d, sty, combos, panel_tag_left="B (ii)")
    dd = copy(d)
    sort!(dd, [:NK, :hK_star, :De])

    cols = c2d2_cols(length(combos); sty=sty)
    combo_to_col = Dict(combo_key(c[1], c[2]) => cols[i] for (i, c) in enumerate(combos))

    for (NK, hK) in combos
        di = dd[
            (coalesce.(dd.NK .== NK, false)) .&
            (coalesce.(isapprox.(dd.hK_star, hK; atol=1e-12), false)),
            :
        ]
        sort!(di, :De)
        isempty(di) && continue

        col = combo_to_col[combo_key(NK, hK)]
        styc = combo_style(NK, hK)

        lines!(axU, di.De, di.Wi_e;
            color=col,
            linewidth=sty.line_lw,
            linestyle=styc.linestyle,
            label=combo_label(NK, hK)
        )
        scatter!(axU, di.De, di.Wi_e;
            color=col,
            marker=styc.marker,
            markersize=sty.marker_size,
            strokecolor=darken(col, 0.15),
            strokewidth=0.6
        )

        xnorm = di.De ./ di.Debar
        ynorm = di.Wi_e ./ di.Wiebar

        lines!(axN, xnorm, ynorm;
            color=col,
            linewidth=sty.line_lw,
            linestyle=styc.linestyle,
            label=combo_label(NK, hK)
        )
        scatter!(axN, xnorm, ynorm;
            color=col,
            marker=styc.marker,
            markersize=sty.marker_size,
            strokecolor=darken(col, 0.15),
            strokewidth=0.6
        )
    end

    hlines!(axU, [2 / 3];
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    reltext!(axU, 0.02, 0.92, panel_tag_left; sty=sty, color=:black, fs_scale=1.35)
    reltext!(axU, 0.06, 0.80, L"\phi = 0.1"; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.30, 0.50, L"(N_{\mathrm{K}}, h_{\mathrm{K}}^{\ast}) ="; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.50, 0.50, L"(1000, 0.05)"; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.40, 0.25, L"(3000, 0.05)"; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.30, 0.15, L"(10000, 0.05)"; sty=sty, color=soft_gray(sty))
    reltext!(axU, 0.30, 0.05, L"(3000, 0.25)"; sty=sty, color=soft_gray(sty))

    apply_log10_x!(axU, 1e-2, 1.8e2)
    apply_log10_y!(axU, 1e-1, 1e1)

    return nothing
end

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir = length(args) >= 2 ? args[2] : joinpath(paper_root(), "figures")
    renorm_csv = length(args) >= 3 ? args[3] : ""
    ensure_dir(figdir)

    curves = load_curves(harvested_dir)
    feats = load_features_selected(harvested_dir)
    sf = attach_renorm(load_sweep_features(harvested_dir), load_renorm_params(harvested_dir; renorm_csv=renorm_csv))

    # Make sure the normalized data are available for the curves we want.
    required_cols = [:Debar, :Wiebar]
    for c in required_cols
        any(ismissing, sf[!, c]) && @warn "Some sweep-feature rows have missing $(c); subset panels may fail if they rely on those rows."
    end

    sty = style()
    PaperFig.activate_theme!(sty)

    drag_model = "c2d2"
    NK0 = 3000
    hK0 = 0.05

    DeA1 = [0.1, 1.0, 3.0, 10.0]
    DeA2 = [0.1, 1.0, 10.0, 30.0, 100.0]
    combos = [(3000, 0.05), (3000, 0.25), (1000, 0.05), (10000, 0.05)]

    # Figure geometry.
    sq_cm = 6.5
    top_w_cm = 4.8
    row_h_cm = sq_cm
    top_gap_cm = 0.4
    mid_gap_cm = 0.45
    row_gap_cm = 0.60
    fig_w_cm = 17.0
    fig_h_cm = 3 * row_h_cm + 2 * row_gap_cm + 1.0

    fig = Figure(
        size=(PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor=:white,
        figure_padding=(2, 2, 2, 6),
    )

    # --- Top row: 3 non-square panels ---
    glA = GridLayout(fig[1, 1])

    axA1 = Axis(glA[1, 1], xlabel=L"\mathrm{Wi}", ylabel=L"\tilde{\eta}_{\mathrm{p}}")
    axA2 = Axis(glA[1, 2], xlabel=L"\mathrm{Wi}", ylabel=L"\tilde{\eta}_{\mathrm{p}}")
    axA3 = Axis(glA[1, 3], xlabel=L"\mathrm{Wi}", ylabel=L"\tilde{\eta}_{\mathrm{p}}")
    axA2.ylabelvisible = false
    axA3.ylabelvisible = false
    axA2.yticklabelsvisible = false
    axA3.yticklabelsvisible = false

    colsize!(glA, 1, Fixed(PaperFig.cm_to_pt(top_w_cm)))
    colsize!(glA, 2, Fixed(PaperFig.cm_to_pt(top_w_cm)))
    colsize!(glA, 3, Fixed(PaperFig.cm_to_pt(top_w_cm)))
    colgap!(glA, PaperFig.cm_to_pt(top_gap_cm))

    # --- Middle row: 2 square panels ---
    glB = GridLayout(fig[2, 1])

    axB1 = Axis(glB[1, 1], xlabel=L"\mathrm{De}", ylabel=L"\mathrm{Wi}_{\mathrm{e}}")
    axB2 = Axis(glB[1, 2], xlabel=L"\mathrm{De}", ylabel=L"\mathrm{Wi}_{\mathrm{e}}")
    colsize!(glB, 1, Fixed(PaperFig.cm_to_pt(sq_cm)))
    colsize!(glB, 2, Fixed(PaperFig.cm_to_pt(sq_cm)))
    colgap!(glB, PaperFig.cm_to_pt(mid_gap_cm))

    # --- Bottom row: single square panel, centered ---
    glC = GridLayout(fig[3, 1])
    axC = Axis(glC[1, 1],
        xlabel=L"\mathrm{De}/\overline{\mathrm{De}}",
        ylabel=L"\mathrm{Wi}_{\mathrm{e}}/\overline{\mathrm{Wi}}_{\mathrm{e}}"
    )
    colsize!(glC, 1, Fixed(PaperFig.cm_to_pt(sq_cm)))

    DEBUG_GUIDES = false

    if DEBUG_GUIDES
        for ax in (axA1, axA2, axA3, axB1, axB2, axC)
            relative_guide_grid!(ax; color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
        end
    end

    rowgap!(fig.layout, PaperFig.cm_to_pt(row_gap_cm))

    # --- Top row plotting ---
    plot_combined_panel!(axA1; curves=curves, feats=feats, phi=0.01, Devals=DeA1, sty=sty, drag_model=drag_model, NK=NK0, hK=hK0, panel_tag="A (i)")
    plot_combined_panel!(axA2; curves=curves, feats=feats, phi=0.5, Devals=DeA2, sty=sty, drag_model=drag_model, NK=NK0, hK=hK0, panel_tag="A (ii)")
    plot_eta_variation_panel!(axA3; curves=curves, feats=feats, phi=0.1, De=1.0, combos=combos, sty=sty, drag_model=drag_model, panel_tag="A (iii)")

    # --- Middle/Bottom row data subsets ---
    dbase = subset_pipkin(sf; drag_model=drag_model, NK=NK0, hK_star=hK0)
    dvar = subset_pipkin(sf; drag_model=drag_model, phi=0.1)
    dvar = dvar[map(eachrow(dvar)) do r
        any((Int(round(r.NK)) == NK && isapprox(r.hK_star, hK; atol=1e-12)) for (NK, hK) in combos)
    end, :]

    for c in [:Debar, :Wiebar]
        any(ismissing, dbase[!, c]) && error("Baseline C2D2 rows contain missing $(c). Check pipkin_renorm_params.csv coverage.")
        any(ismissing, dvar[!, c]) && error("Variation C2D2 rows contain missing $(c). Check pipkin_renorm_params.csv coverage.")
    end

    plot_pipkin_baseline!(axB1, axC; d=dbase, sty=sty, panel_tag_left="B (i)", panel_tag_right="C")
    plot_pipkin_variation!(axB2, axC; d=dvar, sty=sty, combos=combos, panel_tag_left="B (ii)")

    hlines!(axC, [1.0]; color=sty.ref_dark, linestyle=:dot, linewidth=0.85 * sty.line_lw / 2)
    xmaster = 10 .^ range(0.0, log10(30.0); length=400)
    lines!(axC, xmaster, [master_curve(x) for x in xmaster]; color=sty.ref_dark, linestyle=:dash, linewidth=0.95 * sty.line_lw / 2)


    # small legend only for dashed variation family; solid phi family is directly labelled.
    #axislegend(axC; position=:lb, framevisible=false, labelsize=0.88 * sty.annotation_scale * sty.font_base_pt)
    #reltext!(axC, 0.06, 0.84, "solid: baseline \u03d5-family"; sty=sty, color=soft_gray(sty))
    #reltext!(axC, 0.06, 0.76, "dashed: (N_K, h^*_K) at \u03d5 = 0.1"; sty=sty, color=soft_gray(sty))
    #reltext!(axC, 0.06, 0.68, "C2D2"; sty=sty, color=soft_gray(sty), fs_scale=1.05)

    out_pdf = joinpath(figdir, "fig4_c2d2_composite.pdf")
    out_png = joinpath(figdir, "fig4_c2d2_composite.png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)

    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
end

main(ARGS)
