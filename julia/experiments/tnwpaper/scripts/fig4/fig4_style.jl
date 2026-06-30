#!/usr/bin/env julia

module Fig4Style

using CSV
using DataFrames
using CairoMakie
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "../..", "src", "TNW.jl"))
using .TNW

include(joinpath(TNW.julia_root(), "models", "utils", "paperfig.jl"))
using .PaperFig

export default_harvested_dir, default_figdir, style_bundle,
       load_curves, load_features_selected, load_sweep_features, load_renorm_params,
       attach_renorm, approx_mask, curve_colsym, positive_bounds,
       pick_curve_run, pick_feature_row, subset_pipkin,
       reltext!, c2d2_cols, combo_style, combo_label, pretty_hK, pretty_num,
       apply_log10_x!, apply_log10_y!, darken,
       panel_fig_top, panel_fig_square, save_panel, master_curve

# ------------------------------------------------------------
# defaults
# ------------------------------------------------------------

default_harvested_dir() = joinpath(paper_root(), "outputs", "harvested", "generic_features")
default_figdir() = joinpath(paper_root(), "figures", "fig4_panels")

function style_bundle()
    sty = style()
    PaperFig.activate_theme!(sty)
    return sty
end

# ------------------------------------------------------------
# wrappers for PaperFig helpers used by panel scripts
# ------------------------------------------------------------

pretty_hK(h) = @sprintf("%.2f", h)
pretty_num(x) = PaperFig.pretty_num(x)
apply_log10_x!(ax, xmin, xmax) = PaperFig.apply_log10_x!(ax, xmin, xmax)
apply_log10_y!(ax, ymin, ymax) = PaperFig.apply_log10_y!(ax, ymin, ymax)
darken(c, a=0.15) = PaperFig.darken(c, a)

master_curve(x) = x <= 1 ? 1.0 : x^(4 / 3)

# ------------------------------------------------------------
# data loading
# ------------------------------------------------------------

function load_curves(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "curves.csv")
    isfile(csv) || error("Missing harvested curves.csv: $csv")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function load_features_selected(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "features_selected.csv")
    isfile(csv) || error("Missing harvested features_selected.csv: $csv")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function load_sweep_features(harvested_dir::AbstractString)
    csv = joinpath(harvested_dir, "sweep_features.csv")
    isfile(csv) || error("Missing harvested sweep_features.csv: $csv")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

function load_renorm_params(harvested_dir::AbstractString; renorm_csv::AbstractString="")
    csv = isempty(renorm_csv) ? joinpath(harvested_dir, "pipkin_renorm_params.csv") : renorm_csv
    isfile(csv) || error("Missing harvested pipkin_renorm_params.csv: $csv")
    DataFrame(CSV.File(csv; silencewarnings=true))
end

# ------------------------------------------------------------
# renorm attachment
# ------------------------------------------------------------

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
    error("None of the candidate columns $(candidates) found.")
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
    hasproperty(d, :drag_model)  && (mask .&= coalesce.(d.drag_model .== drag_model, false))
    hasproperty(d, :NK)          && (mask .&= coalesce.(d.NK .== NK, false))
    hasproperty(d, :hK_star)     && (mask .&= coalesce.(isapprox.(d.hK_star, hK_star; atol=1e-12), false))

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
    di = dd[dd.run_id .== rid, :]
    :t in propertynames(di) && sort!(di, :t)
    return di
end

function pick_feature_row(features::DataFrame, run_id::AbstractString)
    f = features[features.run_id .== run_id, :]
    isempty(f) && error("No selected-feature rows found for run_id=$run_id")

    if hasproperty(f, :selection_reason)
        g = f[f.selection_reason .== "deepest_min", :]
        !isempty(g) && return g[1, :]
        g = f[f.selection_reason .== "fallback_inflexion", :]
        !isempty(g) && return g[1, :]
    end
    if hasproperty(f, :feature_type)
        g = f[f.feature_type .== "min", :]
        !isempty(g) && return g[1, :]
    end
    return f[1, :]
end

function subset_pipkin(df; drag_model, NK=nothing, hK_star=nothing, phi=nothing)
    mask = trues(nrow(df))
    mask .&= coalesce.(df.sweep_feature_kind .== "Wi_e_vs_De", false)
    mask .&= coalesce.(df.flow_family .== "caber", false)
    mask .&= coalesce.(df.drag_model .== drag_model, false)
    NK !== nothing      && (mask .&= coalesce.(df.NK .== NK, false))
    hK_star !== nothing && (mask .&= coalesce.(isapprox.(df.hK_star, hK_star; atol=1e-12), false))
    phi !== nothing     && (mask .&= approx_mask(coalesce.(df.phi, NaN), phi))
    return df[mask, :]
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

combo_label(NK, hK) = L"(N_{\mathrm{K}},\,h_{\mathrm{K}}^{\ast}) = (%$(Int(round(NK))),\,%$(pretty_hK(hK)))"

# ------------------------------------------------------------
# fixed-geometry panel figures
# ------------------------------------------------------------

function panel_fig_top(;
    axis_w_cm = 4.85,
    axis_h_cm = 6.15,
    left_gutter_cm = 0.7,
    right_gutter_cm = 0.93,
    top_gutter_cm = 0.66,
    bottom_gutter_cm = 0.52,
    show_ylabel = true,
    show_yticklabels = true,
)
    fig_w_cm = left_gutter_cm + axis_w_cm + right_gutter_cm
    fig_h_cm = top_gutter_cm + axis_h_cm + bottom_gutter_cm

    fig = Figure(
        size = (PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor = :white,
        figure_padding = (0, 0, 0, 0),
    )

    gl = GridLayout(fig[1, 1])
    colgap!(gl, 0)
    rowgap!(gl, 0)

    # instantiate the 3x3 scaffold so the rows/cols exist before sizing
    Box(gl[1,1], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[1,2], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[1,3], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[2,1], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[2,3], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[3,1], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[3,2], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[3,3], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)

    ax = Axis(
        gl[2, 2],
        xlabel = L"\mathrm{Wi}",
        ylabel = L"\tilde{\eta}_{\mathrm{p}}",
    )

    colsize!(gl, 1, Fixed(PaperFig.cm_to_pt(left_gutter_cm)))
    colsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_w_cm)))
    colsize!(gl, 3, Fixed(PaperFig.cm_to_pt(right_gutter_cm)))

    rowsize!(gl, 1, Fixed(PaperFig.cm_to_pt(top_gutter_cm)))
    rowsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_h_cm)))
    rowsize!(gl, 3, Fixed(PaperFig.cm_to_pt(bottom_gutter_cm)))

    ax.ylabelvisible = show_ylabel
    ax.yticklabelsvisible = show_yticklabels
    if !show_yticklabels
        ax.yticksvisible = false
    end

    return fig, ax
end

function panel_fig_square(;
    axis_w_cm = 5.65,
    axis_h_cm = 5.65,
    left_gutter_cm = 0.7,
    right_gutter_cm = 0.93,
    top_gutter_cm = 0.66,
    bottom_gutter_cm = 0.52,
    xlabel = L"\mathrm{De}",
    ylabel = L"\mathrm{Wi}_{\mathrm{e}}",
    show_ylabel = true,
    show_yticklabels = true,
)
    fig_w_cm = left_gutter_cm + axis_w_cm + right_gutter_cm
    fig_h_cm = top_gutter_cm + axis_h_cm + bottom_gutter_cm

    fig = Figure(
        size = (PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor = :white,
        figure_padding = (0, 0, 0, 0),
    )

    gl = GridLayout(fig[1, 1])
    colgap!(gl, 0)
    rowgap!(gl, 0)

    Box(gl[1,1], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[1,2], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[1,3], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[2,1], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[2,3], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[3,1], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[3,2], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)
    Box(gl[3,3], color=:transparent, strokecolor=:transparent, tellwidth=false, tellheight=false)

    ax = Axis(
        gl[2, 2],
        xlabel = xlabel,
        ylabel = ylabel,
    )

    colsize!(gl, 1, Fixed(PaperFig.cm_to_pt(left_gutter_cm)))
    colsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_w_cm)))
    colsize!(gl, 3, Fixed(PaperFig.cm_to_pt(right_gutter_cm)))

    rowsize!(gl, 1, Fixed(PaperFig.cm_to_pt(top_gutter_cm)))
    rowsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_h_cm)))
    rowsize!(gl, 3, Fixed(PaperFig.cm_to_pt(bottom_gutter_cm)))

    ax.ylabelvisible = show_ylabel
    ax.yticklabelsvisible = show_yticklabels
    if !show_yticklabels
        ax.yticksvisible = false
    end

    return fig, ax
end

# ------------------------------------------------------------
# save helper
# ------------------------------------------------------------

function ensure_dir(d::AbstractString)
    isdir(d) || mkpath(d)
    return d
end

function save_panel(fig, figdir::AbstractString, stem::AbstractString)
    ensure_dir(figdir)
    out_pdf = joinpath(figdir, stem * ".pdf")
    out_png = joinpath(figdir, stem * ".png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)
    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
    return (pdf=out_pdf, png=out_png)
end

end # module