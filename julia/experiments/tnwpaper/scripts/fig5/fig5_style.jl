#!/usr/bin/env julia

module Fig5Style

using CSV
using DataFrames
using CairoMakie
using Colors: Colorant, red, green, blue
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "../../src/TNW.jl"))
using .TNW

include(joinpath(TNW.julia_root(), "models", "utils", "paperfig.jl"))
using .PaperFig

export default_harvested_dir, default_figdir, style_bundle,
       DATASETS, dataset_color, dataset_label, sample_markers_for,
       load_panel_unnormalized, load_expt_normalized_all, load_expt_unnormalized_all,
       load_sample_manifest_all,
       reltext!, apply_log10_x!, apply_log10_y!, darken, soft_gray,
       panel_fig_square, panel_fig_inset, save_panel, master_curve,
       positive_bounds, auto_log10_limits,
       plot_dataset_unnorm!, plot_expt_normalized!, plot_expt_unnorm!, plot_expt_unnorm_inset!,
       annotate_nk_per_sample!, phi_shade_colors, annotate_phi_left!,
       debug_guides!

# ------------------------------------------------------------
# defaults
# ------------------------------------------------------------

default_harvested_dir() = joinpath(paper_root(), "outputs", "harvested")
default_figdir()         = joinpath(paper_root(), "figures", "fig5_panels")

function style_bundle()
    sty = style()
    PaperFig.activate_theme!(sty)
    return sty
end

# ------------------------------------------------------------
# dataset identity
# ------------------------------------------------------------

const DATASETS = ["gaillard", "anna", "clasen", "calabrese"]

# Calabrese colour not yet in PaperFig — plum / magenta
const _CALABRESE_COLOR = RGBf(0xAA/255, 0x22/255, 0x77/255)

function dataset_color(ds::AbstractString; sty::PaperStyle = style())
    s = lowercase(String(ds))
    s == "calabrese" && return _CALABRESE_COLOR
    return experiment_family_color(s; sty = sty)
end

function dataset_label(ds::AbstractString)
    s = lowercase(String(ds))
    return s == "gaillard"  ? "Gaillard"  :
           s == "anna"      ? "Anna"      :
           s == "clasen"    ? "Clasen"    :
           s == "calabrese" ? "Calabrese" :
           uppercasefirst(s)
end

# Ordered marker sequence; index by position of sample_id in sorted list
const _MARKER_SEQ = [:circle, :rect, :utriangle, :dtriangle, :diamond, :pentagon, :star5, :xcross]

"""
Return a Dict mapping each sample_id in `ids` to a marker symbol.
`ids` should be the sorted unique sample IDs for the dataset.
"""
function sample_markers_for(ids::Vector{<:AbstractString})
    Dict(sid => _MARKER_SEQ[mod1(i, length(_MARKER_SEQ))] for (i, sid) in enumerate(ids))
end

# ------------------------------------------------------------
# data loading
# ------------------------------------------------------------

function load_panel_unnormalized(harvested_dir::AbstractString, dataset::AbstractString)
    panel_path = joinpath(harvested_dir, "experimental_$(dataset)", "panel_unnormalized.csv")
    model_path = joinpath(harvested_dir, "experimental_$(dataset)", "model_curves_c2d2.csv")
    isfile(panel_path) || error("Missing panel_unnormalized.csv for $dataset:\n  $panel_path\nRun harvest_experimental_all.jl first.")
    df = DataFrame(CSV.File(panel_path; silencewarnings = true))

    # panel_unnormalized.csv can contain duplicate (sample_id, De) model rows when both
    # drag_constant and drag_c2d2 CaBER sweeps are present. Replace the model rows with
    # a sweep_folder-filtered version from model_curves_c2d2.csv (keeps only drag_c2d2).
    if isfile(model_path)
        mc      = DataFrame(CSV.File(model_path; silencewarnings = true))
        mc_c2d2 = mc[occursin.("drag_c2d2", coalesce.(string.(mc.sweep_folder), "")), :]
        expt    = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]
        model_clean = mc_c2d2[:, intersect(names(df), names(mc_c2d2))]
        df = vcat(expt, model_clean; cols = :intersect)
    end

    return df
end

function load_expt_normalized_all(harvested_dir::AbstractString)
    path = joinpath(harvested_dir, "experimental_all", "all_expt_normalized.csv")
    isfile(path) || error("Missing all_expt_normalized.csv:\n  $path")
    DataFrame(CSV.File(path; silencewarnings = true))
end

function load_expt_unnormalized_all(harvested_dir::AbstractString)
    path = joinpath(harvested_dir, "experimental_all", "all_expt_unnormalized.csv")
    isfile(path) || error("Missing all_expt_unnormalized.csv:\n  $path")
    DataFrame(CSV.File(path; silencewarnings = true))
end

function load_sample_manifest_all(harvested_dir::AbstractString)
    path = joinpath(harvested_dir, "experimental_all", "all_sample_manifest.csv")
    isfile(path) || error("Missing all_sample_manifest.csv:\n  $path")
    DataFrame(CSV.File(path; silencewarnings = true))
end

# ------------------------------------------------------------
# generic helpers
# ------------------------------------------------------------

reltext!(ax, x, y, txt; sty, color = soft_gray(sty), fs_scale = 1.0, align = (:left, :bottom)) =
    text!(ax, [x], [y]; text = [txt], space = :relative, color = color,
          align = align, fontsize = fs_scale * sty.annotation_scale * sty.font_base_pt,
          font = :regular)

apply_log10_x!(ax, xmin, xmax) = PaperFig.apply_log10_x!(ax, xmin, xmax)
apply_log10_y!(ax, ymin, ymax) = PaperFig.apply_log10_y!(ax, ymin, ymax)
darken(c, a = 0.15)            = PaperFig.darken(c, a)

master_curve(x::Real) = x <= 1 ? 1.0 : x^(4/3)

function positive_bounds(cols...; pad_low = 0.08, pad_high = 0.08)
    vals = Float64[]
    for v in cols
        for x in v
            ismissing(x) && continue
            fx = Float64(x)
            isfinite(fx) && fx > 0 && push!(vals, fx)
        end
    end
    isempty(vals) && error("No positive finite values found for axis bounds.")
    vmin, vmax = extrema(vals)
    lo = 10.0^(floor(log10(vmin)) - pad_low)
    hi = 10.0^(ceil(log10(vmax))  + pad_high)
    return lo, hi
end

"""Compute decade-snapped log10 limits from one or more data vectors."""
function auto_log10_limits(cols...)
    positive_bounds(cols...; pad_low = 0.0, pad_high = 0.0)
end

"""
Generate `n` shades of `base_col` ordered lightest (index 1) to darkest (index n).
Used to encode phi rank within a dataset: lowest phi → lightest, highest phi → darkest.
Middle shade (n odd, middle index) is closest to the original base color.
"""
function phi_shade_colors(base_col::Colorant, n::Int)
    n == 1 && return RGBf[RGBf(red(base_col), green(base_col), blue(base_col))]
    r0, g0, b0 = Float64(red(base_col)), Float64(green(base_col)), Float64(blue(base_col))
    lighten_max = 0.45   # how far towards white for the lightest shade
    darken_max  = 0.30   # how far towards black for the darkest shade
    result = Vector{RGBf}(undef, n)
    for i in 1:n
        t = (i - 1) / (n - 1)   # 0 = lightest, 1 = darkest
        if t < 0.5
            s = (0.5 - t) / 0.5             # 1 → 0 as t → 0.5
            fac = lighten_max * s
            result[i] = RGBf(r0 + fac*(1-r0), g0 + fac*(1-g0), b0 + fac*(1-b0))
        elseif t > 0.5
            s   = (t - 0.5) / 0.5           # 0 → 1 as t → 1.0
            fac = 1.0 - darken_max * s
            result[i] = RGBf(r0*fac, g0*fac, b0*fac)
        else
            result[i] = RGBf(r0, g0, b0)   # middle = base color
        end
    end
    return result
end

"""
Stack N_K labels (one per unique NK value) in the upper-right of the axis.
`sample_colors`: optional Dict{String,<:Colorant} — if provided, each label is
colored with the phi-shade of its representative curve; otherwise uses `col`.
"""
function annotate_nk_per_sample!(ax, model_df, col, xbounds, ybounds;
                                  sty, sample_colors::Union{Nothing,Dict} = nothing)
    log_xmin, log_xmax = log10.(xbounds)
    log_ymin, log_ymax = log10.(ybounds)

    # One label per unique NK. Track representative sample_id (highest Wi_e endpoint).
    nk_max_wie = Dict{Int, Float64}()
    nk_rep_sid = Dict{Int, String}()
    for sid in unique(string.(model_df.sample_id))
        sub = model_df[string.(model_df.sample_id) .== sid, :]
        isempty(sub) && continue
        nk_val  = Int(round(first(sub.NK)))
        wie_end = maximum(sub.Wi_e)
        if !haskey(nk_max_wie, nk_val) || wie_end > nk_max_wie[nk_val]
            nk_max_wie[nk_val] = wie_end
            nk_rep_sid[nk_val] = sid
        end
    end

    nk_sorted = sort(collect(keys(nk_max_wie)), by = nk -> -nk_max_wie[nk])
    y_top  = 0.88
    y_step = 0.10
    for (i, nk_val) in enumerate(nk_sorted)
        y_label = y_top - (i - 1) * y_step
        lc  = isnothing(sample_colors) ? col : get(sample_colors, nk_rep_sid[nk_val], col)
        # First label shows full "N_\mathrm{K} = value"; subsequent labels show just the value.
        lbl = i == 1 ? latexstring("N_\\mathrm{K} = $(nk_val)") : latexstring("$(nk_val)")
        reltext!(ax, 0.97, y_label, lbl;
                 sty = sty, color = lc, align = (:right, :bottom))
    end
end

"""
Place φ-value labels just above each sample's low-De plateau, at the left edge of
the axis. Colors are taken from `sample_colors` (phi-shade dict), matching each line.
Samples are processed in ascending phi order so labels track curve ordering.
"""
function annotate_phi_left!(ax, model_df, xbounds, ybounds;
                             sty, sample_colors::Dict)
    log_ymin, log_ymax = log10.(ybounds)
    y_span = log_ymax - log_ymin

    # Collect (sid, phi, wie_low) per sample — wie_low = Wi_e at minimum De
    phi_by_sid  = Dict{String,Float64}()
    wie_low_sid = Dict{String,Float64}()
    for row in eachrow(model_df)
        sid = string(row.sample_id)
        haskey(phi_by_sid, sid) && continue
        phi_by_sid[sid]  = Float64(row.phi)
    end
    for sid in keys(phi_by_sid)
        sub = sort(model_df[string.(model_df.sample_id) .== sid, :], :De)
        wie_low_sid[sid] = isempty(sub) ? NaN : sub.Wi_e[1]
    end

    # Process in ascending phi order (lightest shade first = bottom curve)
    phi_sids = sort(collect(keys(phi_by_sid)); by = s -> phi_by_sid[s])

    first_shown = true
    for sid in phi_sids
        wie_low = wie_low_sid[sid]
        isnan(wie_low) && continue
        col     = get(sample_colors, sid, soft_gray(sty))
        y_rel   = clamp((log10(wie_low) - log_ymin) / y_span, 0.02, 0.94)
        phi_str = @sprintf("%.3g", phi_by_sid[sid])
        lbl     = first_shown ? latexstring("\\phi = $(phi_str)") : latexstring("$(phi_str)")
        first_shown = false
        reltext!(ax, 0.03, y_rel + 0.04, lbl;
                 sty = sty, color = col, align = (:left, :bottom))
    end
end

# ------------------------------------------------------------
# panel figure constructors (same gutter spec as Fig 4)
# ------------------------------------------------------------

function panel_fig_square(;
    axis_w_cm        = 5.65,
    axis_h_cm        = 5.65,
    left_gutter_cm   = 0.7,
    right_gutter_cm  = 0.93,
    top_gutter_cm    = 0.66,
    bottom_gutter_cm = 0.52,
    xlabel           = L"\mathrm{De}",
    ylabel           = L"\mathrm{Wi}_{\mathrm{e}}",
    show_ylabel      = true,
    show_yticklabels = true,
)
    fig_w_cm = left_gutter_cm + axis_w_cm + right_gutter_cm
    fig_h_cm = top_gutter_cm  + axis_h_cm + bottom_gutter_cm

    fig = Figure(
        size            = (PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor = :white,
        figure_padding  = (0, 0, 0, 0),
    )

    gl = GridLayout(fig[1, 1])
    colgap!(gl, 0); rowgap!(gl, 0)

    # scaffold boxes so all rows/cols exist before sizing
    for (r, c) in Iterators.product(1:3, 1:3)
        (r == 2 && c == 2) && continue
        Box(gl[r, c]; color = :transparent, strokecolor = :transparent,
            tellwidth = false, tellheight = false)
    end

    ax = Axis(gl[2, 2]; xlabel = xlabel, ylabel = ylabel)

    colsize!(gl, 1, Fixed(PaperFig.cm_to_pt(left_gutter_cm)))
    colsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_w_cm)))
    colsize!(gl, 3, Fixed(PaperFig.cm_to_pt(right_gutter_cm)))
    rowsize!(gl, 1, Fixed(PaperFig.cm_to_pt(top_gutter_cm)))
    rowsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_h_cm)))
    rowsize!(gl, 3, Fixed(PaperFig.cm_to_pt(bottom_gutter_cm)))

    ax.ylabelvisible      = show_ylabel
    ax.yticklabelsvisible = show_yticklabels
    !show_yticklabels && (ax.yticksvisible = false)

    return fig, ax
end

"""Smaller figure for the unnormalized inset (panel F)."""
function panel_fig_inset(;
    axis_w_cm        = 2.8,
    axis_h_cm        = 2.8,
    left_gutter_cm   = 0.55,
    right_gutter_cm  = 0.30,
    top_gutter_cm    = 0.30,
    bottom_gutter_cm = 0.48,
    xlabel           = L"\mathrm{De}",
    ylabel           = L"\mathrm{Wi}_{\mathrm{e}}",
)
    fig_w_cm = left_gutter_cm + axis_w_cm + right_gutter_cm
    fig_h_cm = top_gutter_cm  + axis_h_cm + bottom_gutter_cm

    fig = Figure(
        size            = (PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor = :white,
        figure_padding  = (0, 0, 0, 0),
    )

    gl = GridLayout(fig[1, 1])
    colgap!(gl, 0); rowgap!(gl, 0)

    for (r, c) in Iterators.product(1:3, 1:3)
        (r == 2 && c == 2) && continue
        Box(gl[r, c]; color = :transparent, strokecolor = :transparent,
            tellwidth = false, tellheight = false)
    end

    ax = Axis(gl[2, 2];
        xlabel        = xlabel,
        ylabel        = ylabel,
        xlabelpadding = 1.0,
        ylabelpadding = 1.0,
    )

    colsize!(gl, 1, Fixed(PaperFig.cm_to_pt(left_gutter_cm)))
    colsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_w_cm)))
    colsize!(gl, 3, Fixed(PaperFig.cm_to_pt(right_gutter_cm)))
    rowsize!(gl, 1, Fixed(PaperFig.cm_to_pt(top_gutter_cm)))
    rowsize!(gl, 2, Fixed(PaperFig.cm_to_pt(axis_h_cm)))
    rowsize!(gl, 3, Fixed(PaperFig.cm_to_pt(bottom_gutter_cm)))

    return fig, ax
end

# ------------------------------------------------------------
# save helper
# ------------------------------------------------------------

function save_panel(fig, figdir::AbstractString, stem::AbstractString)
    isdir(figdir) || mkpath(figdir)
    out_pdf = joinpath(figdir, stem * ".pdf")
    out_png = joinpath(figdir, stem * ".png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)
    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
    return (pdf = out_pdf, png = out_png)
end

# ------------------------------------------------------------
# core plotting routines
# ------------------------------------------------------------

"""
Plot one dataset's unnormalized Pipkin panel.

Model rows  → one line per sample_id, all in dataset color (no marker differentiation).
Expt rows   → scatter per sample_id, dataset color, per-sample marker.
Returns the sorted sample-id list.

`sample_colors`: optional Dict{String,<:Colorant} mapping sample_id → color.
When provided each sample gets its own color (e.g. phi-shaded); otherwise all
samples share the dataset color.
"""
function plot_dataset_unnorm!(ax, df::DataFrame, dataset::AbstractString;
                              sty::PaperStyle,
                              sample_colors::Union{Nothing,Dict} = nothing)
    col      = dataset_color(dataset; sty = sty)
    model    = df[coalesce.(string.(df.source_kind) .== "model",      false), :]
    expt     = df[coalesce.(string.(df.source_kind) .== "experiment", false), :]
    all_ids  = sort(unique(string.(df.sample_id)))
    mrk_map  = sample_markers_for(all_ids)
    sc(sid)  = isnothing(sample_colors) ? col : get(sample_colors, sid, col)

    # model curves
    for sid in all_ids
        m = model[coalesce.(string.(model.sample_id) .== sid, false), :]
        isempty(m) && continue
        sort!(m, :De)
        lines!(ax, Float64.(m.De), Float64.(m.Wi_e);
               color = sc(sid), linewidth = sty.line_lw)
    end

    # experimental scatter
    for sid in all_ids
        e = expt[coalesce.(string.(expt.sample_id) .== sid, false), :]
        isempty(e) && continue
        scatter!(ax, Float64.(e.De), Float64.(e.Wi_e);
                 color       = sc(sid),
                 marker      = mrk_map[sid],
                 markersize  = sty.marker_size,
                 strokecolor = :black,
                 strokewidth = 0.9)
    end

    return all_ids
end

"""
Plot all experimental normalized points (panel E master).
One color per dataset, per-sample markers within each dataset.
Adds master curve and Wi_e_bar = 1 reference line.
"""
function plot_expt_normalized!(ax, df::DataFrame; sty::PaperStyle)
    for ds in DATASETS
        sub = df[coalesce.(string.(df.dataset) .== ds, false), :]
        isempty(sub) && continue
        col     = dataset_color(ds; sty = sty)
        all_ids = sort(unique(string.(sub.sample_id)))
        mrk_map = sample_markers_for(all_ids)
        for sid in all_ids
            e = sub[coalesce.(string.(sub.sample_id) .== sid, false), :]
            isempty(e) && continue
            De_bar   = Float64.(e.De_bar)
            Wi_e_bar = Float64.(e.Wi_e_bar)
            good = isfinite.(De_bar) .& isfinite.(Wi_e_bar) .& (De_bar .> 0) .& (Wi_e_bar .> 0)
            scatter!(ax, De_bar[good], Wi_e_bar[good];
                     color       = col,
                     marker      = mrk_map[sid],
                     markersize  = sty.marker_size,
                     strokecolor = :black,
                     strokewidth = 0.9)
        end
    end

    # Wi_e_bar = 1 reference
    hlines!(ax, [1.0];
            color     = sty.ref_dark,
            linestyle = :dot,
            linewidth = 0.85 * sty.line_lw / 2)

    # master curve y = (De_bar)^(4/3)
    xmc = 10 .^ range(-1.0, log10(50.0); length = 500)
    lines!(ax, xmc, master_curve.(xmc);
           color     = sty.ref_dark,
           linestyle = :dash,
           linewidth = 0.95 * sty.line_lw / 2)
end

"""
Plot all experimental unnormalized points (panel F inset).
One color per dataset, per-sample markers. No model curves.
"""
function plot_expt_unnorm_inset!(ax, df::DataFrame; sty::PaperStyle)
    for ds in DATASETS
        sub = df[coalesce.(string.(df.dataset) .== ds, false), :]
        isempty(sub) && continue
        col     = dataset_color(ds; sty = sty)
        all_ids = sort(unique(string.(sub.sample_id)))
        mrk_map = sample_markers_for(all_ids)
        for sid in all_ids
            e = sub[coalesce.(string.(sub.sample_id) .== sid, false), :]
            isempty(e) && continue
            De  = Float64.(e.De)
            Wie = Float64.(e.Wi_e)
            good = isfinite.(De) .& isfinite.(Wie) .& (De .> 0) .& (Wie .> 0)
            scatter!(ax, De[good], Wie[good];
                     color       = col,
                     marker      = mrk_map[sid],
                     markersize  = 0.85 * sty.marker_size,   # slightly smaller for inset
                     strokecolor = :black,
                     strokewidth = 0.7)
        end
    end
end

"""
Plot all experimental unnormalized points (full-size square panel version).
One color per dataset, per-sample markers. No model curves.
"""
function plot_expt_unnorm!(ax, df::DataFrame; sty::PaperStyle)
    for ds in DATASETS
        sub = df[coalesce.(string.(df.dataset) .== ds, false), :]
        isempty(sub) && continue
        col     = dataset_color(ds; sty = sty)
        all_ids = sort(unique(string.(sub.sample_id)))
        mrk_map = sample_markers_for(all_ids)
        for sid in all_ids
            e = sub[coalesce.(string.(sub.sample_id) .== sid, false), :]
            isempty(e) && continue
            De  = Float64.(e.De)
            Wie = Float64.(e.Wi_e)
            good = isfinite.(De) .& isfinite.(Wie) .& (De .> 0) .& (Wie .> 0)
            scatter!(ax, De[good], Wie[good];
                     color       = col,
                     marker      = mrk_map[sid],
                     markersize  = sty.marker_size,
                     strokecolor = :black,
                     strokewidth = 0.9)
        end
    end
end

"""
Overlay a relative-coordinate guide grid for label positioning.
Draws faint horizontal and vertical lines at 0.1 increments in relative axis
space, with coordinate labels along the left and bottom edges.
Call this before `save_panel` and remove when the figure is finalized.
"""
function debug_guides!(ax; sty = nothing)
    for v in 0.0:0.1:1.0
        linesegments!(ax, [0f0, 1f0], [Float32(v), Float32(v)];
                      space = :relative, color = (:black, 0.12), linewidth = 0.4)
        linesegments!(ax, [Float32(v), Float32(v)], [0f0, 1f0];
                      space = :relative, color = (:black, 0.12), linewidth = 0.4)
        lbl = @sprintf("%.1f", v)
        text!(ax, [Float32(0.01)], [Float32(v + 0.01)];
              text    = [lbl],
              space   = :relative,
              color   = (:black, 0.45),
              fontsize = 4.5,
              align   = (:left, :bottom),
              font    = :regular)
    end
end

end # module Fig5Style
