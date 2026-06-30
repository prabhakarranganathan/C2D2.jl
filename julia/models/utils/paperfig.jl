module PaperFig

using CairoMakie
using Colors
using LaTeXStrings

export PaperStyle, style, new_figure, figure_size_cm, make_axis,
       apply_log10_y!, apply_log10_x!, apply_linear_midpoint_minorticks!,
       panel_letter!, parameter_block!, selected_feature_marker!, regime_labels!,
       relative_guide_grid!,
       fenep_gray_series, c2d2_green_series, experiment_family_color,
       soft_red, soft_gray, darken, paper_save, pretty_num

Base.@kwdef struct PaperStyle
    font_regular::String = "Times New Roman"
    font_bold::String = "Times New Roman"
    font_italic::String = "Times New Roman"
    font_bold_italic::String = "Times New Roman"

    font_base_pt::Float64 = 12.0
    tick_label_scale::Float64 = 0.75
    axis_label_scale::Float64 = 1.0
    annotation_scale::Float64 = 0.65
    panel_letter_scale::Float64 = 1.05

    axis_lw::Float64 = 1.0
    tick_w::Float64 = 0.75
    tick_major::Float64 = 3.0
    tick_minor::Float64 = 2.0

    line_lw::Float64 = 2.0
    marker_size::Float64 = 6.0

    fenep_gray_lo::Float64 = 0.28
    fenep_gray_hi::Float64 = 0.82
    c2d2_base::RGBf = RGBf(0.07, 0.47, 0.20)  # muted green anchor
    exp_anna::RGBf = RGBf(0x00/255, 0x77/255, 0xBB/255)
    exp_clasen::RGBf = RGBf(0xEE/255, 0x77/255, 0x33/255)
    exp_gaillard::RGBf = RGBf(0x00/255, 0x99/255, 0x88/255)
    ref_dark::RGBf = RGBf(0.25, 0.25, 0.25)
    callout_red::RGBf = RGBf(0.78, 0.18, 0.14)
end

style(; kwargs...) = PaperStyle(; kwargs...)

cm_to_pt(x_cm) = x_cm * 72.0 / 2.54

function figure_size_cm(kind::Symbol)
    if kind == :fig1_textwidth_row3
        return (17.0, 8.5)
    elseif kind == :fig2_singlecol_square
        return (8.5, 8.5)
    elseif kind == :singlecol_wide
        return (8.5, 5.8)
    else
        error("Unknown figure kind: $kind")
    end
end

function activate_theme!(sty::PaperStyle)
    set_theme!(Theme(
        fonts = (
            regular = sty.font_regular,
            bold = sty.font_bold,
            italic = sty.font_italic,
            bold_italic = sty.font_bold_italic,
        ),
        fontsize = sty.font_base_pt,
        Axis = (
            spinewidth = sty.axis_lw,
            xlabelsize = sty.axis_label_scale * sty.font_base_pt,
            ylabelsize = sty.axis_label_scale * sty.font_base_pt,
            xticklabelsize = sty.tick_label_scale * sty.font_base_pt,
            yticklabelsize = sty.tick_label_scale * sty.font_base_pt,
            xlabelfont = :regular,
            ylabelfont = :regular,
            xticklabelfont = :regular,
            yticklabelfont = :regular,
            xgridvisible = false,
            ygridvisible = false,
            xminorgridvisible = false,
            yminorgridvisible = false,
            xtickalign = 1.0,
            ytickalign = 1.0,
            xminortickalign = 1.0,
            yminortickalign = 1.0,
            xticksmirrored = true,
            yticksmirrored = true,
            xticksize = sty.tick_major,
            yticksize = sty.tick_major,
            xtickwidth = sty.tick_w,
            ytickwidth = sty.tick_w,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorticksize = sty.tick_minor,
            yminorticksize = sty.tick_minor,
            xminortickwidth = sty.tick_w,
            yminortickwidth = sty.tick_w,
        ),
        Legend = (
            framevisible = false,
            labelsize = sty.tick_label_scale * sty.font_base_pt,
        ),
    ))
end

function new_figure(kind::Symbol; sty::PaperStyle=style(), width_cm=nothing, height_cm=nothing)
    activate_theme!(sty)
    if isnothing(width_cm) || isnothing(height_cm)
        (wcm, hcm) = figure_size_cm(kind)
        width_cm = something(width_cm, wcm)
        height_cm = something(height_cm, hcm)
    end
    fig = Figure(size=(cm_to_pt(width_cm), cm_to_pt(height_cm)), backgroundcolor=:white)
    return fig, sty
end

function make_axis(slot; xlabel="", ylabel="", xscale=identity, yscale=identity)
    ax = Axis(slot; xlabel=xlabel, ylabel=ylabel)
    ax.xscale = xscale
    ax.yscale = yscale
    return ax
end

pretty_num(x) = isapprox(x, round(x); atol=1e-12) ? string(Int(round(x))) : string(x)

# -------- colours --------
soft_red(sty::PaperStyle=style()) = sty.callout_red
soft_gray(sty::PaperStyle=style()) = RGBf(0.38, 0.38, 0.38)

darken(c::Colorant, frac::Real=0.08) = begin
    rgb = RGB(c)
    f = clamp(1 - frac, 0, 1)
    RGBf(f * rgb.r, f * rgb.g, f * rgb.b)
end

function fenep_gray_series(n::Integer; sty::PaperStyle=style())
    n <= 1 && return [RGBf(sty.fenep_gray_hi, sty.fenep_gray_hi, sty.fenep_gray_hi)]
    g = collect(range(sty.fenep_gray_lo, sty.fenep_gray_hi; length=n))
    # return light->dark ordering by caller choice; here dark->light raw ramp
    [RGBf(x, x, x) for x in g]
end

function c2d2_green_series(n::Integer; sty::PaperStyle=style())
    n <= 1 && return [sty.c2d2_base]
    base = RGB(sty.c2d2_base)
    ts = collect(range(0.65, 0.05; length=n))
    [RGBf((1-t)*base.r + t, (1-t)*base.g + t, (1-t)*base.b + t) for t in ts]
end

function experiment_family_color(name::AbstractString; sty::PaperStyle=style())
    s = lowercase(String(name))
    if occursin("anna", s) || occursin("mckinley", s)
        return sty.exp_anna
    elseif occursin("clasen", s)
        return sty.exp_clasen
    elseif occursin("gaillard", s)
        return sty.exp_gaillard
    else
        return RGBf(0.2, 0.2, 0.2)
    end
end

# -------- ticks --------
function decade_label(n::Int)
    n == 0 && return "1"
    return rich("10", superscript(string(n)))
end

function apply_log10_y!(ax, ymin, ymax)
    nmin = floor(Int, log10(ymin))
    nmax = ceil(Int, log10(ymax))
    majors = Float64[]
    labels = Any[]
    minors = Float64[]
    for n in nmin:nmax
        ymaj = 10.0^n
        if ymin <= ymaj <= ymax
            push!(majors, ymaj)
            push!(labels, decade_label(n))
        end
        ym = sqrt(10.0) * 10.0^n
        if ymin < ym < ymax
            push!(minors, ym)
        end
    end
    ax.yminorticksvisible = false
    ylims!(ax, ymin, ymax)
    ax.yscale = log10
    ax.yticks = (majors, labels)
    ax.yminorticks = minors
    ax.yminorticksvisible = true
end

function apply_log10_x!(ax, xmin, xmax)
    nmin = floor(Int, log10(xmin))
    nmax = ceil(Int, log10(xmax))
    majors = Float64[]
    labels = Any[]
    minors = Float64[]
    for n in nmin:nmax
        xmaj = 10.0^n
        if xmin <= xmaj <= xmax
            push!(majors, xmaj)
            push!(labels, decade_label(n))
        end
        xm = sqrt(10.0) * 10.0^n
        if xmin < xm < xmax
            push!(minors, xm)
        end
    end
    ax.xminorticksvisible = false
    xlims!(ax, xmin, xmax)
    ax.xscale = log10
    ax.xticks = (majors, labels)
    ax.xminorticks = minors
    ax.xminorticksvisible = true
end

function apply_linear_midpoint_minorticks!(ax, major_ticks::AbstractVector{<:Real}; axis::Symbol=:x)
    mids = 0.5 .* (major_ticks[1:end-1] .+ major_ticks[2:end])
    if axis == :x
        ax.xticks = collect(major_ticks)
        ax.xminorticks = mids
    else
        ax.yticks = collect(major_ticks)
        ax.yminorticks = mids
    end
end

# -------- annotations --------
function panel_letter!(ax, letter::AbstractString; sty::PaperStyle=style(), x=0.03, y=0.95)
    text!(ax, [x], [y]; text=[letter], space=:relative,
          align=(:left, :top), fontsize=sty.panel_letter_scale*sty.font_base_pt,
          font=:bold, color=:black)
end

function parameter_block!(ax, entries; sty::PaperStyle=style(), x=0.96, y0=0.22, dy=0.075, color=soft_gray(sty))
    for (k, txt) in enumerate(entries)
        text!(ax, [x], [y0 - (k-1)*dy]; text=[txt], space=:relative,
              color=color, align=(:right, :top), fontsize=0.95*sty.annotation_scale*sty.font_base_pt,
              font=:regular)
    end
end

function selected_feature_marker!(ax, x, y; label=nothing, sty::PaperStyle=style(), color=soft_red(sty), xfac=1.03, yfac=1.10)
    scatter!(ax, [x], [y]; color=color, markersize=0.95*sty.marker_size)
    if !isnothing(label)
        text!(ax, [x*xfac], [y*yfac]; text=[label], color=color,
              align=(:left,:bottom), fontsize=sty.annotation_scale*sty.font_base_pt,
              font=:regular)
    end
end

function regime_labels!(ax, xs, ys, labels; sty::PaperStyle=style(), color=RGBf(0.42,0.42,0.42))
    text!(ax, xs, ys; text=labels, space=:relative, color=color,
          align=(:center,:top), fontsize=0.95*sty.annotation_scale*sty.font_base_pt,
          font=:regular)
end

function paper_save(fig, path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".png"
        save(path, fig; px_per_unit=4)
    else
        save(path, fig; pt_per_unit=1)
    end
    path
end

function relative_guide_grid!(ax;
    xs = 0.1:0.1:0.9,
    ys = 0.1:0.1:0.9,
    color = RGBAf(0.0, 0.0, 0.0, 0.10),
    linewidth = 0.5,
    linestyle = :solid
)
    # verticals
    for x in xs
        lines!(ax, [x, x], [0.0, 1.0];
            space = :relative,
            color = color,
            linewidth = linewidth,
            linestyle = linestyle
        )
    end

    # horizontals
    for y in ys
        lines!(ax, [0.0, 1.0], [y, y];
            space = :relative,
            color = color,
            linewidth = linewidth,
            linestyle = linestyle
        )
    end

    return nothing
end

end # module
