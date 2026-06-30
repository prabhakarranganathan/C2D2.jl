#!/usr/bin/env julia

using CSV
using DataFrames
using CairoMakie
using LaTeXStrings

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

function get_pos(D::AbstractDict, x; atol=1e-12)
    for (k, v) in D
        if isapprox(k, x; atol=atol)
            return v
        end
    end
    error("No manual position configured for value $x")
end

function reltext!(ax, x, y, txt;
    sty,
    color=soft_gray(sty),
    fs_scale=1.0,
)
    text!(ax, [x], [y];
        text=[txt],
        space=:relative,
        color=color,
        align=(:left, :bottom),
        fontsize=fs_scale * sty.annotation_scale * sty.font_base_pt,
        font=:regular
    )
end

function pick_curve_run(curves::DataFrame;
    flow_family::String,
    drag_model::String="constant",
    NK::Int=3000,
    hK_star::Float64=0.05,
    phi::Float64=0.01,
    De::Union{Nothing,Float64}=nothing,
)
    d = curves
    mask = trues(nrow(d))

    hasproperty(d, :flow_family) && (mask .&= d.flow_family .== flow_family)
    hasproperty(d, :drag_model) && (mask .&= d.drag_model .== drag_model)
    hasproperty(d, :NK) && (mask .&= d.NK .== NK)
    hasproperty(d, :hK_star) && (mask .&= approx_mask(d.hK_star, hK_star))

    # concentration filter:
    # - styext rows use setting_var="phi", setting_value=...
    # - caber rows should have phi populated directly
    if flow_family == "styext"
        if hasproperty(d, :setting_var) && hasproperty(d, :setting_value)
            mask .&= (string.(d.setting_var) .== "phi")
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
            mask .&= (string.(d.setting_var) .== "phi")
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
    isempty(dd) && error("No run found for flow_family=$flow_family, phi=$phi, De=$(De).")

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

# ------------------------------------------------------------
# plot one row
# ------------------------------------------------------------
function plot_row!(
    axR, axWi, axM, axEta;
    curves::DataFrame,
    feats::DataFrame,
    stycurve::DataFrame,
    phi::Float64,
    Devals::Vector{Float64},
    labels,
    sty::PaperStyle,
    drag_model::String,
    NK::Int,
    hK::Float64,
    show_param_block::Bool=false,
    debug_guides::Bool=true,
    t_ticks::Vector{Float64},
    t_xlim::Tuple{Float64,Float64},
    eps_ticks::Vector{Float64},
    eps_xlim::Tuple{Float64,Float64},
)
    wicol  = curve_colsym(curves, [:Wi, :wi])
    etacol = curve_colsym(curves, [:etaE1, :etaP])
    Mcol   = curve_colsym(curves, [:M, :E2, :Y])

    cols = reverse(fenep_gray_series(length(Devals); sty=sty))

    # steady curves
    Wi_sty  = Vector{Float64}(stycurve[!, wicol])
    M_sty   = Vector{Float64}(stycurve[!, Mcol])
    eta_sty = Vector{Float64}(stycurve[!, etacol])

    sp = sortperm(Wi_sty)
    Wi_sty  = Wi_sty[sp]
    M_sty   = M_sty[sp]
    eta_sty = eta_sty[sp]

    # collect caber trajectories + selected points
    rows = NamedTuple[]
    for (i, De) in enumerate(Devals)
        di = pick_curve_run(curves; flow_family="caber", drag_model=drag_model, NK=NK, hK_star=hK, phi=phi, De=De)
        fr = pick_feature_row(feats, di.run_id[1])
        idx = clamp(Int(fr.feature_idx), 1, nrow(di))

        t      = Vector{Float64}(di.t)
        strain = Vector{Float64}(di.estrain)
        Wi     = Vector{Float64}(di[!, wicol])
        M      = Vector{Float64}(di[!, Mcol])
        eta_p  = Vector{Float64}(di[!, etacol])
        R      = exp.(-strain ./ 2)

        push!(rows, (
            De=De,
            col=cols[i],
            t=t,
            strain=strain,
            Wi=Wi,
            M=M,
            eta_p=eta_p,
            R=R,
            idx=idx,
            t_e=t[idx],
            strain_e=strain[idx],
            Wi_e=Wi[idx],
            M_e=M[idx],
            eta_p_e=eta_p[idx],
            R_e=R[idx],
        ))
    end

    # ---------- debug guides ----------
    if debug_guides
        relative_guide_grid!(axR;   color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
        relative_guide_grid!(axWi;  color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
        relative_guide_grid!(axM;   color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
        relative_guide_grid!(axEta; color=RGBAf(0, 0, 0, 0.06), linewidth=0.4)
    end

    # ---------- panel i: R vs t ----------
    for row in rows
        lines!(axR, row.t, row.R; color=row.col, linewidth=sty.line_lw)
        scatter!(axR, [row.t_e], [row.R_e];
            color=row.col,
            markersize=sty.marker_size,
            strokecolor=darken(row.col, 0.15),
            strokewidth=0.6
        )
    end

    # ---------- panel ii: Wi vs ε ----------
    for row in rows
        lines!(axWi, row.strain, row.Wi; color=row.col, linewidth=sty.line_lw)
        scatter!(axWi, [row.strain_e], [row.Wi_e];
            color=row.col,
            markersize=sty.marker_size,
            strokecolor=darken(row.col, 0.15),
            strokewidth=0.6
        )
    end

    lines!(axWi, [eps_xlim[1], eps_xlim[2]], [2 / 3, 2 / 3];
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    # ---------- panel iii: M vs Wi ----------
    lines!(axM, Wi_sty, M_sty;
        color=sty.ref_dark,
        linewidth=0.9 * sty.line_lw,
        linestyle=:dot
    )

    for row in rows
        lines!(axM, row.Wi, row.M; color=row.col, linewidth=sty.line_lw)
        scatter!(axM, [row.Wi_e], [row.M_e];
            color=row.col,
            markersize=sty.marker_size,
            strokecolor=darken(row.col, 0.15),
            strokewidth=0.6
        )
    end

    yminM, ymaxM = positive_bounds(M_sty, [r.M for r in rows]...; pad_low=0.08, pad_high=0.08)
    lines!(axM, [2 / 3, 2 / 3], [yminM, ymaxM];
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    # ---------- panel iv: etap vs Wi ----------
    lines!(axEta, Wi_sty, eta_sty;
        color=sty.ref_dark,
        linewidth=0.9 * sty.line_lw,
        linestyle=:dot
    )

    for row in rows
        lines!(axEta, row.Wi, row.eta_p; color=row.col, linewidth=sty.line_lw)
        scatter!(axEta, [row.Wi_e], [row.eta_p_e];
            color=row.col,
            markersize=sty.marker_size,
            strokecolor=darken(row.col, 0.15),
            strokewidth=0.6
        )
    end

    yminEta, ymaxEta = positive_bounds(eta_sty, [r.eta_p for r in rows]...; pad_low=0.08, pad_high=0.08)
    lines!(axEta, [2 / 3, 2 / 3], [yminEta, ymaxEta];
        color=sty.ref_dark,
        linestyle=:dash,
        linewidth=0.95 * sty.line_lw / 2
    )

    # ---------- axes ----------
    # panel i
    axR.xticks = t_ticks
    xlims!(axR, t_xlim...)
    apply_linear_midpoint_minorticks!(axR, t_ticks; axis=:x)
    yminR, ymaxR = positive_bounds([r.R for r in rows]...; pad_low=0.08, pad_high=0.08)
    apply_log10_y!(axR, yminR, ymaxR)

    # panel ii
    axWi.xticks = eps_ticks
    xlims!(axWi, eps_xlim...)
    apply_linear_midpoint_minorticks!(axWi, eps_ticks; axis=:x)
    yminWi, ymaxWi = positive_bounds([r.Wi for r in rows]...; pad_low=0.08, pad_high=0.08)
    apply_log10_y!(axWi, yminWi, ymaxWi)

    # panels iii and iv share same Wi axis
    xminX, xmaxX = positive_bounds(Wi_sty, [r.Wi for r in rows]...; pad_low=0.08, pad_high=0.08)
    xminX = 0.1
    apply_log10_x!(axM, xminX, xmaxX)
    apply_log10_x!(axEta, xminX, xmaxX)
    apply_log10_y!(axM, yminM, ymaxM)
    apply_log10_y!(axEta, yminEta, ymaxEta)

    linkxaxes!(axM, axEta)

    # keep major/minor ticks on panel iii, but hide tick labels and xlabel
    axM.xticklabelsvisible = false
    axM.xlabelvisible = false

    # ---------- manual labels: panel tags ----------
    reltext!(axR,   labels.tag_i[1],   labels.tag_i[2],   labels.tag_i[3];   sty=sty, color=:black, fs_scale=1.35)
    reltext!(axWi,  labels.tag_ii[1],  labels.tag_ii[2],  labels.tag_ii[3];  sty=sty, color=:black, fs_scale=1.35)
    reltext!(axM,   labels.tag_iii[1], labels.tag_iii[2], labels.tag_iii[3]; sty=sty, color=:black, fs_scale=1.35)
    reltext!(axEta, labels.tag_iv[1],  labels.tag_iv[2],  labels.tag_iv[3];  sty=sty, color=:black, fs_scale=1.35)

    # ---------- manual labels: De headers ----------
    reltext!(axR,   labels.de_head_R[1],   labels.de_head_R[2],   L"\mathrm{De} ="; sty=sty, color=soft_gray(sty))
    reltext!(axWi,  labels.de_head_Wi[1],  labels.de_head_Wi[2],  L"\mathrm{De} ="; sty=sty, color=soft_gray(sty))
    reltext!(axM,   labels.de_head_M[1],   labels.de_head_M[2],   L"\mathrm{De} ="; sty=sty, color=soft_gray(sty))
    reltext!(axEta, labels.de_head_eta[1], labels.de_head_eta[2], L"\mathrm{De} ="; sty=sty, color=soft_gray(sty))

    # ---------- manual labels: De numbers ----------
    for row in rows
        xR, yR     = get_pos(labels.de_R, row.De)
        xWi, yWi   = get_pos(labels.de_Wi, row.De)
        xM, yM     = get_pos(labels.de_M, row.De)
        xEta, yEta = get_pos(labels.de_eta, row.De)

        reltext!(axR,   xR,   yR,   string(PaperFig.pretty_num(row.De)); sty=sty, color=darken(row.col, 0.08))
        reltext!(axWi,  xWi,  yWi,  string(PaperFig.pretty_num(row.De)); sty=sty, color=darken(row.col, 0.08))
        reltext!(axM,   xM,   yM,   string(PaperFig.pretty_num(row.De)); sty=sty, color=darken(row.col, 0.08))
        reltext!(axEta, xEta, yEta, string(PaperFig.pretty_num(row.De)); sty=sty, color=darken(row.col, 0.08))
    end

    # ---------- manual labels: 2/3 ----------
    reltext!(axWi,  labels.twothirds_Wi[1],  labels.twothirds_Wi[2],  L"2/3"; sty=sty, color=sty.ref_dark)
    reltext!(axM,   labels.twothirds_M[1],   labels.twothirds_M[2],   L"2/3"; sty=sty, color=sty.ref_dark)
    reltext!(axEta, labels.twothirds_eta[1], labels.twothirds_eta[2], L"2/3"; sty=sty, color=sty.ref_dark)

    # ---------- manual labels: steady ----------
    reltext!(axM,   labels.steady_M[1],   labels.steady_M[2],   "steady"; sty=sty, color=sty.ref_dark)
    reltext!(axEta, labels.steady_eta[1], labels.steady_eta[2], "steady"; sty=sty, color=sty.ref_dark)

    # ---------- manual labels: parameter / phi block ----------
    if show_param_block
        for (txt, x, y, s) in labels.param_lines
            reltext!(axR, x, y, txt; sty=sty, color=soft_gray(sty), fs_scale=s)
        end
    else
        reltext!(axR, labels.phi_only[1], labels.phi_only[2], labels.phi_only[3];
            sty=sty, color=soft_gray(sty))
    end

    return nothing
end

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir = length(args) >= 2 ? args[2] : joinpath(paper_root(), "figures")
    ensure_dir(figdir)

    curves = load_curves(harvested_dir)
    feats = load_features_selected(harvested_dir)

    sty = style()

    # --------------------------------------------------------
    # size controls
    # --------------------------------------------------------
    w_small_cm = 3.0
    w_big_cm   = 6.0
    row_h_cm   = 6.0

    col_gap_cm   = 0.40
    row_gap_cm   = 0.60
    inner_gap_cm = 0.25

    fig_pad_pt = (1, 1, 1, 1)

    fig_w_cm = 17.0
    fig_h_cm = 15.0

    PaperFig.activate_theme!(sty)

    fig = Figure(
        size=(PaperFig.cm_to_pt(fig_w_cm), PaperFig.cm_to_pt(fig_h_cm)),
        backgroundcolor=:white,
        figure_padding=fig_pad_pt,
    )

    # left and middle panels unchanged
    axA1 = Axis(fig[1, 1], xlabel=L"t", ylabel=L"R")
    axA2 = Axis(fig[1, 2], xlabel=L"\varepsilon", ylabel=L"\mathrm{Wi}")
    axB1 = Axis(fig[2, 1], xlabel=L"t", ylabel=L"R")
    axB2 = Axis(fig[2, 2], xlabel=L"\varepsilon", ylabel=L"\mathrm{Wi}")

    # split old panel (iii) into stacked (iii)/(iv)
    glA = GridLayout(fig[1, 3])
    glB = GridLayout(fig[2, 3])

    axA3 = Axis(glA[1, 1], ylabel=L"M")
    axA4 = Axis(glA[2, 1], xlabel=L"\mathrm{Wi}", ylabel=L"\tilde{\eta}_{\mathrm{p}}")

    axB3 = Axis(glB[1, 1], ylabel=L"M")
    axB4 = Axis(glB[2, 1], xlabel=L"\mathrm{Wi}", ylabel=L"\tilde{\eta}_{\mathrm{p}}")

    # explicit axis block sizes
    colsize!(fig.layout, 1, Fixed(PaperFig.cm_to_pt(w_small_cm)))
    colsize!(fig.layout, 2, Fixed(PaperFig.cm_to_pt(w_small_cm)))
    colsize!(fig.layout, 3, Fixed(PaperFig.cm_to_pt(w_big_cm)))

    rowsize!(fig.layout, 1, Fixed(PaperFig.cm_to_pt(row_h_cm)))
    rowsize!(fig.layout, 2, Fixed(PaperFig.cm_to_pt(row_h_cm)))

    colgap!(fig.layout, PaperFig.cm_to_pt(col_gap_cm))
    rowgap!(fig.layout, PaperFig.cm_to_pt(row_gap_cm))

    rowgap!(glA, PaperFig.cm_to_pt(inner_gap_cm))
    rowgap!(glB, PaperFig.cm_to_pt(inner_gap_cm))

    DEBUG_GUIDES = false

    drag_model = "constant"
    NK = 3000
    hK = 0.05

    phiA = 0.01
    phiB = 0.5

    DeA = [0.1, 1.0, 3.0, 10.0]
    DeB = [0.1, 1.0, 10.0, 30.0, 100.0]

    # explicit x ticks/ranges
    t_ticks_A = [0.0, 5.0, 10.0, 15.0, 20.0]
    t_xlim_A = (0.0, 20.0)
    eps_ticks_A = [0.0, 5.0, 10.0, 15.0, 20.0]
    eps_xlim_A = (0.0, 20.0)

    t_ticks_B = [0.0, 50.0, 100.0, 150.0, 200.0]
    t_xlim_B = (0.0, 200.0)
    eps_ticks_B = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    eps_xlim_B = (0.0, 25.0)

    # --------------------------------------------------------
    # manual label positions
    # --------------------------------------------------------
    LAB_A = (
        tag_i=(0.75, 0.85, "A (i)"),
        tag_ii=(0.80, 0.85, "(ii)"),
        tag_iii=(0.80, 0.80, "(iii)"),
        tag_iv=(0.80, 0.80, "(iv)"),

        de_head_R=(0.70, 0.70),
        de_head_Wi=(0.10, 0.77),
        de_head_M=(0.35, 0.64),
        de_head_eta=(0.35, 0.64),

        de_R=Dict(
            10.0 => (0.90, 0.70),
            3.0  => (0.80, 0.48),
            1.0  => (0.75, 0.40),
            0.1  => (0.75, 0.14),
        ),
        de_Wi=Dict(
            10.0 => (0.30, 0.77),
            3.0  => (0.50, 0.73),
            1.0  => (0.60, 0.73),
            0.1  => (0.80, 0.73),
        ),
        de_M=Dict(
            10.0 => (0.55, 0.18),
            3.0  => (0.55, 0.48),
            1.0  => (0.45, 0.48),
            0.1  => (0.46, 0.64),
        ),
        de_eta=Dict(
            10.0 => (0.5, 0.18),
            3.0  => (0.56, 0.48),
            1.0  => (0.5, 0.55),
            0.1  => (0.46, 0.64),
        ),

        twothirds_Wi=(0.80, 0.47),
        twothirds_M=(0.28, 0.84),
        twothirds_eta=(0.28, 0.84),

        steady_M=(0.08, 0.38),
        steady_eta=(0.08, 0.38),

        param_lines=[
            ("FENE-P", 0.10, 0.18, 1.10),
            (L"\phi = 0.01", 0.08, 0.11, 1.00),
            (L"N_{\mathrm{K}} = 3000", 0.04, 0.04, 1.00),
        ],
        phi_only=(0.0, 0.0, ""),
    )

    LAB_B = (
        tag_i=(0.75, 0.88, "B (i)"),
        tag_ii=(0.80, 0.88, "(ii)"),
        tag_iii=(0.80, 0.80, "(iii)"),
        tag_iv=(0.80, 0.80, "(iv)"),

        de_head_R=(0.60, 0.79),
        de_head_Wi=(0.20, 0.84),
        de_head_M=(0.7, 0.9),
        de_head_eta=(0.05, 0.22),

        de_R=Dict(
            100.0 => (0.81, 0.79),
            30.0  => (0.80, 0.60),
            10.0  => (0.50, 0.50),
            1.0   => (0.18, 0.40),
            0.1   => (0.10, 0.20),
        ),
        de_Wi=Dict(
            100.0 => (0.40, 0.84),
            30.0  => (0.45, 0.75),
            10.0  => (0.60, 0.75),
            1.0   => (0.75, 0.75),
            0.1   => (0.87, 0.75),
        ),
        de_M=Dict(
            100.0 => (0.7, 0.8),
            30.0  => (0.74, 0.6),
            10.0  => (0.55, 0.4),
            1.0   => (0.4, 0.2),
            0.1   => (0.28, 0.2),
        ),
        de_eta=Dict(
            100.0 => (0.85, 0.20),
            30.0  => (0.7, 0.30),
            10.0  => (0.53, 0.40),
            1.0   => (0.38, 0.40),
            0.1   => (0.15, 0.22),
        ),

        twothirds_Wi=(0.80, 0.47),
        twothirds_M=(0.28, 0.84),
        twothirds_eta=(0.28, 0.84),

        steady_M=(0.10, 0.60),
        steady_eta=(0.10, 0.60),

        param_lines=[
            ("FENE-P", 0.60, 0.18, 1.10),
            (L"\phi = 0.5", 0.58, 0.11, 1.00),
            (L"N_{\mathrm{K}} = 3000", 0.54, 0.04, 1.00),
        ],
        phi_only=(0.06, 0.06, L"\phi = 0.5"),
    )

    # steady curves
    styA = pick_curve_run(curves; flow_family="styext", drag_model=drag_model, NK=NK, hK_star=hK, phi=phiA)
    styB = pick_curve_run(curves; flow_family="styext", drag_model=drag_model, NK=NK, hK_star=hK, phi=phiB)

    plot_row!(
        axA1, axA2, axA3, axA4;
        curves=curves, feats=feats, stycurve=styA,
        phi=phiA, Devals=DeA, labels=LAB_A,
        sty=sty, drag_model=drag_model, NK=NK, hK=hK,
        show_param_block=true, debug_guides=DEBUG_GUIDES,
        t_ticks=t_ticks_A, t_xlim=t_xlim_A,
        eps_ticks=eps_ticks_A, eps_xlim=eps_xlim_A,
    )

    plot_row!(
        axB1, axB2, axB3, axB4;
        curves=curves, feats=feats, stycurve=styB,
        phi=phiB, Devals=DeB, labels=LAB_B,
        sty=sty, drag_model=drag_model, NK=NK, hK=hK,
        show_param_block=true, debug_guides=DEBUG_GUIDES,
        t_ticks=t_ticks_B, t_xlim=t_xlim_B,
        eps_ticks=eps_ticks_B, eps_xlim=eps_xlim_B,
    )

    out_pdf = joinpath(figdir, "fig3_fenep_detailed.pdf")
    out_png = joinpath(figdir, "fig3_fenep_detailed.png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)

    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
end

main(ARGS)