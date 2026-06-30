#!/usr/bin/env julia

using CSV
using DataFrames
using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "TNW.jl"))
using .TNW

include(joinpath(TNW.julia_root(), "models", "utils", "paperfig.jl"))
using .PaperFig

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

function pick_rep_run(curves::DataFrame; drag_model="constant", NK=3000, hK_star=0.05, phi=0.5, De=0.1)
    d = curves
    mask = trues(nrow(d))
    hasproperty(d, :flow_family) && (mask .&= d.flow_family .== "caber")
    hasproperty(d, :drag_model) && (mask .&= d.drag_model .== drag_model)
    hasproperty(d, :NK) && (mask .&= d.NK .== NK)
    hasproperty(d, :hK_star) && (mask .&= approx_mask(d.hK_star, hK_star))
    hasproperty(d, :phi) && (mask .&= approx_mask(d.phi, phi))
    if hasproperty(d, :De_run)
        mask .&= approx_mask(coalesce.(d.De_run, NaN), De)
    elseif hasproperty(d, :De)
        mask .&= approx_mask(coalesce.(d.De, NaN), De)
    end
    dd = d[mask, :]
    isempty(dd) && error("No representative run found.")
    rid = unique(dd.run_id)[1]
    di = dd[dd.run_id.==rid, :]
    sort!(di, :t)
    di
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
    f[1, :]
end

function soft_red_fit_line(t, strain, R, strain_e; dstrain=10.0)
    mask = (strain .>= strain_e - dstrain) .& (strain .<= strain_e + dstrain) .&
           (R .> 0) .& isfinite.(R) .& isfinite.(t)

    if count(mask) < 2
        error("Not enough points in selected strain window for red guide fit.")
    end

    tfit = t[mask]
    Rfit = R[mask]

    X = hcat(ones(length(tfit)), tfit)
    β = X \ log.(Rfit)

    tg = range(minimum(tfit), maximum(tfit), length=100)
    Rg = exp.(β[1] .+ β[2] .* tg)

    return tg, Rg
end

function main(args)
    harvested_dir = length(args) >= 1 ? args[1] : default_harvested_dir()
    figdir = length(args) >= 2 ? args[2] : joinpath(paper_root(), "figures")
    ensure_dir(figdir)

    curves = load_curves(harvested_dir)
    feats = load_features_selected(harvested_dir)

    rep_drag = "constant"
    rep_NK = 3000
    rep_hK = 0.05
    rep_phi = 0.5
    rep_De = 1.0

    d = pick_rep_run(curves; drag_model=rep_drag, NK=rep_NK, hK_star=rep_hK, phi=rep_phi, De=rep_De)
    frow = pick_feature_row(feats, d.run_id[1])

    strain = Vector{Float64}(d.estrain)
    t = Vector{Float64}(d.t)
    Wi = Vector{Float64}(d.Wi)
    eta_p = Vector{Float64}(d.etaE1)
    R = exp.(-strain ./ 2)

    idx_e = clamp(Int(frow.feature_idx), 1, nrow(d))
    strain_e = strain[idx_e]
    Wi_e = Wi[idx_e]
    t_e = t[idx_e]
    R_e = R[idx_e]
    tg, Rg = soft_red_fit_line(t, strain, R, strain_e; dstrain=0.1 * maximum(strain))

    fig, sty = new_figure(:fig1_textwidth_row3, height_cm=6.0)
    axA = make_axis(fig[1, 1]; xlabel=L"t", ylabel=L"R")
    axB = make_axis(fig[1, 2]; xlabel=L"\varepsilon", ylabel=L"\mathrm{Wi}")
    axC = make_axis(fig[1, 3]; xlabel=L"\varepsilon", ylabel=L"\tilde{\eta}_{\mathrm{p}}")
    colgap!(fig.layout, 14)

    DEBUG_GUIDES = false

    if DEBUG_GUIDES
        relative_guide_grid!(axA; color=RGBAf(0,0,0,0.06), linewidth=0.4)
        relative_guide_grid!(axB; color=RGBAf(0,0,0,0.06), linewidth=0.4)
        relative_guide_grid!(axC; color=RGBAf(0,0,0,0.06), linewidth=0.4)
    end


    # Panel A
    lines!(axA, t, R; color=soft_gray(sty), linewidth=sty.line_lw)
    lines!(axA, tg, Rg; color=soft_red(sty), linewidth=0.9)
    selected_feature_marker!(axA, t_e, R_e; sty=sty, color=soft_red(sty))
    regime_labels!(axA, [0.61, 0.73, 0.87], [0.9, 0.75, 0.3], ["viscous", "elastic", "viscous"]; sty=sty)
    yminA = 10.0^(floor(log10(minimum(R))) - 0.1)
    apply_log10_y!(axA, yminA, 1.05)
    xlims!(axA, minimum(t), maximum(t))
    panel_letter!(axA, "(i)"; sty=sty)
    parameter_block!(axA, [
            L"\phi = %$(PaperFig.pretty_num(rep_phi))",
            L"N_{\mathrm{K}} = %$(PaperFig.pretty_num(rep_NK))",
            L"\mathrm{De} = %$(PaperFig.pretty_num(rep_De))",
        ]; sty=sty, x=0.33, y0=0.22, dy=0.075, color=soft_gray(sty))

    # Panel B
    lines!(axB, strain, Wi; color=soft_gray(sty), linewidth=sty.line_lw)
    yminB = 10.0^(floor(log10(minimum(Wi[Wi.>0]))) - 0.15)
    ymaxB = 10.0^(ceil(log10(maximum(Wi))) + 0.10)
    lines!(axB, [0.0, strain_e], [Wi_e, Wi_e]; color=soft_red(sty), linewidth=1.1, linestyle=:dash)
    selected_feature_marker!(axB, strain_e, Wi_e; sty=sty, color=soft_red(sty))
    regime_labels!(axB, [0.1], [0.39], [L"\mathrm{Wi}_\mathrm{e}"]; sty=sty, color=soft_red(sty))
    regime_labels!(axB, [0.2, 0.45, 0.81], [0.2, 0.4, 0.65], ["viscous", "elastic", "viscous"]; sty=sty)
    apply_log10_y!(axB, yminB, ymaxB)
    xlims!(axB, minimum(strain), maximum(strain))
    xt_major = [0.0, 5.0, 10.0, 15.0, 20.0]
    xlims!(axB, 0.0, maximum(strain))
    apply_linear_midpoint_minorticks!(axB, xt_major; axis=:x)
    panel_letter!(axB, "(ii)"; sty=sty)

    # Panel C
    goodC = (eta_p .> 0) .& isfinite.(eta_p)
    strainC = strain[goodC]
    etaC = eta_p[goodC]
    lines!(axC, strainC, etaC; color=soft_gray(sty), linewidth=sty.line_lw)
    yminC = 10.0^(floor(log10(minimum(etaC))) - 0.1)
    ymaxC = 10.0^(ceil(log10(maximum(etaC))) + 0.1)
    apply_log10_y!(axC, yminC, ymaxC)
    xlims!(axC, 0.0, maximum(strain))
    apply_linear_midpoint_minorticks!(axC, xt_major; axis=:x)
    panel_letter!(axC, "(iii)"; sty=sty)

    out_pdf = joinpath(figdir, "fig1_intro.pdf")
    out_png = joinpath(figdir, "fig1_intro.png")
    paper_save(fig, out_pdf)
    paper_save(fig, out_png)

    println("[TNW] Wrote:")
    println("  $out_pdf")
    println("  $out_png")
end

main(ARGS)
