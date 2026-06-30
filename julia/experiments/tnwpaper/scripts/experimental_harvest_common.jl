#!/usr/bin/env julia

module ExperimentalHarvestCommon

using CSV
using DataFrames

try
    using XLSX
catch
    error("This harvester requires XLSX.jl. Install once with: import Pkg; Pkg.add(\"XLSX\")")
end

function _tnw_include_path()
    candidates = [
        joinpath(@__DIR__, "..", "src", "TNW.jl"),          # if scripts live in .../tnwpaper/scripts
        joinpath(@__DIR__, "..", "..", "src", "TNW.jl"),    # if scripts live in .../tnwpaper/scripts/fig5_harvesters
    ]
    for p in candidates
        if isfile(p)
            return normpath(p)
        end
    end
    error("Could not locate TNW.jl from $(abspath(@__DIR__)). Tried:\n  " * join(candidates, "\n  "))
end

include(_tnw_include_path())
using .TNW

export repo_root, default_matlab_tnwpaper_root, default_plan_xlsx,
    default_exptl_comparison_root, default_harvested_root,
    canonicalize_headers, normalize_enabled, num2token,
    read_plan_dataset_selection, processed_dir_from_plan,
    read_csv_checked, add_dataset_column!, keep_present_cols,
    build_sample_manifest, build_dataset_meta,
    build_panel_unnormalized, build_panel_normalized

# ------------------------------------------------------------------
# Repo roots / defaults for your actual split tree
# ------------------------------------------------------------------

repo_root() = normpath(joinpath(paper_root(), "..", "..", ".."))
default_matlab_tnwpaper_root() = joinpath(repo_root(), "matlab", "experiments", "tnwpaper")
default_plan_xlsx() = joinpath(default_matlab_tnwpaper_root(), "parameter-sets.xlsx")

# Julia side expects the symlink:
# julia/experiments/tnwpaper/outputs/exptl_comparison -> ../../../../matlab/experiments/tnwpaper/outputs/exptl_comparison
default_exptl_comparison_root() = joinpath(paper_root(), "outputs", "exptl_comparison")
default_harvested_root() = joinpath(paper_root(), "outputs", "harvested")

# ------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------

function canonicalize_headers(header_row)
    headers = Vector{String}(undef, length(header_row))
    for i in eachindex(header_row)
        h = string(header_row[i])
        h = lowercase(strip(h))
        h = replace(h, r"\s+" => "_")
        h = replace(h, r"[^a-z0-9_]" => "")
        headers[i] = h
    end
    return headers
end

function normalize_enabled(x)
    tf = falses(length(x))
    for i in eachindex(x)
        xi = x[i]
        if xi isa Bool
            tf[i] = xi
        elseif xi isa Number
            tf[i] = xi != 0
        else
            s = lowercase(strip(string(xi)))
            tf[i] = s in ("1", "true", "yes", "y", "on")
        end
    end
    return tf
end

function num2token(x)
    s = repr(round(Float64(x), sigdigits=12))
    s = replace(s, "." => "p")
    s = replace(s, "-" => "m")
    return s
end

# ------------------------------------------------------------------
# Plan parsing
# ------------------------------------------------------------------

function read_plan_dataset_selection(xlsx_path::AbstractString;
    plan_sheet_name::AbstractString="Plan_Exptl",
    dataset_name::AbstractString)

    isfile(xlsx_path) || error("parameter-sets.xlsx not found: $xlsx_path")

    xf = XLSX.readxlsx(xlsx_path)
    XLSX.hassheet(xf, plan_sheet_name) || error(
        "Sheet $plan_sheet_name not found in $xlsx_path. Available sheets: " *
        join(XLSX.sheetnames(xf), ", ")
    )
    raw = DataFrame(XLSX.readtable(xlsx_path, plan_sheet_name))

    vars0 = string.(names(raw))
    vars = canonicalize_headers(vars0)
    rename!(raw, Pair.(names(raw), Symbol.(vars)))

    have = Set(string.(names(raw)))
    ("folder" in have || "out_root" in have) || error("Plan_Exptl must contain folder or out_root")
    "hk_star" in have || error("Plan_Exptl must contain hk_star")

    if !("enabled" in have)
        raw.enabled = trues(nrow(raw))
    end

    folder = "folder" in have ? lowercase.(strip.(string.(raw.folder))) : fill("", nrow(raw))
    outroot = "out_root" in have ? lowercase.(strip.(string.(raw.out_root))) : fill("", nrow(raw))
    enabled = normalize_enabled(raw.enabled)
    ds = lowercase(strip(string(dataset_name)))

    keep = enabled .& ((folder .== ds) .| occursin.(Ref("/" * ds), outroot) .| occursin.(Ref(ds), outroot))
    T = raw[keep, :]

    nrow(T) > 0 || error("No enabled rows found in $plan_sheet_name for dataset $ds")

    hk = Float64[]
    for v in T.hk_star
        try
            x = Float64(v)
            isfinite(x) && push!(hk, x)
        catch
        end
    end
    !isempty(hk) || error("No finite hk_star values found for dataset $ds")
    hk_unique = unique(round.(hk; digits=12))
    length(hk_unique) == 1 || error("Enabled rows for dataset $ds do not share a unique hK_star. Found: $(hk_unique)")
    hK_star = hk_unique[1]

    out_root_base = if "out_root" in have
        vals = unique(strip.(string.(T.out_root)))
        vals = filter(!isempty, vals)
        isempty(vals) ? "tnwpaper/outputs/exptl_comparison/$(ds)" : vals[1]
    else
        "tnwpaper/outputs/exptl_comparison/$(ds)"
    end

    fe_model_default = "fe_model" in have ? first(unique(string.(T.fe_model))) : missing

    return (
        table=T,
        dataset=ds,
        n_enabled_rows=nrow(T),
        hK_star=hK_star,
        hsk_token=num2token(hK_star),
        trial_tag="hsK_" * num2token(hK_star),
        out_root_base=out_root_base,
        fe_model_default=fe_model_default,
    )
end

function processed_dir_from_plan(sel; exptl_comparison_root::AbstractString=default_exptl_comparison_root())
    ds = sel.dataset
    return joinpath(exptl_comparison_root, ds, sel.trial_tag, "processed")
end

# ------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------

function read_csv_checked(path::AbstractString)
    isfile(path) || error("Missing required CSV: $path")
    DataFrame(CSV.File(path; silencewarnings=true))
end

function add_dataset_column!(df::DataFrame, dataset::AbstractString)
    df.dataset = fill(String(dataset), nrow(df))
    return df
end

function keep_present_cols(df::DataFrame, cols::Vector{Symbol})
    got = [c for c in cols if c in names(df)]
    return df[:, got]
end

function _first_nonmissing(v)
    for x in v
        if !(ismissing(x) || (x isa AbstractString && isempty(strip(String(x)))))
            return x
        end
    end
    return missing
end

# ------------------------------------------------------------------
# Harvest table builders
# ------------------------------------------------------------------

function build_sample_manifest(dataset::AbstractString, expt::DataFrame, model_all::DataFrame, model_c2d2::DataFrame, plan_table::DataFrame)
    ids_expt = "sample_id" in names(expt) ? String.(unique(expt.sample_id)) : String[]
    ids_mod = "sample_id" in names(model_all) ? String.(unique(model_all.sample_id)) : String[]
    sample_ids = unique(vcat(ids_expt, ids_mod))

    rows = DataFrame(
        dataset=String[],
        sample_id=String[],
        NK_plan=Any[],
        phi_plan=Any[],
        NK_expt=Any[],
        phi_expt=Any[],
        hK_star=Any[],
        fe_model=Any[],
        has_c2d2=Bool[],
        has_constant=Bool[],
        n_expt_points=Int[],
        n_model_points_c2d2=Int[],
    )

    plan_has_sample = "sample_id" in string.(names(plan_table))

    for sid_str in sample_ids
        Te = "sample_id" in names(expt) ? expt[String.(expt.sample_id).==sid_str, :] : expt[1:0, :]
        Tm = "sample_id" in names(model_c2d2) ? model_c2d2[String.(model_c2d2.sample_id).==sid_str, :] : model_c2d2[1:0, :]
        Ta = "sample_id" in names(model_all) ? model_all[String.(model_all.sample_id).==sid_str, :] : model_all[1:0, :]

        NK_expt = :NK in names(Te) ? _first_nonmissing(Te.NK) : missing
        phi_expt = :phi in names(Te) ? _first_nonmissing(Te.phi) : missing
        hK_star = :hK_star in names(Ta) ? _first_nonmissing(Ta.hK_star) :
                  (:hK_star in names(Te) ? _first_nonmissing(Te.hK_star) : missing)
        fe_model = :fe_model in names(Ta) ? _first_nonmissing(Ta.fe_model) : missing

        NK_plan = missing
        phi_plan = missing
        if plan_has_sample
            Tp = plan_table[String.(plan_table.sample_id).==sid_str, :]
            if nrow(Tp) > 0
                NK_plan = :nk in names(Tp) ? _first_nonmissing(Tp.nk) : missing
                phi_plan = :phi in names(Tp) ? _first_nonmissing(Tp.phi) : missing
            end
        end

        has_c2d2 = :drag_model in names(Ta) ? any(String.(Ta.drag_model) .== "c2d2") : nrow(Tm) > 0
        has_constant = :drag_model in names(Ta) ? any(String.(Ta.drag_model) .== "constant") : false

        push!(rows, (
            String(dataset),
            sid_str,
            NK_plan,
            phi_plan,
            NK_expt,
            phi_expt,
            hK_star,
            fe_model,
            has_c2d2,
            has_constant,
            nrow(Te),
            nrow(Tm),
        ))
    end
    return rows
end

function build_dataset_meta(dataset::AbstractString, dataset_sheet::AbstractString, sel, processed_dir::AbstractString,
    renorm::DataFrame, model_all::DataFrame, model_c2d2::DataFrame, expt::DataFrame)
    return DataFrame(
        dataset=[String(dataset)],
        dataset_sheet=[String(dataset_sheet)],
        hK_star_selected=[sel.hK_star],
        source_processed_dir=[String(processed_dir)],
        n_enabled_plan_rows=[sel.n_enabled_rows],
        n_samples=["sample_id" in names(expt) ? length(unique(expt.sample_id)) : missing],
        n_model_rows_all=[nrow(model_all)],
        n_model_rows_c2d2=[nrow(model_c2d2)],
        n_expt_rows=[nrow(expt)],
        theta=[:theta in names(renorm) ? _first_nonmissing(renorm.theta) : missing],
        fe_model_default=[sel.fe_model_default],
    )
end

function build_panel_unnormalized(dataset::AbstractString, model_c2d2::DataFrame, expt::DataFrame)
    model = DataFrame(
        dataset=fill(String(dataset), nrow(model_c2d2)),
        sample_id="sample_id" in names(model_c2d2) ? String.(model_c2d2.sample_id) : fill("", nrow(model_c2d2)),
        source_kind=fill("model", nrow(model_c2d2)),
        drag_model=:drag_model in names(model_c2d2) ? String.(model_c2d2.drag_model) : fill("c2d2", nrow(model_c2d2)),
        De=model_c2d2.De,
        Wi_e=model_c2d2.Wi_e,
        NK=:NK in names(model_c2d2) ? model_c2d2.NK : fill(missing, nrow(model_c2d2)),
        phi=:phi in names(model_c2d2) ? model_c2d2.phi : fill(missing, nrow(model_c2d2)),
        hK_star=:hK_star in names(model_c2d2) ? model_c2d2.hK_star : fill(missing, nrow(model_c2d2)),
    )
    ex = DataFrame(
        dataset=fill(String(dataset), nrow(expt)),
        sample_id="sample_id" in names(expt) ? String.(expt.sample_id) : fill("", nrow(expt)),
        source_kind=fill("experiment", nrow(expt)),
        drag_model=fill(missing, nrow(expt)),
        De=expt.De,
        Wi_e=expt.Wi_e,
        NK=:NK in names(expt) ? expt.NK : fill(missing, nrow(expt)),
        phi=:phi in names(expt) ? expt.phi : fill(missing, nrow(expt)),
        hK_star=:hK_star in names(expt) ? expt.hK_star : fill(missing, nrow(expt)),
    )
    return vcat(model, ex; cols=:union)
end

function build_panel_normalized(dataset::AbstractString, model_c2d2::DataFrame, expt::DataFrame)
    model = DataFrame(
        dataset=fill(String(dataset), nrow(model_c2d2)),
        sample_id="sample_id" in names(model_c2d2) ? String.(model_c2d2.sample_id) : fill("", nrow(model_c2d2)),
        source_kind=fill("model", nrow(model_c2d2)),
        drag_model=:drag_model in names(model_c2d2) ? String.(model_c2d2.drag_model) : fill("c2d2", nrow(model_c2d2)),
        De_bar=:De_bar in names(model_c2d2) ? model_c2d2.De_bar : model_c2d2.De ./ model_c2d2.Debar,
        Wi_e_bar=:Wi_e_bar in names(model_c2d2) ? model_c2d2.Wi_e_bar : model_c2d2.Wi_e ./ model_c2d2.Wiebar,
        Debar=:Debar in names(model_c2d2) ? model_c2d2.Debar : fill(missing, nrow(model_c2d2)),
        Wiebar=:Wiebar in names(model_c2d2) ? model_c2d2.Wiebar : fill(missing, nrow(model_c2d2)),
        phibar=:phibar in names(model_c2d2) ? model_c2d2.phibar : fill(missing, nrow(model_c2d2)),
        hK_star=:hK_star in names(model_c2d2) ? model_c2d2.hK_star : fill(missing, nrow(model_c2d2)),
    )
    ex = DataFrame(
        dataset=fill(String(dataset), nrow(expt)),
        sample_id="sample_id" in names(expt) ? String.(expt.sample_id) : fill("", nrow(expt)),
        source_kind=fill("experiment", nrow(expt)),
        drag_model=fill(missing, nrow(expt)),
        De_bar=:De_bar in names(expt) ? expt.De_bar : expt.De ./ expt.Debar,
        Wi_e_bar=:Wi_e_bar in names(expt) ? expt.Wi_e_bar : expt.Wi_e ./ expt.Wiebar,
        Debar=:Debar in names(expt) ? expt.Debar : fill(missing, nrow(expt)),
        Wiebar=:Wiebar in names(expt) ? expt.Wiebar : fill(missing, nrow(expt)),
        phibar=:phibar in names(expt) ? expt.phibar : fill(missing, nrow(expt)),
        hK_star=:hK_star in names(expt) ? expt.hK_star : fill(missing, nrow(expt)),
    )
    return vcat(model, ex; cols=:union)
end

end # module
