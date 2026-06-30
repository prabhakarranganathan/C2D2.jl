#!/usr/bin/env julia

using CSV
using DataFrames

include(joinpath(@__DIR__, "experimental_harvest_common.jl"))
using .ExperimentalHarvestCommon

function pretty_dataset_sheet_name(dataset_name::String)
    ds = lowercase(strip(dataset_name))
    return ds == "gaillard"  ? "Gaillard"  :
           ds == "anna"      ? "Anna"      :
           ds == "clasen"    ? "Clasen"    :
           ds == "calabrese" ? "Calabrese" :
           uppercase(first(ds)) * ds[2:end]
end

function harvest_one(dataset_name::String, xlsx_path::String, plan_sheet_name::String, out_root::String)
    sel = read_plan_dataset_selection(xlsx_path; plan_sheet_name=plan_sheet_name, dataset_name=dataset_name)
    processed_dir = processed_dir_from_plan(sel)
    outdir = joinpath(out_root, "experimental_" * dataset_name)
    isdir(outdir) || mkpath(outdir)

    renorm = read_csv_checked(joinpath(processed_dir, "renorm_lookup.csv"))
    model_all = read_csv_checked(joinpath(processed_dir, "model_pipkin_curves.csv"))
    expt = read_csv_checked(joinpath(processed_dir, "expt_pipkin_points.csv"))

    add_dataset_column!(renorm, dataset_name)
    add_dataset_column!(model_all, dataset_name)
    add_dataset_column!(expt, dataset_name)

    model_c2d2 = :drag_model in names(model_all) ? model_all[String.(model_all.drag_model) .== "c2d2", :] : copy(model_all)
    sample_manifest = build_sample_manifest(dataset_name, expt, model_all, model_c2d2, sel.table)
    dataset_meta = build_dataset_meta(dataset_name, pretty_dataset_sheet_name(dataset_name), sel, processed_dir, renorm, model_all, model_c2d2, expt)
    panel_unnorm = build_panel_unnormalized(dataset_name, model_c2d2, expt)
    panel_norm = build_panel_normalized(dataset_name, model_c2d2, expt)

    CSV.write(joinpath(outdir, "dataset_meta.csv"), dataset_meta)
    CSV.write(joinpath(outdir, "sample_manifest.csv"), sample_manifest)
    CSV.write(joinpath(outdir, "renorm_lookup.csv"), renorm)
    CSV.write(joinpath(outdir, "model_curves_all.csv"), model_all)
    CSV.write(joinpath(outdir, "model_curves_c2d2.csv"), model_c2d2)
    CSV.write(joinpath(outdir, "expt_points.csv"), expt)
    CSV.write(joinpath(outdir, "panel_unnormalized.csv"), panel_unnorm)
    CSV.write(joinpath(outdir, "panel_normalized.csv"), panel_norm)

    return (
        dataset_meta = dataset_meta,
        sample_manifest = sample_manifest,
        expt_points = expt,
        panel_unnormalized = panel_unnorm[panel_unnorm.source_kind .== "experiment", :],
        panel_normalized = panel_norm[panel_norm.source_kind .== "experiment", :],
    )
end

function main(args)
    xlsx_path = length(args) >= 1 ? String(args[1]) : default_plan_xlsx()
    plan_sheet_name = length(args) >= 2 ? String(args[2]) : "Plan_Exptl"
    out_root = length(args) >= 3 ? String(args[3]) : default_harvested_root()

    datasets = ["gaillard", "anna", "clasen", "calabrese"]
    all_meta = DataFrame()
    all_manifest = DataFrame()
    all_expt = DataFrame()
    all_expt_unnorm = DataFrame()
    all_expt_norm = DataFrame()

    println("[TNW] Harvesting all experimental datasets")
    println("  xlsx     = $(xlsx_path)")
    println("  out_root = $(out_root)")

    for ds in datasets
        println("[TNW] Harvesting $(ds)")
        out = harvest_one(ds, xlsx_path, plan_sheet_name, out_root)
        all_meta = isempty(all_meta) ? out.dataset_meta : vcat(all_meta, out.dataset_meta; cols=:union)
        all_manifest = isempty(all_manifest) ? out.sample_manifest : vcat(all_manifest, out.sample_manifest; cols=:union)
        all_expt = isempty(all_expt) ? out.expt_points : vcat(all_expt, out.expt_points; cols=:union)
        all_expt_unnorm = isempty(all_expt_unnorm) ? out.panel_unnormalized : vcat(all_expt_unnorm, out.panel_unnormalized; cols=:union)
        all_expt_norm = isempty(all_expt_norm) ? out.panel_normalized : vcat(all_expt_norm, out.panel_normalized; cols=:union)
    end

    outdir = joinpath(out_root, "experimental_all")
    isdir(outdir) || mkpath(outdir)

    CSV.write(joinpath(outdir, "all_dataset_meta.csv"), all_meta)
    CSV.write(joinpath(outdir, "all_sample_manifest.csv"), all_manifest)
    CSV.write(joinpath(outdir, "all_expt_points.csv"), all_expt)
    CSV.write(joinpath(outdir, "all_expt_unnormalized.csv"), all_expt_unnorm)
    CSV.write(joinpath(outdir, "all_expt_normalized.csv"), all_expt_norm)

    println("[TNW] Wrote:")
    println("  " * joinpath(outdir, "all_dataset_meta.csv"))
    println("  " * joinpath(outdir, "all_sample_manifest.csv"))
    println("  " * joinpath(outdir, "all_expt_points.csv"))
    println("  " * joinpath(outdir, "all_expt_unnormalized.csv"))
    println("  " * joinpath(outdir, "all_expt_normalized.csv"))
end

main(ARGS)
