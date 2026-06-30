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

function main(args)
    dataset_name = length(args) >= 1 ? lowercase(strip(String(args[1]))) :
        error("Usage: harvest_experimental_dataset.jl <dataset_name> [xlsx_path] [plan_sheet_name] [out_root]")
    xlsx_path = length(args) >= 2 ? String(args[2]) : default_plan_xlsx()
    plan_sheet_name = length(args) >= 3 ? String(args[3]) : "Plan_Exptl"
    out_root = length(args) >= 4 ? String(args[4]) : default_harvested_root()

    dataset_sheet = pretty_dataset_sheet_name(dataset_name)
    sel = read_plan_dataset_selection(xlsx_path; plan_sheet_name=plan_sheet_name, dataset_name=dataset_name)
    processed_dir = processed_dir_from_plan(sel)
    outdir = joinpath(out_root, "experimental_" * dataset_name)
    isdir(outdir) || mkpath(outdir)

    println("[TNW] Harvesting experimental dataset")
    println("  dataset       = $(dataset_name)")
    println("  xlsx          = $(xlsx_path)")
    println("  hK_star       = $(sel.hK_star)")
    println("  trial tag     = $(sel.trial_tag)")
    println("  processed_dir = $(processed_dir)")
    println("  outdir        = $(outdir)")

    renorm = read_csv_checked(joinpath(processed_dir, "renorm_lookup.csv"))
    model_all = read_csv_checked(joinpath(processed_dir, "model_pipkin_curves.csv"))
    expt = read_csv_checked(joinpath(processed_dir, "expt_pipkin_points.csv"))

    add_dataset_column!(renorm, dataset_name)
    add_dataset_column!(model_all, dataset_name)
    add_dataset_column!(expt, dataset_name)

    model_c2d2 = :drag_model in names(model_all) ? model_all[String.(model_all.drag_model) .== "c2d2", :] : copy(model_all)

    sample_manifest = build_sample_manifest(dataset_name, expt, model_all, model_c2d2, sel.table)
    dataset_meta = build_dataset_meta(dataset_name, dataset_sheet, sel, processed_dir, renorm, model_all, model_c2d2, expt)
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

    println("[TNW] Wrote:")
    println("  " * joinpath(outdir, "dataset_meta.csv"))
    println("  " * joinpath(outdir, "sample_manifest.csv"))
    println("  " * joinpath(outdir, "renorm_lookup.csv"))
    println("  " * joinpath(outdir, "model_curves_all.csv"))
    println("  " * joinpath(outdir, "model_curves_c2d2.csv"))
    println("  " * joinpath(outdir, "expt_points.csv"))
    println("  " * joinpath(outdir, "panel_unnormalized.csv"))
    println("  " * joinpath(outdir, "panel_normalized.csv"))
end

main(ARGS)
