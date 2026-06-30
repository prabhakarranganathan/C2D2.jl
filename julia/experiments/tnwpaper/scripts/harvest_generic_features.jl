#!/usr/bin/env julia

using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "TNW.jl"))
using .TNW

function main(args)
    root = length(args) >= 1 ? args[1] : default_generic_outputs_dir()
    outdir = length(args) >= 2 ? args[2] : default_harvested_dir()

    println("[TNW] Harvesting generic outputs")
    println("  root   = $(root)")
    println("  outdir = $(outdir)")

    out = harvest_all_generic_features(root=root, outdir=outdir)

    println("[TNW] Wrote:")
    println("  $(joinpath(outdir, "manifest.csv"))")
    println("  $(joinpath(outdir, "curves.csv"))")
    println("  $(joinpath(outdir, "features_selected.csv"))")
    println("  $(joinpath(outdir, "sweep_features.csv"))")

    println("[TNW] Counts:")
    println("  manifest rows          = $(nrow(out.manifest))")
    println("  curve rows             = $(nrow(out.curves))")
    println("  selected feature rows  = $(nrow(out.features_selected))")
    println("  sweep feature rows     = $(nrow(out.sweep_features))")
end

main(ARGS)
