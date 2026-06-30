"""
    run_exptl_comparison.jl — top-level driver

Full pipeline for one dataset + hK_star trial:
  1. Export plan TOML from parameter-sets.xlsx (Piece 1)
  2. Create sweep folder tree + patched input.toml files (Piece 2a)
  3. Run each sweep (styext or caber) and extract Wi features (Pieces 2b + 3)

Usage (from repo root):
    julia --project=julia julia/experiments/tnwpaper/run/run_exptl_comparison.jl \\
        <dataset_name> [hK_star]

    dataset_name  e.g. "gaillard", "calabrese", "anna", "clasen", "gaillard_aq"
    hK_star       optional; if omitted, uses the value from the xlsx

    Plan TOML is written to:
        julia/experiments/tnwpaper/run/plans/<dataset>_hsK_<tok>.toml

    Outputs go to:
        julia/experiments/tnwpaper/outputs/exptl_comparison/<dataset>/hsK_<tok>/
"""

import TOML
using Printf

# ── bootstrap includes ────────────────────────────────────────────────────────
# export_plan_from_xlsx.jl uses XLSX.jl (must be in julia/ Project)
# make_prod_sweeps.jl uses TOML (stdlib)
# run_sweep.jl includes all physics models

include(joinpath(@__DIR__, "export_plan_from_xlsx.jl"))
include(joinpath(@__DIR__, "make_prod_sweeps.jl"))
include(joinpath(@__DIR__, "run_sweep.jl"))   # heavy: loads physics modules

# ── helpers ───────────────────────────────────────────────────────────────────

function _get_phi_values_from_toml(input_toml::AbstractString)
    d    = TOML.parsefile(input_toml)
    conc = get(d, "concentration", Dict())
    pv   = get(conc, "phi_values", nothing)
    pv === nothing && (pv = get(conc, "phi", [1.0]))
    return Float64.(isa(pv, AbstractVector) ? pv : [pv])
end

function _get_De_values_from_toml(input_toml::AbstractString)
    d    = TOML.parsefile(input_toml)
    flow = get(d, "flow", Dict())
    dv   = get(flow, "De_values", nothing)
    dv === nothing && (dv = get(flow, "De", [1.0]))
    return Float64.(isa(dv, AbstractVector) ? dv : [dv])
end

function _x_mode_from_kind(kind::AbstractString)
    startswith(lowercase(kind), "caber") ? "E2" : "E2"
end

# ── main driver ───────────────────────────────────────────────────────────────

"""
    run_exptl_comparison(dataset_name; hK_star=nothing, verbose=false)

Run the full pipeline for one dataset.
"""
function run_exptl_comparison(dataset_name::AbstractString;
                               hK_star::Union{Float64,Nothing} = nothing,
                               verbose::Bool                   = false)

    ds = lowercase(strip(string(dataset_name)))

    # 1. Export plan TOML
    plans_dir = joinpath(@__DIR__, "plans")
    mkpath(plans_dir)

    # we need a temporary plan to get the hK_star token — do a probe export
    probe_toml = tempname() * ".toml"
    export_plan_from_xlsx(ds, probe_toml; hK_star=hK_star)

    plan_data  = TOML.parsefile(probe_toml)
    prod       = get(plan_data, "production", Dict())
    out_root   = get(prod, "out_root", "")
    # extract the trial tag from the out_root (last path component = "hsK_...")
    trial_tag  = basename(out_root)

    plan_toml = joinpath(plans_dir, "$(ds)_$(trial_tag).toml")
    cp(probe_toml, plan_toml; force=true)
    rm(probe_toml; force=true)

    println("\n=== run_exptl_comparison: $ds / $trial_tag ===\n")
    println("Plan TOML: $plan_toml")
    println("Out root : $out_root\n")

    # 2. Create sweep folders
    println("--- Piece 2a: make_prod_sweeps ---")
    sweep_specs = make_prod_sweeps(plan_toml)
    println()

    # 3. Run each sweep
    println("--- Piece 2b+3: run sweeps + extract Wi features ---")
    n_sweeps = length(sweep_specs)
    for (i, spec) in enumerate(sweep_specs)
        sweep_dir   = spec.sweep_dir
        input_toml  = spec.input_toml
        kind        = spec.kind
        folder      = spec.folder

        @printf("\n[%2d/%d] %s  (%s)\n", i, n_sweeps, folder, kind)

        try
            if startswith(kind, "styext")
                phi_vals = _get_phi_values_from_toml(input_toml)
                run_styext_sweep(sweep_dir, input_toml;
                                 phi_values=phi_vals,
                                 x_mode=_x_mode_from_kind(kind),
                                 verbose=verbose)
            elseif startswith(kind, "caber")
                de_vals = _get_De_values_from_toml(input_toml)
                run_usyext_sweep(sweep_dir, input_toml;
                                 De_values=de_vals,
                                 x_mode=_x_mode_from_kind(kind),
                                 verbose=verbose)
            else
                @warn "Unknown sweep kind '$kind' for $folder — skipping"
            end
        catch e
            @error "Sweep $folder FAILED" exception=(e, catch_backtrace())
        end
    end

    println("\n=== Done: $ds / $trial_tag ===")
    println("Outputs: $(joinpath(_REPO_RS, out_root))")
    return joinpath(_REPO_RS, out_root)
end

# ── CLI entry ──────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error(
        "Usage: run_exptl_comparison.jl <dataset_name> [hK_star]\n" *
        "  e.g. run_exptl_comparison.jl gaillard 0.005"
    )
    ds  = ARGS[1]
    hk  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing
    run_exptl_comparison(ds; hK_star=hk)
end
