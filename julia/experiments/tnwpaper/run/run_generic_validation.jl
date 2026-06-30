#!/usr/bin/env julia
"""
    run_generic_validation.jl <plan_toml>

Thin driver to run a hand-built generic-behaviour plan TOML through the existing
(validated) sweep machinery, WITHOUT the xlsx->plan export step (which is
Plan_Exptl-specific). Reuses make_prod_sweeps + run_styext_sweep/run_usyext_sweep.

Place alongside the other run/ modules so @__DIR__ resolves the includes.
Aru, 2026-06-08 — validation against MATLAB generic_features.
"""

import TOML
using Printf

include(joinpath(@__DIR__, "make_prod_sweeps.jl"))
include(joinpath(@__DIR__, "run_sweep.jl"))   # heavy: physics modules

function _get_phi_values(input_toml)
    d = TOML.parsefile(input_toml); c = get(d, "concentration", Dict())
    pv = get(c, "phi_values", nothing); pv === nothing && (pv = get(c, "phi", [1.0]))
    return Float64.(isa(pv, AbstractVector) ? pv : [pv])
end

function _get_De_values(input_toml)
    d = TOML.parsefile(input_toml); f = get(d, "flow", Dict())
    dv = get(f, "De_values", nothing); dv === nothing && (dv = get(f, "De", [1.0]))
    return Float64.(isa(dv, AbstractVector) ? dv : [dv])
end

length(ARGS) >= 1 || error("Usage: run_generic_validation.jl <plan_toml>")
plan = ARGS[1]

println("=== generic validation: $plan ===")
specs = make_prod_sweeps(plan)
println()

for (i, spec) in enumerate(specs)
    @printf("\n[%2d/%d] %s  (%s)\n", i, length(specs), spec.folder, spec.kind)
    try
        if startswith(spec.kind, "styext")
            run_styext_sweep(spec.sweep_dir, spec.input_toml;
                             phi_values=_get_phi_values(spec.input_toml),
                             x_mode="E2", verbose=false)
        elseif startswith(spec.kind, "caber")
            run_usyext_sweep(spec.sweep_dir, spec.input_toml;
                             De_values=_get_De_values(spec.input_toml),
                             x_mode="E2", verbose=false)
        else
            @warn "Unknown kind '$(spec.kind)' for $(spec.folder)"
        end
    catch e
        @error "Sweep FAILED: $(spec.folder)" exception=(e, catch_backtrace())
    end
end

println("\n=== generic validation DONE ===")
