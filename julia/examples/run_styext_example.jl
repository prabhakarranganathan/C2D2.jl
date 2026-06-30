# Minimal STYEXT (steady uniaxial extension) example for C2D2.
#
#   julia julia/examples/run_styext_example.jl
#
# Loads a reference TOML, builds the steady-extension manifold with the
# pseudo-arclength continuation (PAC) solver, and writes a summary CSV.
# The C2D2 STYEXT/USYEXT solver depends only on the Julia standard library
# (LinearAlgebra, Printf, TOML) — no package instantiation is required to
# run this example.

const _MODELS = joinpath(@__DIR__, "..", "models")

include(joinpath(_MODELS, "kernel", "Fene.jl"))
include(joinpath(_MODELS, "kernel", "Drag.jl"))
include(joinpath(_MODELS, "kernel", "Spectrum.jl"))
include(joinpath(_MODELS, "kernel", "CoeffsRelax.jl"))
include(joinpath(_MODELS, "kernel", "evals", "Stress.jl"))
include(joinpath(_MODELS, "kernel", "evals", "StateObs.jl"))
include(joinpath(_MODELS, "kernel", "flows", "StyextFlows.jl"))
include(joinpath(_MODELS, "kernel", "modelpacks", "StyextModelPack.jl"))
include(joinpath(_MODELS, "numerics", "StyextNewton.jl"))
include(joinpath(_MODELS, "numerics", "StyextManifold.jl"))
include(joinpath(_MODELS, "front_end", "StyextRunContext.jl"))

using .StyextRunContext: load_styext_context
using .StyextManifold: build_manifold, write_summary_csv

# Reference input: FENE-PM, constant drag, Nm=3, phi=0.1 (C2D2 drag).
toml_path = joinpath(@__DIR__, "..", "experiments", "c2d2ref", "inputs", "styext_ref_02.toml")

# load_styext_context returns (params, runtime_ctx, solver_opts, manifold_opts).
params, mp_ctx, solver_opts, man_opts = load_styext_context(toml_path; phi = 0.1)

# Build the steady-extension manifold (stress vs. Weissenberg number).
traj = build_manifold(mp_ctx, solver_opts, man_opts)

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
write_summary_csv(joinpath(outdir, "styext_summary.csv"), traj)
cp(toml_path, joinpath(outdir, "input.toml"); force = true)

println("STYEXT example complete: $(length(traj)) manifold points -> ",
        joinpath(outdir, "styext_summary.csv"))
