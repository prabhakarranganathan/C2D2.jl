# Minimal USYEXT (uniaxial step-strain / CaBER-type) example for C2D2.
#
#   julia julia/examples/run_usyext_example.jl
#
# Loads a reference TOML, integrates the transient uniaxial-extension /
# capillary-thinning ODE with the adaptive ETD2 integrator, and writes a
# summary CSV plus a CaBER stress-decomposition diagnostic CSV.
# Depends only on the Julia standard library (LinearAlgebra, Printf, TOML).

const _MODELS = joinpath(@__DIR__, "..", "models")

include(joinpath(_MODELS, "kernel", "Fene.jl"))
include(joinpath(_MODELS, "kernel", "Drag.jl"))
include(joinpath(_MODELS, "kernel", "Spectrum.jl"))
include(joinpath(_MODELS, "kernel", "CoeffsRelax.jl"))
include(joinpath(_MODELS, "kernel", "evals", "Stress.jl"))
include(joinpath(_MODELS, "kernel", "evals", "StateObs.jl"))
include(joinpath(_MODELS, "kernel", "flows", "StyextFlows.jl"))
include(joinpath(_MODELS, "kernel", "flows", "UsyextFlows.jl"))
include(joinpath(_MODELS, "kernel", "modelpacks", "StyextModelPack.jl"))
include(joinpath(_MODELS, "numerics", "StyextNewton.jl"))
include(joinpath(_MODELS, "numerics", "StyextManifold.jl"))
include(joinpath(_MODELS, "numerics", "UsyextIntegrator.jl"))
include(joinpath(_MODELS, "front_end", "StyextRunContext.jl"))
include(joinpath(_MODELS, "front_end", "UsyextRunContext.jl"))

using .UsyextRunContext: load_usyext_context, build_initial_M
using .UsyextIntegrator: integrate_usyext, write_summary_csv, write_caber_diag_csv

# Reference input: FENE-PM, constant drag, Nm=1, phi=0.03, De=1.0.
toml_path = joinpath(@__DIR__, "..", "experiments", "c2d2ref", "inputs", "usyext_ref_01.toml")

# load_usyext_context returns (runtime_ctx, flow, integration_plan, tol_opts).
ctx, flow, plan, tol = load_usyext_context(toml_path; De = 1.0)

# Optional pre-stretched initial conformation (nothing -> equilibrium IC).
M0 = build_initial_M(toml_path, ctx)

traj = integrate_usyext(ctx, flow, plan, tol; M0 = M0)

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
write_summary_csv(joinpath(outdir, "usyext_summary.csv"), traj)
write_caber_diag_csv(joinpath(outdir, "usyext_caber_diag.csv"), traj)
cp(toml_path, joinpath(outdir, "input.toml"); force = true)

println("USYEXT example complete: $(length(traj)) samples -> ",
        joinpath(outdir, "usyext_summary.csv"))
