# Capillary thinning (CaBER-type) with C2D2.jl: the smallest complete run.
#
#   julia julia/examples/run_usyext_example.jl
#
# Reads julia/examples/inputs/caber_example.toml (every key is commented there
# and in docs/input_reference.md), integrates the chain conformation through
# the thinning of a filament with the adaptive exponential time-differencing
# integrator, and writes two files to julia/examples/output/ (columns in
# docs/output_reference.md):
#   usyext_summary.csv     strain, time, Wi, stretch, stresses, extensional viscosity
#   usyext_caber_diag.csv  the terms of the capillary stress balance at each sample
# The solver uses only the Julia standard library.
#
# The same script runs start-up of extension at constant Wi if the TOML has
# [flow] kind = "constant".

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

toml_path = joinpath(@__DIR__, "inputs", "caber_example.toml")

# load_usyext_context returns (runtime_ctx, flow, integration_plan, tol_opts).
# The keyword `De` overrides [flow] De of the file: handy for De sweeps.
ctx, flow, plan, tol = load_usyext_context(toml_path; De = 1.0)

# Initial conformation: `nothing` means chains at rest; the TOML's
# [initial_condition] section can ask for a pre-stretched start instead.
M0 = build_initial_M(toml_path, ctx)

traj = integrate_usyext(ctx, flow, plan, tol; M0 = M0)

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
write_summary_csv(joinpath(outdir, "usyext_summary.csv"), traj)
write_caber_diag_csv(joinpath(outdir, "usyext_caber_diag.csv"), traj)
cp(toml_path, joinpath(outdir, "input.toml"); force = true)

println("USYEXT example complete: $(length(traj)) samples -> ",
        joinpath(outdir, "usyext_summary.csv"))
