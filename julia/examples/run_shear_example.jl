# Steady simple shear with C2D2.jl: the smallest complete run.
#
#   julia julia/examples/run_shear_example.jl
#
# Reads julia/examples/inputs/shear_example.toml (every key is commented there
# and in docs/input_reference.md), solves the steady state at each shear
# Weissenberg number of the [sweep] grid, and writes
# julia/examples/output/shear_summary.csv with the polymer shear viscosity and
# the first and second normal-stress coefficients (columns in
# docs/output_reference.md).  The solver uses only the Julia standard library.
#
# Simple shear is the planar mixed flow with r = 1; the same script handles
# any r (0 = planar extension) and the continuation solver can be run on the
# same input with julia/experiments/pmf/run_pmf.jl --pac.

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
include(joinpath(_MODELS, "numerics", "SteadyWiSweep.jl"))
include(joinpath(_MODELS, "front_end", "StyextRunContext.jl"))

using .StyextRunContext: load_styext_context
using .SteadyWiSweep: sweep_fixed_Wi, write_pmf_summary_csv

toml_path = joinpath(@__DIR__, "inputs", "shear_example.toml")

# load_styext_context returns (params, runtime_ctx, solver_opts, manifold_opts);
# params.Wi_sweep holds the [sweep] grid (in the report's Wi, half the shear Wi).
params, mp_ctx, solver_opts, man_opts = load_styext_context(toml_path)

# Solve the steady state at each Wi of the grid, warm-starting from the previous one.
traj = sweep_fixed_Wi(mp_ctx, params.Wi_sweep, solver_opts; X_stop = man_opts.X_stop)

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
write_pmf_summary_csv(joinpath(outdir, "shear_summary.csv"), traj)
cp(toml_path, joinpath(outdir, "input.toml"); force = true)

n_ok = count(p -> p.status_str in ("eq", "FWi-cnv"), traj)
println("Shear example complete: $n_ok of $(length(traj)) points converged -> ",
        joinpath(outdir, "shear_summary.csv"))
