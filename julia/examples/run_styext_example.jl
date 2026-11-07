# Steady uniaxial extension (STYEXT) with C2D2.jl: the smallest complete run.
#
#   julia julia/examples/run_styext_example.jl
#
# Reads julia/examples/inputs/styext_example.toml (every key is commented
# there and in docs/input_reference.md), traces the curve of steady states
# from the coiled chain to full extension with the pseudo-arclength
# continuation solver, and writes julia/examples/output/styext_summary.csv
# (columns in docs/output_reference.md).  The solver uses only the Julia
# standard library, so no packages need to be installed.
#
# To change a parameter, edit the TOML (for example `phi`, `NK`, `hK_star` or
# `drag_interp`) and run the script again; or pass a keyword override as shown
# below.  The first run takes about a minute because Julia compiles the code.

const _MODELS = joinpath(@__DIR__, "..", "models")

# The solver is a set of plain modules; include them in dependency order.
include(joinpath(_MODELS, "kernel", "Fene.jl"))             # spring laws
include(joinpath(_MODELS, "kernel", "Drag.jl"))             # C2D2 drag
include(joinpath(_MODELS, "kernel", "Spectrum.jl"))         # relaxation spectrum
include(joinpath(_MODELS, "kernel", "CoeffsRelax.jl"))      # modal coefficients
include(joinpath(_MODELS, "kernel", "evals", "Stress.jl"))  # polymer stress
include(joinpath(_MODELS, "kernel", "evals", "StateObs.jl"))# observables
include(joinpath(_MODELS, "kernel", "flows", "StyextFlows.jl"))
include(joinpath(_MODELS, "kernel", "modelpacks", "StyextModelPack.jl"))
include(joinpath(_MODELS, "numerics", "StyextNewton.jl"))
include(joinpath(_MODELS, "numerics", "StyextManifold.jl"))
include(joinpath(_MODELS, "front_end", "StyextRunContext.jl"))

using .StyextRunContext: load_styext_context
using .StyextManifold: build_manifold, write_summary_csv

toml_path = joinpath(@__DIR__, "inputs", "styext_example.toml")

# load_styext_context returns (params, runtime_ctx, solver_opts, manifold_opts).
# The keyword `phi` overrides [concentration] phi of the file: handy for sweeps.
params, mp_ctx, solver_opts, man_opts = load_styext_context(toml_path; phi = 0.1)

# Trace the steady-state curve (polymer stress and extensional viscosity
# against the Weissenberg number Wi).
traj = build_manifold(mp_ctx, solver_opts, man_opts)

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
write_summary_csv(joinpath(outdir, "styext_summary.csv"), traj)
cp(toml_path, joinpath(outdir, "input.toml"); force = true)

println("STYEXT example complete: $(length(traj)) steady states -> ",
        joinpath(outdir, "styext_summary.csv"))
