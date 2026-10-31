# c2d2_run.jl - run the C2D2 inputs of one sweep folder.
#
# Written into every run folder by the C2D2 CaBER toolkit page. Needs only Julia
# (>= 1.9) and a copy of C2D2.jl; it uses nothing beyond the Julia standard library.
#
#   julia c2d2_run.jl <path to C2D2.jl> [inputs | file.toml ...] [--force]
#
#   <path to C2D2.jl>  the cloned repository (the folder that contains julia/models)
#   inputs             a folder: every *.toml in it is run (steady runs first)
#   file.toml ...      or one or more input files
#   --force            re-run inputs whose outputs already say "ok"
#
# Each input  inputs/<id>.toml  writes to  outputs/<id>/ :
#   summary.csv      the solver's summary (all runs)
#   caber_diag.csv   the CaBER stress balance (CaBER runs only)
#   input.toml       a copy of the input
#   status.json      ok / failed, wall time, number of points, error message
#
# The kind of run (steady extension or CaBER) is read from [flow] kind.

const ARGS_ = copy(ARGS)
force = "--force" in ARGS_
filter!(a -> a != "--force", ARGS_)
length(ARGS_) >= 1 || error("usage: julia c2d2_run.jl <path to C2D2.jl> [inputs | file.toml ...] [--force]")

root = abspath(expanduser(ARGS_[1]))
models = isdir(joinpath(root, "julia", "models")) ? joinpath(root, "julia", "models") :
         isdir(joinpath(root, "models")) ? joinpath(root, "models") :
         error("Cannot find julia/models under $root - is this the C2D2.jl folder?")

const RUNDIR = @__DIR__

for f in ("kernel/Fene.jl", "kernel/Drag.jl", "kernel/Spectrum.jl", "kernel/CoeffsRelax.jl",
          "kernel/evals/Stress.jl", "kernel/evals/StateObs.jl",
          "kernel/flows/StyextFlows.jl", "kernel/flows/UsyextFlows.jl",
          "kernel/modelpacks/StyextModelPack.jl",
          "numerics/StyextNewton.jl", "numerics/StyextManifold.jl", "numerics/UsyextIntegrator.jl",
          "front_end/StyextRunContext.jl", "front_end/UsyextRunContext.jl")
    include(joinpath(models, f))
end

import TOML
using Printf
using Dates

# ---------------------------------------------------------------- which files
targets = ARGS_[2:end]
isempty(targets) && (targets = [joinpath(RUNDIR, "inputs")])
files = String[]
for t in targets
    p = isabspath(t) ? t : (ispath(t) ? abspath(t) : joinpath(RUNDIR, t))
    if isdir(p)
        append!(files, sort(filter(x -> endswith(x, ".toml"), readdir(p; join=true))))
    elseif isfile(p)
        push!(files, p)
    else
        @warn "Not found, skipped: $t"
    end
end
kindof(f) = lowercase(String(get(get(TOML.parsefile(f), "flow", Dict()), "kind", "caber")))
sort!(files; by = f -> (kindof(f) == "styext" ? 0 : 1, basename(f)))   # steady runs are cheap: do them first
isempty(files) && error("No input files to run.")

jstr(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => " ") * "\""
function write_status(dir, d)
    open(joinpath(dir, "status.json"), "w") do io
        println(io, "{", join(["$(jstr(k)): $(v isa AbstractString ? jstr(v) : v)" for (k, v) in d], ", "), "}")
    end
end

println(@sprintf("C2D2 run: %d input(s), C2D2.jl at %s", length(files), root))
nfail = 0
for (k, f) in enumerate(files)
    id = splitext(basename(f))[1]
    outdir = joinpath(RUNDIR, "outputs", id)
    st = joinpath(outdir, "status.json")
    if !force && isfile(st) && occursin("\"ok\"", read(st, String))
        println(@sprintf("[%d/%d] %-28s already done (use --force to redo)", k, length(files), id))
        continue
    end
    mkpath(outdir)
    cp(f, joinpath(outdir, "input.toml"); force=true)
    kind = kindof(f)
    write_status(outdir, ["status" => "running", "kind" => kind, "started" => string(now())])
    t0 = time()
    try
        if kind == "styext"
            params, mp_ctx, solver_opts, man_opts = StyextRunContext.load_styext_context(f)
            traj = StyextManifold.build_manifold(mp_ctx, solver_opts, man_opts)
            StyextManifold.write_summary_csv(joinpath(outdir, "summary.csv"), traj)
        else
            ctx, flow, plan, tol = UsyextRunContext.load_usyext_context(f)
            M0 = UsyextRunContext.build_initial_M(f, ctx)
            traj = UsyextIntegrator.integrate_usyext(ctx, flow, plan, tol; M0 = M0)
            UsyextIntegrator.write_summary_csv(joinpath(outdir, "summary.csv"), traj)
            UsyextIntegrator.write_caber_diag_csv(joinpath(outdir, "caber_diag.csv"), traj)
        end
        dt = time() - t0
        write_status(outdir, ["status" => "ok", "kind" => kind, "wall_s" => round(dt; digits=2),
                              "n_points" => length(traj), "finished" => string(now())])
        println(@sprintf("[%d/%d] %-28s ok      %5d pts  %8.1f s", k, length(files), id, length(traj), dt))
    catch e
        global nfail += 1
        dt = time() - t0
        msg = sprint(showerror, e)
        write_status(outdir, ["status" => "failed", "kind" => kind, "wall_s" => round(dt; digits=2),
                              "error" => first(msg, 400), "finished" => string(now())])
        println(@sprintf("[%d/%d] %-28s FAILED  after %.1f s: %s", k, length(files), id, dt, first(msg, 200)))
    end
end
println(nfail == 0 ? "All done." : "Done, with $nfail failed run(s): see outputs/<id>/status.json")
