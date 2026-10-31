"""
    run_pmf.jl

Driver for steady planar-mixed-flow (PMF) runs — simple shear at r = 1 — and,
since it shares the model pack, for any steady STYEXT/PMF TOML.

For each TOML it runs
  * a fixed-Wi sweep over the `[sweep]` grid (if present)   → summary_sweep.csv
  * optionally (`--pac`) the PAC manifold builder            → summary_pac.csv
and copies the input TOML next to the outputs.

Usage (from repo root):
  julia --project=julia julia/experiments/pmf/run_pmf.jl [TOML ...] [--pac] [--no-sweep] [--verbose]

With no TOML arguments every `pmf_*.toml` in julia/experiments/pmf/inputs/ is run.
Outputs go to julia/experiments/pmf/outputs/<label>/<timestamp>/ (gitignored).

Conventions (see StyextFlows.jl / SteadyWiSweep.jl)
  * `Wi` columns are the report-appendix Wi (λ₀ × principal strain rate);
    `Wi_shear = 2·Wi = λ₀γ̇` is the conventional shear Weissenberg number.
  * TOML `[flow] Wi_convention = "shear"` lets [sweep]/[limits] Wi be given as λ₀γ̇.
  * Stresses follow Stress.jl: τ/(c kBT) = −Σ(3f_pM_p − I); η_p, Ψ₁, Ψ₂ are positive.
"""

using Dates
using Printf

const _REPO   = joinpath(@__DIR__, "..", "..", "..")
const _MODELS = joinpath(_REPO, "julia", "models")
const _INPUTS = joinpath(@__DIR__, "inputs")
const _OUT    = joinpath(@__DIR__, "outputs")

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

using .Fene, .Drag, .Spectrum, .CoeffsRelax, .Stress, .StateObs
using .StyextFlows, .StyextModelPack, .StyextNewton, .StyextManifold
using .SteadyWiSweep, .StyextRunContext

function _parse_args(args)
    tomls = String[]; pac = false; sweep = true; verbose = false
    for a in args
        if a == "--pac";          pac = true
        elseif a == "--no-sweep"; sweep = false
        elseif a == "--verbose";  verbose = true
        elseif startswith(a, "--"); error("unknown flag $a")
        else push!(tomls, a)
        end
    end
    if isempty(tomls)
        tomls = sort(filter(f -> startswith(basename(f), "pmf_") && endswith(f, ".toml"),
                            readdir(_INPUTS; join = true)))
    end
    return tomls, pac, sweep, verbose
end

function run_case(toml::AbstractString; pac::Bool, sweep::Bool, verbose::Bool)
    params, ctx, solver_opts, man_opts = load_styext_context(toml)
    ts  = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    out = joinpath(_OUT, params.run_label, ts)
    mkpath(out)
    cp(toml, joinpath(out, "input.toml"); force = true)

    kind = params.flow_kind === :pmf ? "pmf r=$(params.r)" : "styext α=$(params.alphas)"
    @printf("%-40s  %-22s  %s Nm=%d φ=%g\n", params.run_label, kind,
            params.fe_model, params.Nm, params.phi)

    if sweep && !isempty(params.Wi_sweep)
        t0 = time()
        traj = sweep_fixed_Wi(ctx, params.Wi_sweep, solver_opts; verbose = verbose, X_stop = man_opts.X_stop)
        n_ok = count(p -> p.status_str in ("eq", "FWi-cnv"), traj)
        write_pmf_summary_csv(joinpath(out, "summary_sweep.csv"), traj)
        @printf("    sweep : %d/%d pts converged  Wi ∈ [%.3g, %.3g]  (%.1fs)\n",
                n_ok, length(traj), traj[2].obs.Wi, traj[end].obs.Wi, time() - t0)
        if params.flow_kind === :pmf
            o = traj[end].obs
            @printf("            last: Wi_shear=%.4g  eta_p=%.6g  Psi1=%.6g  Psi2=%.2e  Y=%.6g\n",
                    wi_shear(o.Wi), o.eta_p, o.Psi1, o.Psi2, o.E2)
        end
    elseif sweep
        println("    sweep : no [sweep] section in TOML — skipped")
    end

    if pac
        t0 = time()
        mo = ManifoldOpts(ds0 = man_opts.ds0, ds_min = man_opts.ds_min, ds_max = man_opts.ds_max,
                          max_steps = man_opts.max_steps, Wi_max = man_opts.Wi_max,
                          tol_FY_rel_hard = man_opts.tol_FY_rel_hard,
                          target_newton = man_opts.target_newton, target_rho = man_opts.target_rho,
                          verbose = verbose, X_stop = man_opts.X_stop)
        traj = build_manifold(ctx, solver_opts, mo)
        if params.flow_kind === :pmf
            write_pmf_summary_csv(joinpath(out, "summary_pac.csv"), traj)
        else
            StyextManifold.write_summary_csv(joinpath(out, "summary_pac.csv"), traj)
        end
        @printf("    PAC   : %d pts  Wi_end=%.4g  Y_end=%.6g  (%.1fs)\n",
                length(traj), traj[end].obs.Wi, traj[end].Y, time() - t0)
    end
    println("    -> $out")
    return out
end

function main()
    tomls, pac, sweep, verbose = _parse_args(ARGS)
    isempty(tomls) && error("no TOML inputs found")
    println("=== PMF / steady-flow driver  ($(length(tomls)) case(s)) ===")
    for t in tomls
        try
            run_case(t; pac = pac, sweep = sweep, verbose = verbose)
        catch e
            @warn "case failed: $t" exception = (e, catch_backtrace())
        end
    end
end

main()
