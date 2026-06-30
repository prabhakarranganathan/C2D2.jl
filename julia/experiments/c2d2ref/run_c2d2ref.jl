"""
    run_c2d2ref.jl

Julia equivalent of MATLAB's run_all_refs.m.

Runs all 5 C2D2 reference cases and writes output CSVs to
  julia/experiments/c2d2ref/outputs/<case>/<run_id>/

Reference cases (same as MATLAB):
  styext_ref_01 : FENE-PM, constant drag, Nm=1, phi=1.0,  Wi_max=3.0
  styext_ref_02 : FENE-PM, c2d2 drag,    Nm=3, phi=0.1,  Wi_max=5.0
  usyext_ref_01 : FENE-PM, constant drag, Nm=1, phi=0.03, De=1.0   (CaBER)
  usyext_ref_02 : FENE-PM, c2d2 drag,    Nm=3, phi=0.1,  De=0.1   (CaBER)
  usyext_ref_03 : FENE-PM, constant drag, Nm=1, phi=0.09, De=0.02  (CaBER)

Usage (from repo root):
  julia --project=julia julia/experiments/c2d2ref/run_c2d2ref.jl

Outputs are written to:
  julia/experiments/c2d2ref/outputs/

The run_id subfolder is timestamped so repeated runs do not clobber each other.
"""

using Dates
using Printf

# ── Bootstrap ─────────────────────────────────────────────────────────────────

const _REPO = joinpath(@__DIR__, "..", "..", "..")   # repo root
const _MODELS = joinpath(_REPO, "julia", "models")
const _INPUTS = joinpath(@__DIR__, "inputs")
const _OUT_ROOT = joinpath(@__DIR__, "outputs")

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
include(joinpath(_MODELS, "kernel", "Params.jl"))
include(joinpath(_MODELS, "kernel", "flows", "UsyextFlows.jl"))
include(joinpath(_MODELS, "numerics", "UsyextIntegrator.jl"))
include(joinpath(_MODELS, "front_end", "UsyextRunContext.jl"))

using .Fene
using .Drag
using .Spectrum
using .CoeffsRelax
using .Stress
using .StateObs
using .StyextFlows
using .StyextModelPack
using .StyextNewton
using .StyextManifold
using .StyextRunContext
using .UsyextFlows
using .UsyextIntegrator
using .UsyextRunContext

# ── Helpers ───────────────────────────────────────────────────────────────────

function _run_dir(label, subdir, fe, drag, Nm, tag="")
    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    nm_tag = "$(lowercase(fe))_$(lowercase(drag))_Nm=$(Nm)$(isempty(tag) ? "" : "_$(tag)")_$(ts)"
    d = joinpath(_OUT_ROOT, label, subdir, nm_tag)
    mkpath(d)
    return d
end

function _fmt_sec(elapsed)
    elapsed < 60 ? @sprintf("%.1fs", elapsed) : @sprintf("%.1fmin", elapsed/60)
end

# ── STYEXT runner ─────────────────────────────────────────────────────────────

"""
    run_styext_case(label, toml_file, phi_tag; verbose=false)

Run one STYEXT PAC case and write:
  <out_dir>/summary.csv      — manifold trajectory
  <out_dir>/input.toml       — copy of the TOML

Returns the output directory path.
"""
function run_styext_case(label, toml_file, phi_tag; verbose=false)
    params, mp_ctx, solver_opts, man_opts = load_styext_context(toml_file)

    man_opts_run = ManifoldOpts(
        ds0            = man_opts.ds0,
        ds_min         = man_opts.ds_min,
        ds_max         = man_opts.ds_max,
        max_steps      = man_opts.max_steps,
        Wi_max         = man_opts.Wi_max,
        tol_FY_rel_hard = man_opts.tol_FY_rel_hard,
        target_newton  = man_opts.target_newton,
        target_rho     = man_opts.target_rho,
        verbose        = verbose,
    )

    t_start = time()
    traj    = build_manifold(mp_ctx, solver_opts, man_opts_run)
    elapsed = time() - t_start

    # Write outputs
    out_dir = _run_dir(label, phi_tag,
                       string(params.fe_model),
                       string(params.drag_model),
                       params.Nm)
    StyextManifold.write_summary_csv(joinpath(out_dir, "summary.csv"), traj)
    cp(toml_file, joinpath(out_dir, "input.toml"); force=true)

    return out_dir, length(traj), elapsed
end

# ── USYEXT runner ─────────────────────────────────────────────────────────────

"""
    run_usyext_case(label, toml_file, de_tag; verbose=false)

Run one USYEXT CaBER case and write:
  <out_dir>/summary.csv      — trajectory observables
  <out_dir>/caber_diag.csv   — CaBER capillary diagnostics
  <out_dir>/input.toml       — copy of the TOML

Returns the output directory path.
"""
function run_usyext_case(label, toml_file, de_tag; verbose=false)
    ctx, flow, plan, tol = load_usyext_context(toml_file)

    t_start = time()
    traj    = integrate_usyext(ctx, flow, plan, tol)
    elapsed = time() - t_start

    # Write outputs
    De_val = isa(flow, CaberFlow) ? flow.De :
             isa(flow, ConstantWiFlow) ? flow.De : NaN
    out_dir = _run_dir(label, de_tag,
                       string(ctx.fe_model),
                       string(ctx.kernel.drag_lut.drag_model),
                       ctx.Nm,
                       @sprintf("De=%.4g", De_val))
    UsyextIntegrator.write_summary_csv(joinpath(out_dir, "summary.csv"), traj)
    UsyextIntegrator.write_caber_diag_csv(joinpath(out_dir, "caber_diag.csv"), traj)
    cp(toml_file, joinpath(out_dir, "input.toml"); force=true)

    return out_dir, length(traj), elapsed
end

# ── Case definitions ──────────────────────────────────────────────────────────

const CASES = [
    # (label,             toml,                     kind,     sweep_tag)
    ("c2d2ref_styext_01", "styext_ref_01.toml", :styext, "phi_1"),
    ("c2d2ref_styext_02", "styext_ref_02.toml", :styext, "phi_0.1"),
    ("c2d2ref_usyext_01", "usyext_ref_01.toml", :usyext, "De_1"),
    ("c2d2ref_usyext_02", "usyext_ref_02.toml", :usyext, "De_0.1"),
    ("c2d2ref_usyext_03", "usyext_ref_03.toml", :usyext, "De_0.02"),
]

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    println("\n=== C2D2 Julia-port reference run ===")
    println("Inputs : $_INPUTS")
    println("Outputs: $_OUT_ROOT\n")

    mkpath(_OUT_ROOT)

    ok_count = 0
    for (i, (label, toml_name, kind, sweep_tag)) in enumerate(CASES)
        toml_file = joinpath(_INPUTS, toml_name)
        print(@sprintf("[%d/5] %-35s ... ", i, label))

        try
            if kind === :styext
                out_dir, n_pts, elapsed =
                    run_styext_case(label, toml_file, sweep_tag)
                @printf("done  (%d pts, %s)\n", n_pts, _fmt_sec(elapsed))
                println("       -> $out_dir")
            else
                out_dir, n_pts, elapsed =
                    run_usyext_case(label, toml_file, sweep_tag)
                @printf("done  (%d pts, %s)\n", n_pts, _fmt_sec(elapsed))
                println("       -> $out_dir")
            end
            ok_count += 1
        catch e
            println("FAILED")
            @warn "Error in $label" exception=(e, catch_backtrace())
        end
    end

    println("\n=== Done: $ok_count/5 cases succeeded ===")
    if ok_count == 5
        println("All outputs in: $_OUT_ROOT")
    end
    return ok_count
end

main()
