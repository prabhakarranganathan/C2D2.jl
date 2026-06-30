"""
    run_sweep.jl  (Piece 2b — per-sweep runner)

Bootstraps all Julia physics models and exposes:
    run_styext_sweep(sweep_dir, input_toml; phi_values, verbose)
    run_usyext_sweep(sweep_dir, input_toml; De_values, verbose)

Each runs every point in the phi/De array, writes per-run outputs, runs
Wi-feature detection, and returns sweep-level aggregate DataFrames.

Includes wi_features.jl from the same directory.

Do NOT call this file directly; use run_exptl_comparison.jl.
"""

using Dates
using Printf
using CSV
using DataFrames
import TOML

# ── Bootstrap model includes ──────────────────────────────────────────────────
# Same as run_c2d2ref.jl — include all physics modules.

const _REPO_RS = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const _MODELS_RS = joinpath(_REPO_RS, "julia", "models")

include(joinpath(_MODELS_RS, "kernel", "Fene.jl"))
include(joinpath(_MODELS_RS, "kernel", "Drag.jl"))
include(joinpath(_MODELS_RS, "kernel", "Spectrum.jl"))
include(joinpath(_MODELS_RS, "kernel", "CoeffsRelax.jl"))
include(joinpath(_MODELS_RS, "kernel", "evals", "Stress.jl"))
include(joinpath(_MODELS_RS, "kernel", "evals", "StateObs.jl"))
include(joinpath(_MODELS_RS, "kernel", "flows", "StyextFlows.jl"))
include(joinpath(_MODELS_RS, "kernel", "modelpacks", "StyextModelPack.jl"))
include(joinpath(_MODELS_RS, "numerics", "StyextNewton.jl"))
include(joinpath(_MODELS_RS, "numerics", "StyextManifold.jl"))
include(joinpath(_MODELS_RS, "front_end", "StyextRunContext.jl"))
include(joinpath(_MODELS_RS, "kernel", "Params.jl"))
include(joinpath(_MODELS_RS, "kernel", "flows", "UsyextFlows.jl"))
include(joinpath(_MODELS_RS, "numerics", "UsyextIntegrator.jl"))
include(joinpath(_MODELS_RS, "front_end", "UsyextRunContext.jl"))

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
using .UsyextFlows
using .UsyextIntegrator

include(joinpath(@__DIR__, "wi_features.jl"))

# ── helpers ───────────────────────────────────────────────────────────────────

function _fmt_sec(elapsed)
    elapsed < 60 ? @sprintf("%.1fs", elapsed) : @sprintf("%.1fmin", elapsed / 60)
end

function _run_id(fe, drag, Nm, tag="")
    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    s  = "$(lowercase(string(fe)))_$(lowercase(string(drag)))_Nm=$(Nm)"
    isempty(tag) ? "$(s)_$(ts)" : "$(s)_$(tag)_$(ts)"
end

# ── STYEXT sweep ──────────────────────────────────────────────────────────────

"""
    run_styext_sweep(sweep_dir, input_toml; phi_values, x_mode, verbose)

Run the styext manifold for each phi in phi_values.
sweep_dir/phi_<v>/<run_id>/ holds summary.csv, wi_feature_candidates.csv,
wi_features_selected.csv, input.toml.

Returns wi_features_vs_phi DataFrame.
"""
function run_styext_sweep(sweep_dir::AbstractString,
                          input_toml::AbstractString;
                          phi_values::Vector{Float64} = Float64[],
                          x_mode::String              = "E2",
                          verbose::Bool               = false)

    isfile(input_toml) || error("input.toml not found: $input_toml")

    # read phi_values from TOML if not supplied
    if isempty(phi_values)
        d = TOML.parsefile(input_toml)
        conc = get(d, "concentration", Dict())
        pv   = get(conc, "phi_values", nothing)
        if pv === nothing
            pv = get(conc, "phi", [1.0])
        end
        phi_values = Float64.(isa(pv, AbstractVector) ? pv : [pv])
    end

    isempty(phi_values) && error("No phi_values to sweep in $input_toml")

    feature_rows = Tuple[]
    sweep_id     = ""

    for phi in phi_values
        phi_dir = joinpath(sweep_dir, @sprintf("phi_%g", phi))
        mkpath(phi_dir)

        # Load context with phi override
        params, mp_ctx, solver_opts, man_opts =
            StyextRunContext.load_styext_context(input_toml; phi=phi)

        run_id  = _run_id(params.fe_model, params.drag_model, params.Nm)
        outdir  = joinpath(phi_dir, run_id)
        mkpath(outdir)

        isempty(sweep_id) && (sweep_id = run_id)

        man_opts_run = StyextManifold.ManifoldOpts(
            ds0             = man_opts.ds0,
            ds_min          = man_opts.ds_min,
            ds_max          = man_opts.ds_max,
            max_steps       = man_opts.max_steps,
            Wi_max          = man_opts.Wi_max,
            tol_FY_rel_hard = man_opts.tol_FY_rel_hard,
            target_newton   = man_opts.target_newton,
            target_rho      = man_opts.target_rho,
            verbose         = verbose,
        )
        t0   = time()
        traj = StyextManifold.build_manifold(mp_ctx, solver_opts, man_opts_run)
        elapsed = time() - t0

        StyextManifold.write_summary_csv(joinpath(outdir, "summary.csv"), traj)
        cp(input_toml, joinpath(outdir, "input.toml"); force=true)

        @printf("  phi=%-8g  %3d pts  %s  -> %s\n",
                phi, length(traj), _fmt_sec(elapsed), outdir)

        # Wi features
        try
            cands_df, _ = extract_wi_features(joinpath(outdir, "summary.csv"); x_mode=x_mode)
            sel_df      = select_tnw_batch(cands_df)
            write_feature_csvs(outdir, cands_df, sel_df)
            push!(feature_rows, ("phi", phi, run_id, outdir, sel_df))
        catch e
            @warn "Wi feature extraction failed for phi=$phi" exception=(e, catch_backtrace())
            push!(feature_rows, ("phi", phi, run_id, outdir, DataFrame()))
        end
    end

    # sweep-level aggregate
    feats_df = aggregate_styext(feature_rows)
    if !isempty(feats_df)
        CSV.write(joinpath(sweep_dir, "wi_features_vs_phi.csv"), feats_df)
        isempty(sweep_id) || CSV.write(
            joinpath(sweep_dir, "wi_features_vs_phi_$(sweep_id).csv"), feats_df)
    end
    return feats_df
end

# ── USYEXT (CaBER) sweep ──────────────────────────────────────────────────────

"""
    run_usyext_sweep(sweep_dir, input_toml; De_values, x_mode, verbose)

Run the usyext (CaBER) integrator for each De in De_values.
sweep_dir/De_<v>/<run_id>/ holds summary.csv, caber_diag.csv,
wi_feature_candidates.csv, wi_features_selected.csv, input.toml.

Returns (wi_features_vs_De, Wi_e_vs_De) DataFrames.
"""
function run_usyext_sweep(sweep_dir::AbstractString,
                          input_toml::AbstractString;
                          De_values::Vector{Float64} = Float64[],
                          x_mode::String             = "E2",
                          verbose::Bool              = false)

    isfile(input_toml) || error("input.toml not found: $input_toml")

    # read De_values from TOML if not supplied
    if isempty(De_values)
        d = TOML.parsefile(input_toml)
        flow = get(d, "flow", Dict())
        dv   = get(flow, "De_values", nothing)
        if dv === nothing
            dv = get(flow, "De", [1.0])
        end
        De_values = Float64.(isa(dv, AbstractVector) ? dv : [dv])
    end

    isempty(De_values) && error("No De_values to sweep in $input_toml")

    feature_rows = Tuple[]
    sweep_id     = ""

    for De in De_values
        De_dir = joinpath(sweep_dir, @sprintf("De_%g", De))
        mkpath(De_dir)

        # Load context with De override
        ctx, flow, plan, tol =
            UsyextRunContext.load_usyext_context(input_toml; De=De)

        # Pre-stretch IC from [initial_condition] (nothing ⇒ equilibrium (1/3)I)
        M0 = UsyextRunContext.build_initial_M(input_toml, ctx)

        fe_sym   = ctx.fe_model
        drag_sym = ctx.kernel.drag_lut.drag_model
        Nm_val   = ctx.Nm
        run_id   = _run_id(fe_sym, drag_sym, Nm_val, @sprintf("De=%.4g", De))
        outdir   = joinpath(De_dir, run_id)
        mkpath(outdir)

        isempty(sweep_id) && (sweep_id = _run_id(fe_sym, drag_sym, Nm_val))

        t0      = time()
        traj    = try
            UsyextIntegrator.integrate_usyext(ctx, flow, plan, tol; M0=M0)
        catch e
            # A failed cell must not kill the sweep — skip it and keep the
            # remaining De points so Wi_e_vs_De.csv is still produced.
            @warn "integrate_usyext failed for De=$De — skipping cell" exception=(e, catch_backtrace())
            push!(feature_rows, ("De", De, run_id, outdir, DataFrame()))
            continue
        end
        elapsed = time() - t0

        UsyextIntegrator.write_summary_csv(joinpath(outdir, "summary.csv"), traj)
        UsyextIntegrator.write_caber_diag_csv(joinpath(outdir, "caber_diag.csv"), traj)
        cp(input_toml, joinpath(outdir, "input.toml"); force=true)

        @printf("  De=%-8g  %3d pts  %s  -> %s\n",
                De, length(traj), _fmt_sec(elapsed), outdir)

        # Wi features
        try
            cands_df, _ = extract_wi_features(joinpath(outdir, "summary.csv"); x_mode=x_mode)
            sel_df      = select_tnw_batch(cands_df)
            write_feature_csvs(outdir, cands_df, sel_df)
            push!(feature_rows, ("De", De, run_id, outdir, sel_df))
        catch e
            @warn "Wi feature extraction failed for De=$De" exception=(e, catch_backtrace())
            push!(feature_rows, ("De", De, run_id, outdir, DataFrame()))
        end
    end

    # sweep-level aggregates
    feat_df, wie_df = aggregate_caber(feature_rows)
    if !isempty(feat_df)
        CSV.write(joinpath(sweep_dir, "wi_features_vs_De.csv"), feat_df)
        isempty(sweep_id) || CSV.write(
            joinpath(sweep_dir, "wi_features_vs_De_$(sweep_id).csv"), feat_df)
    end
    if !isempty(wie_df)
        CSV.write(joinpath(sweep_dir, "Wi_e_vs_De.csv"), wie_df)
    end
    return feat_df, wie_df
end
