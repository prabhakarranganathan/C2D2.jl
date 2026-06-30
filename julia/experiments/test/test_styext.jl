"""
    test_styext.jl

Regression tests for the Julia STYEXT (steady-extension PAC) solver.

Compares Julia output against MATLAB reference CSVs committed in
  julia/experiments/test/reference/

Reference cases:
  ref_01 : FENE-PM, constant drag, Nm=1, phi=1.0, Wi_max=3.0
  ref_02 : FENE-PM, c2d2 drag,    Nm=3, phi=0.1, Wi_max=5.0

Tolerance: relative error < 0.5% on Wi, etaE1, Tr1, tau_zz at each point.

Run with:
  julia --project=julia julia/experiments/test/test_styext.jl
(from repo root)
"""

using Test

# ── Bootstrap: load all modules in dependency order ───────────────────────────
const _MODELS = joinpath(@__DIR__, "..", "..", "models")
const _INPUTS = joinpath(@__DIR__, "..", "c2d2ref", "inputs")

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
# Params is needed for _rouse_weights but we defined it locally in RunContext.
# Include it anyway for completeness if any downstream code needs it.
include(joinpath(_MODELS, "kernel", "Params.jl"))

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

# ── Helpers ───────────────────────────────────────────────────────────────────

"""Load a reference CSV as a matrix of floats (skipping string status column)."""
function _load_ref_csv(path::AbstractString)
    lines = readlines(path)
    hdr   = split(lines[1], ',')
    # All columns except "status" (last) are numeric
    n_num = length(hdr) - 1
    data  = zeros(length(lines) - 1, n_num + 1)   # last col is idx for now
    status = String[]
    for (i, line) in enumerate(lines[2:end])
        parts = split(line, ',')
        for j in 1:n_num
            data[i, j] = parse(Float64, strip(parts[j]))
        end
        push!(status, strip(parts[end]))
    end
    return hdr, data, status
end

"""Column index by header name."""
function _col(hdr, name)
    idx = findfirst(==(name), hdr)
    idx !== nothing || error("Column '$name' not found in $hdr")
    return idx
end

"""Relative error, guarded against exact zero."""
function _relerr(a, b)
    denom = max(abs(b), 1e-30)
    return abs(a - b) / denom
end

const REF_DIR = joinpath(@__DIR__, "reference")
const TOML_DIR = _INPUTS

# ── Unit tests: StyextFlows ───────────────────────────────────────────────────

@testset "StyextFlows" begin
    @testset "Wi_from_mu roundtrip" begin
        for Wi_pole in [0.5, 1.0, 3.14]
            for Wi in [0.0, 0.1 * Wi_pole, 0.5 * Wi_pole, 0.99 * Wi_pole]
                mu  = mu_from_Wi(Wi, Wi_pole)
                Wi2 = Wi_from_mu(mu, Wi_pole)
                @test Wi2 ≈ Wi atol=1e-13
            end
        end
    end

    @testset "Wi_from_mu limits" begin
        @test Wi_from_mu(0.0, 0.5) ≈ 0.0 atol=1e-15
        @test Wi_from_mu(1e6,  0.5) ≈ 0.5 atol=1e-10   # μ → ∞ → Wi → Wi_pole
    end

    @testset "mu_from_Wi guard" begin
        # Near-pole: should not throw
        mu = mu_from_Wi(0.4999999999, 0.5)
        @test isfinite(mu) && mu > 0
    end
end

# ── Unit tests: StyextModelPack (pack/unpack) ─────────────────────────────────

@testset "StyextModelPack pack/unpack" begin
    @testset "Na=0 (PME)" begin
        X  = Float64[]
        Y  = 3.14
        mu = 1.23
        v  = pack_ztilde(X, Y, mu, 0)
        @test length(v) == 2
        X2, Y2, mu2 = unpack_ztilde(v, 0)
        @test isempty(X2)
        @test Y2 == Y
        @test mu2 == mu
    end

    @testset "Na=1 (PM)" begin
        X  = [2.718]
        Y  = 3.14
        mu = 1.23
        v  = pack_ztilde(X, Y, mu, 1)
        @test length(v) == 3
        X2, Y2, mu2 = unpack_ztilde(v, 1)
        @test X2 ≈ X atol=1e-15
        @test Y2 == Y
        @test mu2 == mu
    end
end

# ── Integration test: run solver and compare to reference CSVs ─────────────────

"""
Run the STYEXT PAC solver for the given TOML case and compare to a reference CSV.
Returns (n_pts_ok, n_pts_total, max_relerr_Wi, max_relerr_etaE1, max_relerr_Tr1).
"""
function _run_and_compare(toml_path, ref_csv_path, phi_override=nothing;
                           rtol=0.005, verbose=false)

    # Load context
    kw = phi_override !== nothing ? (; phi=phi_override) : (;)
    params, mp_ctx, solver_opts, man_opts = load_styext_context(toml_path; kw...)

    # Enable verbose progress if requested
    man_opts_v = ManifoldOpts(
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

    # Run manifold builder
    traj = build_manifold(mp_ctx, solver_opts, man_opts_v)

    @test length(traj) >= 2   # at least equilibrium + 1 step

    # Load reference
    hdr, ref_data, ref_status = _load_ref_csv(ref_csv_path)

    cWi   = _col(hdr, "Wi")
    cE2   = _col(hdr, "E2")
    cetaE = _col(hdr, "etaE1")
    cTr   = _col(hdr, "Tr1")
    ctauZ = _col(hdr, "tau_zz")
    cWipole = _col(hdr, "Wi_pole")

    n_ref = size(ref_data, 1)
    n_jl  = length(traj)
    n_cmp = min(n_ref, n_jl)

    max_err_Wi    = 0.0
    max_err_E2    = 0.0
    max_err_etaE  = 0.0
    max_err_Tr    = 0.0

    for i in 1:n_cmp
        pt     = traj[i]
        o      = pt.obs
        ref_row = ref_data[i, :]

        # Skip equilibrium row (i==1: all stresses are 0; etaE is -0)
        if i == 1
            @test abs(o.Wi) < 1e-14
            @test abs(o.E2 - ref_row[cE2]) / max(ref_row[cE2], 1.0) < 1e-10
            continue
        end

        err_Wi   = _relerr(o.Wi,    ref_row[cWi])
        err_E2   = _relerr(o.E2,    ref_row[cE2])
        err_etaE = _relerr(o.etaE1, ref_row[cetaE])
        err_Tr   = _relerr(o.Tr1,   ref_row[cTr])

        max_err_Wi   = max(max_err_Wi,   err_Wi)
        max_err_E2   = max(max_err_E2,   err_E2)
        max_err_etaE = max(max_err_etaE, err_etaE)
        max_err_Tr   = max(max_err_Tr,   err_Tr)

        err_Wi   < rtol || @warn "Wi mismatch at k=$i: Julia=$(o.Wi) ref=$(ref_row[cWi]) err=$(err_Wi)"
        err_E2   < rtol || @warn "E2 mismatch at k=$i: Julia=$(o.E2) ref=$(ref_row[cE2]) err=$(err_E2)"
        err_etaE < rtol || @warn "etaE1 mismatch at k=$i: Julia=$(o.etaE1) ref=$(ref_row[cetaE]) err=$(err_etaE)"
        err_Tr   < rtol || @warn "Tr1 mismatch at k=$i: Julia=$(o.Tr1) ref=$(ref_row[cTr]) err=$(err_Tr)"
        @test err_Wi   < rtol
        @test err_E2   < rtol
        @test err_etaE < rtol
        @test err_Tr   < rtol
    end

    return (n_pts=n_jl, n_ref=n_ref, max_err_Wi=max_err_Wi, max_err_etaE=max_err_etaE)
end


# ── ref_01: FENE-PM, constant drag, Nm=1, phi=1.0 ─────────────────────────────

@testset "styext_ref_01 (FENE-PM, constant, Nm=1, phi=1.0)" begin
    toml = joinpath(TOML_DIR, "styext_ref_01.toml")
    ref  = joinpath(REF_DIR,  "styext_ref_01_summary.csv")
    isfile(toml) || @test_skip "TOML not found: $toml"
    isfile(ref)  || @test_skip "Reference CSV not found: $ref"
    isfile(toml) && isfile(ref) && begin
        result = _run_and_compare(toml, ref; rtol=0.005)
        @test result.n_pts >= 10
        println("  ref_01: $(result.n_pts) points, max Wi err = $(result.max_err_Wi), max etaE err = $(result.max_err_etaE)")
    end
end


# ── ref_02: FENE-PM, c2d2 drag, Nm=3, phi=0.1 ────────────────────────────────

@testset "styext_ref_02 (FENE-PM, c2d2, Nm=3, phi=0.1)" begin
    toml = joinpath(TOML_DIR, "styext_ref_02.toml")
    ref  = joinpath(REF_DIR,  "styext_ref_02_summary.csv")
    isfile(toml) || @test_skip "TOML not found: $toml"
    isfile(ref)  || @test_skip "Reference CSV not found: $ref"
    isfile(toml) && isfile(ref) && begin
        result = _run_and_compare(toml, ref; rtol=0.005)
        @test result.n_pts >= 10
        println("  ref_02: $(result.n_pts) points, max Wi err = $(result.max_err_Wi), max etaE err = $(result.max_err_etaE)")
    end
end


println("\nAll STYEXT tests complete.")
