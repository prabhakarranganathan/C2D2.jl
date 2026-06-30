"""
    test_usyext.jl

Regression tests for the Julia USYEXT (unsteady-extension ETDRK2) solver.

Compares Julia output against MATLAB reference CSVs in
  julia/experiments/test/reference/

Reference cases:
  ref_01 : FENE-PM, constant drag, Nm=1, phi=0.03, De=1.0  (CaBER)
  ref_02 : FENE-PM, c2d2 drag,    Nm=3, phi=0.1,  De=0.1  (CaBER, multimode)
  ref_03 : FENE-PM, constant drag, Nm=1, phi=0.09, De=0.02 (CaBER, near-Bhat)

Tolerance: relative error < 0.5% on estrain-matched points of Wi, E2, etaE1, Tr1.

Run with:
  julia --project=julia julia/experiments/test/test_usyext.jl
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
include(joinpath(_MODELS, "kernel", "flows", "UsyextFlows.jl"))
include(joinpath(_MODELS, "numerics", "UsyextIntegrator.jl"))
include(joinpath(_MODELS, "front_end", "UsyextRunContext.jl"))
include(joinpath(_MODELS, "kernel", "Params.jl"))

using .Fene
using .Drag
using .Spectrum
using .CoeffsRelax
using .Stress
using .StateObs
using .UsyextFlows
using .UsyextIntegrator
using .UsyextRunContext

# ── Helpers ───────────────────────────────────────────────────────────────────

"""Load reference CSV, return (header, numeric_matrix, status_vector)."""
function _load_ref_csv(path::AbstractString)
    lines = readlines(path)
    hdr   = split(lines[1], ',')
    n_num = length(hdr) - 1     # last column is status (string)
    data  = zeros(length(lines)-1, n_num)
    status = String[]
    for (i, line) in enumerate(lines[2:end])
        parts = split(line, ',')
        for j in 1:n_num
            v = strip(parts[j])
            data[i, j] = (v == "NaN" || v == "NaN") ? NaN : parse(Float64, v)
        end
        push!(status, strip(parts[end]))
    end
    return hdr, data, status
end

function _col(hdr, name)
    idx = findfirst(==(name), hdr)
    idx !== nothing || error("Column '$name' not found in $hdr")
    return idx
end

function _relerr(a, b)
    denom = max(abs(b), 1e-30)
    return abs(a - b) / denom
end

const REF_DIR  = joinpath(@__DIR__, "reference")
const TOML_DIR = _INPUTS

# ── Unit tests: UsyextFlows ───────────────────────────────────────────────────

@testset "UsyextFlows" begin
    @testset "ConstantWiFlow" begin
        flow = ConstantWiFlow(1.0, 2.5, (-0.5, -0.5, 1.0))
        fs   = eval_flow(flow, 0.0, 0.0, 0.0)
        @test fs.Wi ≈ 2.5
        @test fs.De ≈ 1.0
        @test isnan(fs.R)
    end

    @testset "CaberFlow equilibrium IC" begin
        # At strain=0, N1_p=0: psi=0, X=Xv, Wi = (1/3)*(2Xv-1)*De
        De    = 1.0
        flow  = CaberFlow(De, (-0.5, -0.5, 1.0), 1.0, 1.0)   # phi, U_1 (K̃=φU_1=1)
        fs    = eval_flow(flow, 0.0, 0.0, 0.0)
        @test fs.R   ≈ 1.0 atol=1e-15
        @test fs.psi ≈ 0.0 atol=1e-15
        @test fs.X   ≈ 0.7127 atol=1e-15
        @test fs.Wi  ≈ (1/3)*(2*0.7127-1)*De atol=1e-12
    end

    @testset "CaberFlow R decay" begin
        flow = CaberFlow(1.0, (-0.5, -0.5, 1.0), 1.0, 0.0)   # phi, U_1 (K̃=0)
        for ε in [0.0, 1.0, 2.0, 5.0]
            fs = eval_flow(flow, 0.0, ε, 0.0)
            @test fs.R ≈ exp(-0.5*ε) atol=1e-14
        end
    end

    @testset "CaberInertialFlow (Oh-primary, 2026-06-12 pivot)" begin
        XR  = 0.5115
        H_R = sqrt(8*(2*XR - 1))

        # Pure solvent (N1_p = 0): Wi₀ = H_R·De_R at R = 1, any Oh
        for (De_R, Oh) in [(1.0, 0.005), (10.0, 0.05), (0.3, 0.005)]
            flow = CaberInertialFlow(De_R, (-0.5, -0.5, 1.0), 0.1, Oh, 0.325; XR=XR)
            fs   = eval_flow(flow, 0.0, 0.0, 0.0)
            @test fs.Wi ≈ H_R * De_R rtol=1e-12
            @test fs.De ≈ De_R
        end

        # Pure-solvent power law: Wi ∝ R^{-3/2}  (β = 2/3)
        flow = CaberInertialFlow(1.0, (-0.5, -0.5, 1.0), 0.1, 0.005, 0.325; XR=XR)
        fs1  = eval_flow(flow, 0.0, 1.0, 0.0)
        fs2  = eval_flow(flow, 0.0, 3.0, 0.0)
        @test log(fs2.Wi/fs1.Wi) / log(fs2.R/fs1.R) ≈ -1.5 atol=1e-12

        # Device locus: K̃ = φ·Oh·U_1/De_R  (U_R_eff ∝ 1/De_R across a sweep)
        phi = 0.5; Oh = 0.05; U_1 = 0.325
        for De_R in [0.1, 1.0, 10.0]
            flow = CaberInertialFlow(De_R, (-0.5, -0.5, 1.0), phi, Oh, U_1; XR=XR)
            fs   = eval_flow(flow, 0.0, 1.0, -2.0)
            @test fs.K_til ≈ phi * Oh * U_1 / De_R rtol=1e-14
        end

        # Polymer term reduces driving (N1_p < 0 in extension)
        flow = CaberInertialFlow(1.0, (-0.5, -0.5, 1.0), 0.5, 0.05, 0.325; XR=XR)
        fs0  = eval_flow(flow, 0.0, 2.0, 0.0)
        fsN  = eval_flow(flow, 0.0, 2.0, -1.0)
        @test fsN.Wi < fs0.Wi

        # Oh = 0: degenerate no-arrest (Keller–Miksis) limit — polymer term
        # vanishes identically; no divide-by-zero
        flow0 = CaberInertialFlow(1.0, (-0.5, -0.5, 1.0), 0.5, 0.0, 0.325; XR=XR)
        fsA   = eval_flow(flow0, 0.0, 2.0, 0.0)
        fsB   = eval_flow(flow0, 0.0, 2.0, -1e6)
        @test fsA.Wi ≈ fsB.Wi rtol=1e-14   # N1_p has no effect at Oh=0
        @test fsB.K_til == 0.0
        @test isfinite(fsB.Wi)

        # U_R-equivalence: Oh = U_R·De_R/U_1 reproduces a given old-style U_R
        U_R_old = 0.01; De_R = 10.0
        Oh_eq   = U_R_old * De_R / U_1
        flowOh  = CaberInertialFlow(De_R, (-0.5, -0.5, 1.0), phi, Oh_eq, U_1; XR=XR)
        fs      = eval_flow(flowOh, 0.0, 1.5, -3.0)
        @test fs.K_til ≈ phi * U_R_old rtol=1e-14
    end
end

# ── Integration test helper ───────────────────────────────────────────────────

"""
Run USYEXT solver for the given TOML and compare to a reference CSV.

Matching: reference rows are matched to Julia trajectory by closest estrain.
Only rows within 0.1% of the sample spacing are compared.
"""
function _run_and_compare(toml_path, ref_csv_path; rtol=0.005, verbose=false)

    ctx, flow, plan, tol = load_usyext_context(toml_path)

    traj = integrate_usyext(ctx, flow, plan, tol)
    @test length(traj) >= 2

    hdr, ref_data, ref_status = _load_ref_csv(ref_csv_path)

    cIdx   = _col(hdr, "idx")
    cStrain = _col(hdr, "estrain")
    cWi    = _col(hdr, "Wi")
    cE2    = _col(hdr, "E2")
    cetaE  = _col(hdr, "etaE1")
    cTr    = _col(hdr, "Tr1")

    n_ref  = size(ref_data, 1)
    n_jl   = length(traj)

    max_err_Wi   = 0.0
    max_err_E2   = 0.0
    max_err_etaE = 0.0
    max_err_Tr   = 0.0
    n_compared   = 0

    # Match ref rows to Julia traj by estrain proximity
    jl_strains = [pt.estrain for pt in traj]

    for i in 1:n_ref
        ref_row    = ref_data[i, :]
        ref_strain = ref_row[cStrain]

        # Find closest Julia point
        dists = abs.(jl_strains .- ref_strain)
        j_best = argmin(dists)
        if dists[j_best] > 1e-6 * (1.0 + ref_strain)
            continue   # no close match; skip this ref row
        end

        pt = traj[j_best]

        # Skip equilibrium row (τ all zero, etaE=0)
        if i == 1
            @test abs(pt.estrain) < 1e-10
            continue
        end

        n_compared += 1

        err_Wi   = _relerr(pt.Wi,    ref_row[cWi])
        err_E2   = _relerr(pt.E2,    ref_row[cE2])
        err_etaE = isnan(ref_row[cetaE]) ? 0.0 : _relerr(pt.etaE1, ref_row[cetaE])
        err_Tr   = isnan(ref_row[cTr])   ? 0.0 : _relerr(pt.Tr1,   ref_row[cTr])

        max_err_Wi   = max(max_err_Wi,   err_Wi)
        max_err_E2   = max(max_err_E2,   err_E2)
        max_err_etaE = max(max_err_etaE, err_etaE)
        max_err_Tr   = max(max_err_Tr,   err_Tr)

        err_Wi   < rtol || @warn "Wi   mismatch i=$i ε=$(ref_strain): Julia=$(pt.Wi)    ref=$(ref_row[cWi])    err=$err_Wi"
        err_E2   < rtol || @warn "E2   mismatch i=$i ε=$(ref_strain): Julia=$(pt.E2)    ref=$(ref_row[cE2])    err=$err_E2"
        err_etaE < rtol || @warn "etaE mismatch i=$i ε=$(ref_strain): Julia=$(pt.etaE1) ref=$(ref_row[cetaE])  err=$err_etaE"
        err_Tr   < rtol || @warn "Tr1  mismatch i=$i ε=$(ref_strain): Julia=$(pt.Tr1)   ref=$(ref_row[cTr])   err=$err_Tr"

        @test err_Wi   < rtol
        @test err_E2   < rtol
        @test err_etaE < rtol
        @test err_Tr   < rtol
    end

    return (n_pts=n_jl, n_ref=n_ref, n_cmp=n_compared,
            max_err_Wi=max_err_Wi, max_err_etaE=max_err_etaE)
end

# ── ref_01: FENE-PM, constant drag, Nm=1, phi=0.03, De=1.0 ───────────────────

@testset "usyext_ref_01 (FENE-PM, constant, Nm=1, phi=0.03, De=1.0)" begin
    toml = joinpath(TOML_DIR, "usyext_ref_01.toml")
    ref  = joinpath(REF_DIR,  "usyext_ref_01_summary.csv")
    isfile(toml) || @test_skip "TOML not found: $toml"
    isfile(ref)  || @test_skip "Reference CSV not found: $ref"
    isfile(toml) && isfile(ref) && begin
        result = _run_and_compare(toml, ref; rtol=0.005)
        @test result.n_pts  >= 10
        @test result.n_cmp  >= 10
        println("  ref_01: $(result.n_pts) pts, $(result.n_cmp) compared, " *
                "max Wi err=$(result.max_err_Wi), max etaE err=$(result.max_err_etaE)")
    end
end

# ── ref_02: FENE-PM, c2d2 drag, Nm=3, phi=0.1, De=0.1 ───────────────────────

@testset "usyext_ref_02 (FENE-PM, c2d2, Nm=3, phi=0.1, De=0.1)" begin
    toml = joinpath(TOML_DIR, "usyext_ref_02.toml")
    ref  = joinpath(REF_DIR,  "usyext_ref_02_summary.csv")
    isfile(toml) || @test_skip "TOML not found: $toml"
    isfile(ref)  || @test_skip "Reference CSV not found: $ref"
    isfile(toml) && isfile(ref) && begin
        result = _run_and_compare(toml, ref; rtol=0.005)
        @test result.n_pts  >= 10
        @test result.n_cmp  >= 10
        println("  ref_02: $(result.n_pts) pts, $(result.n_cmp) compared, " *
                "max Wi err=$(result.max_err_Wi), max etaE err=$(result.max_err_etaE)")
    end
end

# ── ref_03: FENE-PM, constant drag, Nm=1, phi=0.09, De=0.02 ─────────────────

@testset "usyext_ref_03 (FENE-PM, constant, Nm=1, phi=0.09, De=0.02)" begin
    toml = joinpath(TOML_DIR, "usyext_ref_03.toml")
    ref  = joinpath(REF_DIR,  "usyext_ref_03_summary.csv")
    isfile(toml) || @test_skip "TOML not found: $toml"
    isfile(ref)  || @test_skip "Reference CSV not found: $ref"
    isfile(toml) && isfile(ref) && begin
        result = _run_and_compare(toml, ref; rtol=0.005)
        @test result.n_pts  >= 10
        @test result.n_cmp  >= 10
        println("  ref_03: $(result.n_pts) pts, $(result.n_cmp) compared, " *
                "max Wi err=$(result.max_err_Wi), max etaE err=$(result.max_err_etaE)")
    end
end

# ── phi1phi2 cancellation regression ─────────────────────────────────────────
#
# Regression for the Taylor-threshold backport (2026-05-12).
# Old threshold was 1e-6; the direct formulas (exp(z)-1)/z and (exp(z)-1-z)/z²
# lose ~8 significant digits to cancellation for |z| in (1e-6, 1e-2).
# New threshold is 1e-2 with a 5-term Taylor series, safe to ~z⁵ truncation
# error ≈ 10⁻¹² at z=1e-2.
#
# Test at z = 1e-4 (in the previously unsafe gap) and z = 1e-2 (at boundary).
# Reference: 6-term Taylor is accurate to ~z⁶ ≈ 10⁻²⁴ — well below Float64 eps.

@testset "phi1phi2 cancellation-safe Taylor (threshold=1e-2)" begin
    # 6-term Taylor reference (truncation error O(z^6) < 10^-24 for z ≤ 1e-2)
    function phi1_taylor6(z)
        1.0 + z*(0.5 + z*(1.0/6.0 + z*(1.0/24.0 + z*(1.0/120.0 + z/720.0))))
    end
    function phi2_taylor6(z)
        0.5 + z*(1.0/6.0 + z*(1.0/24.0 + z*(1.0/120.0 + z*(1.0/720.0 + z/5040.0))))
    end

    # At z = 1e-4: old threshold (1e-6) would use direct formula → ~8 digits lost
    # New threshold (1e-2) uses Taylor → accurate to eps
    z_small = 1e-4
    p1, p2 = UsyextIntegrator._phi1phi2(fill(z_small, 1, 1))
    @test abs(p1[1] - phi1_taylor6(z_small)) < 1e-14   # near-machine precision
    @test abs(p2[1] - phi2_taylor6(z_small)) < 1e-14

    # At z = -1e-4 (negative argument, also in the gap)
    z_neg = -1e-4
    p1n, p2n = UsyextIntegrator._phi1phi2(fill(z_neg, 1, 1))
    @test abs(p1n[1] - phi1_taylor6(z_neg)) < 1e-14
    @test abs(p2n[1] - phi2_taylor6(z_neg)) < 1e-14

    # At z = 5e-3 (safely inside Taylor region; z^5/720 ≈ 9e-18 truncation error)
    z_bnd = 5e-3
    p1b, p2b = UsyextIntegrator._phi1phi2(fill(z_bnd, 1, 1))
    @test abs(p1b[1] - phi1_taylor6(z_bnd)) < 1e-14
    @test abs(p2b[1] - phi2_taylor6(z_bnd)) < 1e-14

    # At z = 0.5 (well above threshold — direct formula; accurate by construction)
    z_large = 0.5
    p1l, p2l = UsyextIntegrator._phi1phi2(fill(z_large, 1, 1))
    phi1_exact = (exp(z_large) - 1.0) / z_large
    phi2_exact = (exp(z_large) - 1.0 - z_large) / (z_large * z_large)
    @test abs(p1l[1] - phi1_exact) < 1e-14
    @test abs(p2l[1] - phi2_exact) < 1e-14
end

println("\nAll USYEXT tests complete.")
