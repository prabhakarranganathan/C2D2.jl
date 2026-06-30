"""
    test_kernel.jl

Unit tests for the C2D2 kernel modules.
Each test group has a reference value computed from the MATLAB codebase
(or derived analytically) so we can confirm the Julia translation is correct.

Run with:  julia --project=. test/test_kernel.jl
"""

# ── Bootstrap (works whether run standalone or via Pkg.test) ──────────────────
using Test

# Model files live in julia/models/kernel/ (two levels up from julia/experiments/test/)
const _MODELS = joinpath(@__DIR__, "..", "..", "models")

include(joinpath(_MODELS, "kernel", "Fene.jl"))
include(joinpath(_MODELS, "kernel", "Drag.jl"))
include(joinpath(_MODELS, "kernel", "Spectrum.jl"))
include(joinpath(_MODELS, "kernel", "CoeffsRelax.jl"))

using .Fene
using .Drag
using .Spectrum
using .CoeffsRelax

const TOL = 1e-10   # exact arithmetic tolerance
const RTOL = 1e-6   # relative tolerance for LUT / iterative results

# ══════════════════════════════════════════════════════════════════════════════
@testset "Fene kernel" begin

    # ── Peterlin primitive ────────────────────────────────────────────────────
    @testset "fene_peterlin" begin
        Q0_sq   = 1.0
        Qinf_sq = 100.0

        # f(0) = (100-1)/(100-0) = 0.99
        fval, df = fene_peterlin(0.0, Q0_sq, Qinf_sq)
        @test fval ≈ 0.99 atol=TOL
        @test df   ≈ 0.99/100.0 atol=TOL

        # f(50) = 99/50 = 1.98,  df = 99/50² = 0.0396
        fval, df = fene_peterlin(50.0, Q0_sq, Qinf_sq)
        @test fval ≈ 1.98  atol=TOL
        @test df   ≈ 99.0/50.0^2 atol=TOL

        # Pole: chi = Qinf_sq should throw
        @test_throws ArgumentError fene_peterlin(100.0, Q0_sq, Qinf_sq)
        @test_throws ArgumentError fene_peterlin(101.0, Q0_sq, Qinf_sq)
    end

    # ── FENE-PME ──────────────────────────────────────────────────────────────
    @testset "FenePME" begin
        Nm = 4; Q0_sq = 1.0; Qinf_sq = 100.0
        m  = FenePME(Nm, Q0_sq, Qinf_sq)
        Y  = 8.0    # chi = Y/Nm = 2.0
        # f(2) = 99/(100-2) = 99/98
        expected_f = 99.0/98.0
        r = eval_fene(m, Float64[], Y; want_derivs=true)
        @test all(r.fp .≈ expected_f)
        @test size(r.df_dX) == (Nm, 0)   # Na=0
        # df/dY = df/dchi * (1/Nm) = (99/98²) / 4
        expected_dfY = (99.0/98.0^2) / Nm
        @test all(r.df_dY .≈ expected_dfY)
    end

    # ── FENE-PM ───────────────────────────────────────────────────────────────
    @testset "FenePM" begin
        Nm = 4; Q0_sq = 1.0; Qinf_sq = 100.0
        m  = FenePM(Nm, Q0_sq, Qinf_sq)
        Xa = [3.0]    # X = 3, chi = 3
        Y  = 0.0      # unused for PM
        expected_f = 99.0/(100.0-3.0)   # = 99/97
        r = eval_fene(m, Xa, Y; want_derivs=true)
        @test all(r.fp .≈ expected_f)
        @test size(r.df_dX) == (Nm, 1)
        expected_dfdX = 99.0/97.0^2
        @test all(r.df_dX .≈ expected_dfdX)
        @test all(r.df_dY .== 0.0)
    end

    # ── FENE-P (TFNA) ─────────────────────────────────────────────────────────
    @testset "FeneP" begin
        Nm = 2; Q0_sq = 1.0; Qinf_sq = 100.0
        m  = build_fene_model(:fenep, Nm, Q0_sq, Qinf_sq)
        @test m isa FeneP
        @test size(m.W) == (Nm, Nm)
        # Rouse weight matrix columns should sum to 1
        @test all(sum(m.W; dims=1) .≈ 1.0)
        # Row sums: Σ_p W[i,p] = Σ_p Π[i,p]² = 1 (ortho rows)
        @test all(sum(m.W; dims=2) .≈ 1.0)

        # At equilibrium: X_p = Q0²/3 * ... 
        # Simple smoke test: fp at near-zero X should give ≈ Q0²/Qinf² correction
        Xa = fill(0.1, Nm)
        Y  = 0.0
        r  = eval_fene(m, Xa, Y; want_derivs=true)
        @test length(r.fp) == Nm
        @test size(r.df_dX) == (Nm, Nm)
        # At chi ≈ 0.1*W_col_sum ≈ 0.1 each: f ≈ 99/99.9 ≈ 0.991
        @test all(r.fp .> 0.98) && all(r.fp .< 1.02)
    end

    # ── Hookean ──────────────────────────────────────────────────────────────
    @testset "Hookean" begin
        Nm = 3
        # Primary name
        m  = build_fene_model(:hookean, Nm, 1.0, 1000.0)
        @test m isa Hookean
        r  = eval_fene(m, Float64[], 0.0; want_derivs=true)
        @test all(r.fp .== 1.0)           # f=1 always
        @test size(r.df_dX) == (Nm, 0)   # Na=0
        @test all(r.df_dY .== 0.0)
        # Legacy alias :oldroyd_b maps to the same Hookean struct
        m2 = build_fene_model(:oldroyd_b, Nm, 1.0, 1000.0)
        @test m2 isa Hookean
    end

    # ── na_for_model ──────────────────────────────────────────────────────────
    @testset "na_for_model" begin
        @test na_for_model(:hookean,   5) == 0   # primary name
        @test na_for_model(:oldroyd_b, 5) == 0   # legacy alias
        @test na_for_model(:fenepme,   5) == 0
        @test na_for_model(:fenepm,    5) == 1
        @test na_for_model(:fenep,     5) == 5
        @test_throws ErrorException na_for_model(:fenex, 5)
    end

end  # Fene tests

# ══════════════════════════════════════════════════════════════════════════════
@testset "Drag kernel" begin

    # ── Draining ratio ────────────────────────────────────────────────────────
    @testset "drag_draining_ratio" begin
        hK = 0.025
        # Non-draining limit
        alpha_nd = drag_draining_ratio(hK, Inf)
        @test alpha_nd ≈ 3π/1.6366 rtol=1e-4

        # Single Kuhn segment: α = 6π^(3/2) h*_k
        # from formula: P1→0.91/1, P2→1.6-1.1-0.45≈0.05,
        # not exactly 6π^(3/2) h*_k because of finite polynomial.
        # Just check it's positive and finite.
        alpha_1 = drag_draining_ratio(hK, 1.0)
        @test alpha_1 > 0.0 && isfinite(alpha_1)

        # Larger N → closer to non-draining
        # Zimm convergence is O(N^{-1/2}); N=10000 still ~5–6% below the limit
        alpha_large = drag_draining_ratio(hK, 10000.0)
        @test alpha_large > alpha_1
        @test alpha_large ≈ alpha_nd rtol=0.10
    end

    # ── drag_ratio_zimm: constant model ──────────────────────────────────────
    @testset "drag constant" begin
        zr, _ = drag_ratio_zimm(2.0, 1.0, 0.025, 1000.0, :constant, :nondraining)
        @test zr == 1.0
    end

    # ── drag_ratio_zimm: c2d2 at E=1 (coiled) = 1.0 ─────────────────────────
    @testset "drag c2d2 at equilibrium" begin
        # At E=1, phi=0 (dilute limit): E > phi so code takes the tension branch.
        # G→1 at E=1 (Y=0, B→∞ limit), so ζ/ζ_Z = zbT·nT/G = 1 by construction.
        zr, diag = drag_ratio_zimm(1.0, 0.0, 0.025, 1000.0, :c2d2, :partialdraining)
        @test zr ≈ 1.0 rtol=0.01
        # phi=0 < E=1 → tension-blob branch
        @test diag.blob_type === :T
    end

    # ── drag_ratio_zimm: c2d2 increases with stretch ─────────────────────────
    @testset "drag c2d2 monotone in E" begin
        E_vals = [1.0, 2.0, 5.0, 10.0]
        phi = 1.5; hK = 0.025; NK = 1000.0
        zr_vals = [drag_ratio_zimm(e, phi, hK, NK, :c2d2, :partialdraining)[1]
                   for e in E_vals]
        # Should increase then potentially level off near full extension
        # At minimum, values should be positive and finite
        @test all(zr_vals .> 0)
        @test all(isfinite.(zr_vals))
        # Stretched chain (E>phi) should have higher drag than coiled (E<phi)
        # phi=1.5: E=1.0 is coiled, E=2.0 is stretched
        @test zr_vals[2] > zr_vals[1]
    end

    # ── DragAssessor: LUT eval matches direct eval ────────────────────────────
    @testset "DragAssessor consistency" begin
        phi = 2.0; hK = 0.025; NK = 500.0
        a = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining; n_points=300)

        # Test at several E values that the LUT ≈ direct evaluation
        for E in [1.0, 1.5, 2.5, 5.0, 10.0, 20.0]
            zr_direct, _ = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining)
            zr_lut = eval_drag(a, E)
            @test zr_lut ≈ zr_direct rtol=5e-3
        end
    end

end  # Drag tests

# ══════════════════════════════════════════════════════════════════════════════
@testset "Spectrum kernel" begin

    # ── Rouse scaling ─────────────────────────────────────────────────────────
    @testset "rouse_scaling" begin
        lhat = relative_spectrum(:rouse_scaling, 4, 1000.0, 0.025)
        @test lhat[1] ≈ 1.0
        @test lhat[2] ≈ 1/4   rtol=TOL   # p=2: 1^{-2}/2^{-2} = 4... wait
        # λ̂_p = p^{-2}, normalised by λ̂_1 = 1^{-2} = 1
        # So λ̂_2 = 2^{-2}/1^{-2} = 1/4
        @test lhat[2] ≈ 1.0/4.0  rtol=TOL
        @test lhat[4] ≈ 1.0/16.0 rtol=TOL
    end

    # ── Zimm scaling ──────────────────────────────────────────────────────────
    @testset "zimm_scaling" begin
        lhat = relative_spectrum(:zimm_scaling, 4, 1000.0, 0.025)
        @test lhat[1] ≈ 1.0
        @test lhat[2] ≈ 2.0^(-1.5) rtol=TOL
    end

    # ── Rouse full: consistent with analytical ────────────────────────────────
    @testset "rouse_full vs rouse_scaling (large Nm)" begin
        # For large Nm the full eigenvalue spectrum → p^{-2} scaling
        lhat_full    = relative_spectrum(:rouse_full,    20, 1000.0, 0.025)
        lhat_scaling = relative_spectrum(:rouse_scaling, 20, 1000.0, 0.025)
        # First mode is always 1
        @test lhat_full[1] ≈ 1.0
        # Slow modes agree well
        @test lhat_full[1] ≈ lhat_scaling[1] rtol=0.01
        @test lhat_full[2] ≈ lhat_scaling[2] rtol=0.05
    end

    # ── SpectrumAssessor: Λ=1 at equilibrium ─────────────────────────────────
    @testset "SpectrumAssessor: Lambda=1 at E=1" begin
        sa = build_spectrum_assessor(
            spectrum_model = :rouse_scaling,
            Nm = 4, NK = 1000.0, hK_star = 0.025,
            phi = 0.0, drag_model = :constant, draining_model = :nondraining
        )
        zr_eq = eval_drag(build_drag_assessor(0.0, 0.025, 1000.0, :constant, :nondraining), 1.0)
        lhat, Lambda = eval_spectrum(sa, 1.0, zr_eq)
        @test Lambda ≈ 1.0 rtol=TOL
        @test lhat[1] ≈ 1.0 rtol=TOL
    end

end  # Spectrum tests

# ══════════════════════════════════════════════════════════════════════════════
@testset "CoeffsRelax kernel" begin

    # ── closure_from_M_diag: equilibrium state ────────────────────────────────
    @testset "closure_from_M_diag equilibrium" begin
        Nm = 3
        # Equilibrium: tr M_p = Q0² = 1 for all p in our scaling
        # So M_zz = M_rr = 1/3 (isotropic)
        M_diag = fill(1.0/3.0, Nm, 2)   # [Mzz, Mrr] = [1/3, 1/3]
        Sp     = [isodd(p) ? (2.0/(Nm+1))*cot(p*π/(2*(Nm+1)))^2 : 0.0 for p in 1:Nm]

        Z_pme = closure_from_M_diag(M_diag, :fenepme, Nm, Sp)
        # tr M_p = 1/3 + 2*1/3 = 1.  Y = Σ S_p * 1 = 3.
        @test Z_pme.Y ≈ Float64(Nm) rtol=1e-6
        @test length(Z_pme.Xa) == 0

        Z_pm = closure_from_M_diag(M_diag, :fenepm, Nm, Sp)
        @test Z_pm.Y ≈ Float64(Nm) rtol=1e-6
        # X = mean tr M_p = 1
        @test Z_pm.Xa[1] ≈ 1.0 rtol=TOL

        Z_p = closure_from_M_diag(M_diag, :fenep, Nm, Sp)
        @test all(Z_p.Xa .≈ 1.0)

        # Hookean (:oldroyd_b alias): same as fenepme — Xa is empty
        Z_ob = closure_from_M_diag(M_diag, :oldroyd_b, Nm, Sp)
        @test Z_ob.Y ≈ Float64(Nm) rtol=1e-6
        @test length(Z_ob.Xa) == 0
    end

    # ── eval_coeffs: equilibrium gives sigma = 1/(3*1*lhat_p) ────────────────
    @testset "eval_coeffs at equilibrium" begin
        Nm = 2; NK = 100.0; hK = 0.025; phi = 0.0
        Q0_sq = 1.0; Qinf_sq = NK

        fene    = build_fene_model(:fenepme, Nm, Q0_sq, Qinf_sq)
        drag    = build_drag_assessor(phi, hK, NK, :constant, :nondraining)
        spec    = build_spectrum_assessor(
            spectrum_model = :rouse_scaling, Nm=Nm, NK=NK, hK_star=hK,
            phi=phi, drag_model=:constant, draining_model=:nondraining)
        kernel  = build_coeffs_kernel(fene, drag, spec)

        # Equilibrium Z: Y = Nm (since tr Mp = 1 for each mode and Sp sums to Nm)
        Y  = Float64(Nm)
        Z  = CoeffsRelax.ClosureZ(Float64[], Y)
        r  = eval_coeffs(kernel, Z)

        # At equil with constant drag: Λ=1, λ̂_p = p^{-2} (rouse_scaling)
        # f_p^{eq}: chi = Y/Nm = 1; f = (100-1)/(100-1) = 1
        # theta_p = f_p / (1 * lambda_hat_p) = 1 * p^2
        # sigma_p = 1 / (3 * 1 * lambda_hat_p) = p^2 / 3
        @test r.Lambda ≈ 1.0 rtol=1e-6
        @test r.fp[1]  ≈ 1.0 rtol=1e-4    # f at chi=1 for NK=100: 99/99=1 ✓

        # sigma_1 * theta_1: sigma = 1/(3*lhat_1) = 1/3,  theta = f/lhat_1 = 1
        @test r.sigma_p[1] ≈ 1.0/3.0 rtol=1e-4
        @test r.theta_p[1] ≈ 1.0     rtol=1e-4

        # Mode 2 rouse scaling: lhat_2 = 1/4
        @test r.sigma_p[2] ≈ 4.0/3.0 rtol=1e-4
        @test r.theta_p[2] ≈ 4.0     rtol=1e-4
    end

    # ── eval_coeffs with Hookean: fp=1, sigma=1/(3*lhat_p), theta=1/lhat_p ───
    @testset "eval_coeffs Hookean at equilibrium" begin
        Nm = 2; NK = 1000.0; hK = 0.025; phi = 0.0
        fene    = build_fene_model(:oldroyd_b, Nm, 1.0, NK)
        drag    = build_drag_assessor(phi, hK, NK, :constant, :nondraining)
        spec    = build_spectrum_assessor(
            spectrum_model = :rouse_scaling, Nm=Nm, NK=NK, hK_star=hK,
            phi=phi, drag_model=:constant, draining_model=:nondraining)
        kernel  = build_coeffs_kernel(fene, drag, spec)

        # Y = Nm at equilibrium (tr Mp = 1)
        Z = CoeffsRelax.ClosureZ(Float64[], Float64(Nm))
        r = eval_coeffs(kernel, Z)

        @test r.Lambda ≈ 1.0  rtol=1e-10
        @test all(r.fp .≈ 1.0)     # Hookean: f=1 always
        # Mode 1: lhat_1=1 → sigma=1/3, theta=1
        @test r.sigma_p[1] ≈ 1.0/3.0 rtol=1e-10
        @test r.theta_p[1] ≈ 1.0     rtol=1e-10
        # Mode 2 rouse: lhat_2=1/4 → sigma=4/3, theta=4
        @test r.sigma_p[2] ≈ 4.0/3.0 rtol=1e-10
        @test r.theta_p[2] ≈ 4.0     rtol=1e-10
    end

    # ── eval_coeffs: theta_p > sigma_p * 3 near the pole ─────────────────────
    @testset "eval_coeffs: fp increases toward pole" begin
        Nm = 1; NK = 10.0; hK = 0.025; phi = 0.0
        Q0_sq = 1.0; Qinf_sq = NK
        fene    = build_fene_model(:fenepme, Nm, Q0_sq, Qinf_sq)
        drag    = build_drag_assessor(phi, hK, NK, :constant, :nondraining)
        spec    = build_spectrum_assessor(
            spectrum_model=:rouse_scaling, Nm=Nm, NK=NK, hK_star=hK,
            phi=phi, drag_model=:constant, draining_model=:nondraining)
        kernel  = build_coeffs_kernel(fene, drag, spec)

        Y_eq  = Float64(Nm)   # equilibrium
        Y_str = 8.0            # strongly stretched (Y/Nm = 8, Qinf = 10)
        r_eq  = eval_coeffs(kernel, CoeffsRelax.ClosureZ(Float64[], Y_eq))
        r_str = eval_coeffs(kernel, CoeffsRelax.ClosureZ(Float64[], Y_str))

        # fp should be larger at higher stretch (closer to pole)
        @test r_str.fp[1] > r_eq.fp[1]
        # theta_p should also be larger (faster relaxation rate)
        @test r_str.theta_p[1] > r_eq.theta_p[1]
    end

end  # CoeffsRelax tests

# ══════════════════════════════════════════════════════════════════════════════
println("\n✓  All kernel tests passed.")
