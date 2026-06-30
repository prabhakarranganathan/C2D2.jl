"""
    test_evals.jl

Unit tests for julia/models/kernel/evals/ — the M-tensor, stress, and observable
layers built for STYEXT and USYEXT.

All expected values are derived analytically so the tests are independent of
the MATLAB reference CSVs (which serve as integration-level checks in Phase 5).

Run with:  julia --project=julia julia/experiments/test/test_evals.jl
(from the repo root)
"""

using Test

# ── Bootstrap: include kernel modules in dependency order ─────────────────────
const _MODELS = joinpath(@__DIR__, "..", "..", "models")

include(joinpath(_MODELS, "kernel", "Fene.jl"))
include(joinpath(_MODELS, "kernel", "Drag.jl"))
include(joinpath(_MODELS, "kernel", "Spectrum.jl"))
include(joinpath(_MODELS, "kernel", "CoeffsRelax.jl"))

include(joinpath(_MODELS, "kernel", "evals", "MTensor.jl"))
include(joinpath(_MODELS, "kernel", "evals", "Stress.jl"))
include(joinpath(_MODELS, "kernel", "evals", "StateObs.jl"))

using .Fene
using .CoeffsRelax
using .MTensor
using .Stress
using .StateObs

const TOL  = 1e-12   # tight analytic tolerance
const RTOL = 1e-10   # relative tolerance for float arithmetic

# ══════════════════════════════════════════════════════════════════════════════
@testset "MTensor" begin

    # ── modal_trace! ──────────────────────────────────────────────────────────
    @testset "modal_trace!" begin
        M = [0.1 0.2 0.3;
             0.4 0.5 0.6]    # tr: [0.6, 1.5]
        trM = zeros(2)
        modal_trace!(trM, M)
        @test trM[1] ≈ 0.6 atol=TOL
        @test trM[2] ≈ 1.5 atol=TOL

        # Equilibrium: all 1/3 → tr M_p = 1
        Nm = 4
        M_eq  = fill(1.0/3.0, Nm, 3)
        trM_eq = zeros(Nm)
        modal_trace!(trM_eq, M_eq)
        @test all(trM_eq .≈ 1.0)
    end

    # ── modal_E_sq ────────────────────────────────────────────────────────────
    @testset "modal_E_sq" begin
        Nm = 3
        Sp = [0.5, 0.3, 0.2]
        # Equilibrium M → tr M_p = 1 → E² = Σ Sp = 1
        M_eq = fill(1.0/3.0, Nm, 3)
        @test modal_E_sq(M_eq, Sp) ≈ 1.0 atol=TOL

        # Non-uniform M
        M = [0.2 0.3 0.5;   # tr = 1.0
             0.1 0.1 0.3;   # tr = 0.5
             0.4 0.4 0.2]   # tr = 1.0
        expected = 0.5*1.0 + 0.3*0.5 + 0.2*1.0   # = 0.5+0.15+0.2 = 0.85
        @test modal_E_sq(M, Sp) ≈ expected atol=TOL
    end

    # ── closure_from_M_diag_3d: equilibrium ──────────────────────────────────
    @testset "closure_from_M_diag_3d: equilibrium" begin
        Nm = 3
        # Uniform weights for simplicity
        Sp = ones(Nm) / Nm
        M_eq = fill(1.0/3.0, Nm, 3)

        # For fenepm: X = mean(tr M_p) = 1
        Z_pm = closure_from_M_diag_3d(M_eq, :fenepm, Nm, Sp)
        @test Z_pm.Y ≈ 1.0 atol=TOL      # E² = Σ Sp * 1 = 1
        @test Z_pm.E ≈ 1.0 atol=TOL
        @test Z_pm.Xa[1] ≈ 1.0 atol=TOL  # mean tr = 1

        # For fenepme: Xa is empty
        Z_pme = closure_from_M_diag_3d(M_eq, :fenepme, Nm, Sp)
        @test Z_pme.Y ≈ 1.0 atol=TOL
        @test length(Z_pme.Xa) == 0

        # For hookean: Xa is empty
        Z_h = closure_from_M_diag_3d(M_eq, :hookean, Nm, Sp)
        @test Z_h.Y ≈ 1.0 atol=TOL
        @test length(Z_h.Xa) == 0

        # For fenep: Xa[p] = tr M_p = 1 for all p
        Z_p = closure_from_M_diag_3d(M_eq, :fenep, Nm, Sp)
        @test all(Z_p.Xa .≈ 1.0)
    end

    # ── closure_from_M_diag_3d: consistency with 2-D at uniaxial symmetry ────
    @testset "closure_from_M_diag_3d: matches 2D version (M_xx = M_yy)" begin
        # When M_xx = M_yy, the 3D trace equals the 2D trace (M_zz + 2*M_rr)
        Nm = 2
        Sp = [0.6, 0.4]

        # Uniaxially symmetric M (columns: xx, yy, zz — xx = yy = M_rr)
        M3 = [0.20 0.20 0.60;   # tr = 1.00
              0.15 0.15 0.70]   # tr = 1.00

        # Equivalent 2D layout: columns = [M_zz, M_rr]
        M2 = hcat(M3[:,3], M3[:,1])   # [zz, rr]

        Z3 = closure_from_M_diag_3d(M3, :fenepme, Nm, Sp)
        Z2 = CoeffsRelax.closure_from_M_diag(M2, :fenepme, Nm, Sp)

        @test Z3.Y ≈ Z2.Y rtol=TOL
        @test Z3.E ≈ Z2.E rtol=TOL

        # fenepm: mean tr should also match
        Z3_pm = closure_from_M_diag_3d(M3, :fenepm, Nm, Sp)
        Z2_pm = CoeffsRelax.closure_from_M_diag(M2, :fenepm, Nm, Sp)
        @test Z3_pm.Xa[1] ≈ Z2_pm.Xa[1] rtol=TOL
    end

    # ── initial_M_diag: equilibrium ───────────────────────────────────────────
    @testset "initial_M_diag: :equilibrium" begin
        for Nm in [1, 3, 5]
            M = initial_M_diag(:equilibrium, 0.0, Nm, [-0.5, -0.5, 1.0])
            @test size(M) == (Nm, 3)
            @test all(M .≈ 1.0/3.0)
        end
    end

    # ── initial_M_diag: uniaxial_affine ──────────────────────────────────────
    @testset "initial_M_diag: :uniaxial_affine, alphas = [-0.5,-0.5,1]" begin
        eps_i  = 0.5
        Nm     = 2
        alphas = [-0.5, -0.5, 1.0]   # extensional axis = 3 (zz)
        M = initial_M_diag(:uniaxial_affine, eps_i, Nm, alphas)

        @test size(M) == (Nm, 3)
        # Column 3 (zz): (1/3)*exp(2*eps_i)
        @test M[1,3] ≈ (1.0/3.0)*exp(2.0*eps_i) rtol=TOL
        # Columns 1,2 (xx,yy): (1/3)*exp(-eps_i)
        @test M[1,1] ≈ (1.0/3.0)*exp(-eps_i)     rtol=TOL
        @test M[1,2] ≈ (1.0/3.0)*exp(-eps_i)     rtol=TOL
        # All modes are identical
        for p in 1:Nm
            @test M[p,:] ≈ M[1,:] atol=TOL
        end
        # Ratio: M_zz / M_xx = exp(3 eps_i)
        @test M[1,3] / M[1,1] ≈ exp(3.0*eps_i) rtol=TOL
    end

    @testset "initial_M_diag: :uniaxial_affine, eps_i = 0 → equilibrium" begin
        M = initial_M_diag(:uniaxial_affine, 0.0, 3, [-0.5, -0.5, 1.0])
        @test all(M .≈ 1.0/3.0)
    end

    @testset "initial_M_diag: ambiguous axis → error" begin
        @test_throws Exception initial_M_diag(:uniaxial_affine, 0.5, 2,
                                               [0.5, 0.5, -1.0])
        # Two equal maxima → must throw
        @test_throws Exception initial_M_diag(:uniaxial_affine, 0.5, 2,
                                               [1.0, 1.0, -2.0])
    end

    @testset "initial_M_diag: unknown kind → error" begin
        @test_throws Exception initial_M_diag(:bad_kind, 0.0, 1, [0.0, 0.0, 1.0])
    end

end  # MTensor tests

# ══════════════════════════════════════════════════════════════════════════════
@testset "Stress" begin

    # ── Equilibrium Hookean → zero stress ────────────────────────────────────
    @testset "polymer_stress: equilibrium (Hookean, any Nm)" begin
        for Nm in [1, 2, 4]
            M_eq = fill(1.0/3.0, Nm, 3)
            fp   = ones(Nm)
            tau  = polymer_stress(M_eq, fp)
            @test tau.xx ≈ 0.0 atol=1e-14
            @test tau.yy ≈ 0.0 atol=1e-14
            @test tau.zz ≈ 0.0 atol=1e-14
        end
    end

    # ── Scalar fp overload ────────────────────────────────────────────────────
    @testset "polymer_stress: scalar fp overload" begin
        Nm  = 3
        M   = fill(1.0/3.0, Nm, 3)
        tau_vec    = polymer_stress(M, ones(Nm))
        tau_scalar = polymer_stress(M, 1.0)
        @test tau_vec.xx == tau_scalar.xx
        @test tau_vec.yy == tau_scalar.yy
        @test tau_vec.zz == tau_scalar.zz
    end

    # ── Hookean Nm=1 at steady-state Wi=0.1 ──────────────────────────────────
    #
    # Analytic steady state for Nm=1, Hookean (fp=1), constant drag (Λ=1),
    # Rouse mode 1 (θ=1, σ=1/3), uniaxial flow alphas=[-0.5,-0.5,1]:
    #
    #   dM_pα/dt = 0  →  M_pα = σ / (θ - 2 Wi α_α)
    #   M_xx = M_yy = (1/3)/(1 + Wi)        [compressed axes]
    #   M_zz = (1/3)/(1 - 2 Wi)             [extensional axis]
    #
    #   τ_xx = τ_yy = -(3*M_xx - 1) =  Wi/(1+Wi)     > 0
    #   τ_zz         = -(3*M_zz - 1) = -2Wi/(1-2Wi)  < 0
    @testset "polymer_stress: Hookean Nm=1 steady-state Wi=0.1" begin
        Wi  = 0.1
        M_xx = (1.0/3.0) / (1.0 + Wi)
        M_zz = (1.0/3.0) / (1.0 - 2.0*Wi)
        M    = reshape([M_xx, M_xx, M_zz], 1, 3)
        fp   = [1.0]

        tau  = polymer_stress(M, fp)

        # Analytic values
        tau_xx_expected = Wi / (1.0 + Wi)
        tau_zz_expected = -2.0*Wi / (1.0 - 2.0*Wi)

        @test tau.xx ≈ tau_xx_expected rtol=TOL
        @test tau.yy ≈ tau_xx_expected rtol=TOL  # by uniaxial symmetry
        @test tau.zz ≈ tau_zz_expected rtol=TOL

        # Sign convention check: extension → τ_zz < 0, τ_xx > 0
        @test tau.zz < 0.0
        @test tau.xx > 0.0
    end

    # ── Multimode: check linearity in mode count ───────────────────────────────
    # For Nm modes all at the same M_row and same f: result = Nm * (single-mode result)
    @testset "polymer_stress: multimode linearity" begin
        M1 = reshape([0.25, 0.25, 0.50], 1, 3)
        tau1 = polymer_stress(M1, [1.0])

        Nm = 4
        M4 = repeat(M1, Nm, 1)
        tau4 = polymer_stress(M4, ones(Nm))

        @test tau4.xx ≈ Nm * tau1.xx atol=TOL
        @test tau4.yy ≈ Nm * tau1.yy atol=TOL
        @test tau4.zz ≈ Nm * tau1.zz atol=TOL
    end

end  # Stress tests

# ══════════════════════════════════════════════════════════════════════════════
const _ALPHAS_UNI = [-0.5, -0.5, 1.0]   # uniaxial, ext = z

@testset "StateObs" begin

    _alphas = _ALPHAS_UNI

    # ── Wi = 0 → NaN etaE, status :nan_wi ────────────────────────────────────
    @testset "make_obs: Wi = 0 → NaN, :nan_wi" begin
        tau = (xx=0.0, yy=0.0, zz=0.0)
        obs = make_obs(0.0, 1.0, tau, _alphas, 1.0, 1.0, 1.0)
        @test isnan(obs.etaE1)
        @test isnan(obs.etaE2)
        @test isnan(obs.Tr1)
        @test isnan(obs.Tr2)
        @test obs.status === :nan_wi
        @test obs.Wi  == 0.0
        @test obs.E2  == 1.0
    end

    # ── Hookean Nm=1 Wi=0.1 analytic ─────────────────────────────────────────
    #
    # Using the same steady-state stresses as in the Stress tests:
    #   τ_xx = τ_yy = Wi/(1+Wi),  τ_zz = -2Wi/(1-2Wi)
    #
    # etaE1 = -(τ_zz - τ_xx) / Wi
    #       = -(-2Wi/(1-2Wi) - Wi/(1+Wi)) / Wi
    #       = (2/(1-2Wi) + 1/(1+Wi))
    #       = 3 / ((1-2Wi)(1+Wi))           [Newtonian limit: → 3 as Wi→0]
    #
    # Tr1 = etaE1 * ckBT * lambda0 / (3 * etas)
    @testset "make_obs: Hookean Nm=1 Wi=0.1 analytic etaE and Tr" begin
        Wi  = 0.1
        lambda0 = 1.0; etas = 1.0; ckBT = 1.0

        tau_xx = Wi / (1.0 + Wi)
        tau_zz = -2.0*Wi / (1.0 - 2.0*Wi)
        tau    = (xx=tau_xx, yy=tau_xx, zz=tau_zz)

        obs = make_obs(Wi, 1.0, tau, _alphas, lambda0, etas, ckBT)

        etaE_expected = 3.0 / ((1.0 - 2.0*Wi) * (1.0 + Wi))   # = 75/22 ≈ 3.4091

        @test obs.etaE1 ≈ etaE_expected rtol=RTOL
        @test obs.etaE2 ≈ etaE_expected rtol=RTOL  # same by uniaxial symmetry
        @test obs.Tr1   ≈ etaE_expected / 3.0 rtol=RTOL
        @test obs.Tr2   ≈ etaE_expected / 3.0 rtol=RTOL
        @test obs.status === :ok
        @test obs.tau_xx ≈ tau_xx rtol=RTOL
        @test obs.tau_zz ≈ tau_zz rtol=RTOL
    end

    # ── Trouton limit: etaE → 3, Tr → 1 as Wi → 0 ───────────────────────────
    @testset "make_obs: Newtonian Trouton limit" begin
        Wi  = 1e-8
        tau_xx = Wi / (1.0 + Wi)
        tau_zz = -2.0*Wi / (1.0 - 2.0*Wi)
        tau    = (xx=tau_xx, yy=tau_xx, zz=tau_zz)

        obs = make_obs(Wi, 1.0, tau, _alphas, 1.0, 1.0, 1.0)
        @test obs.etaE1 ≈ 3.0 rtol=1e-5
        @test obs.Tr1   ≈ 1.0 rtol=1e-5
    end

    # ── Scales enter linearly ─────────────────────────────────────────────────
    @testset "make_obs: Trouton ratio scales with ckBT/etas" begin
        Wi  = 0.2
        tau = (xx=0.1, yy=0.1, zz=-0.2)
        obs1 = make_obs(Wi, 1.0, tau, _alphas, 1.0, 1.0, 2.0)  # ckBT=2
        obs2 = make_obs(Wi, 1.0, tau, _alphas, 1.0, 1.0, 1.0)  # ckBT=1
        @test obs1.Tr1 ≈ 2.0 * obs2.Tr1 rtol=TOL

        obs3 = make_obs(Wi, 1.0, tau, _alphas, 1.0, 0.5, 1.0)  # etas=0.5
        @test obs3.Tr1 ≈ 2.0 * obs2.Tr1 rtol=TOL
    end

    # ── Stress fields are stored verbatim ────────────────────────────────────
    @testset "make_obs: stores tau fields verbatim" begin
        tau = (xx=0.123, yy=0.456, zz=-0.789)
        obs = make_obs(0.5, 2.0, tau, _alphas, 1.0, 1.0, 1.0)
        @test obs.tau_xx == tau.xx
        @test obs.tau_yy == tau.yy
        @test obs.tau_zz == tau.zz
        @test obs.E2 == 2.0
        @test obs.Wi == 0.5
    end

end  # StateObs tests

# ══════════════════════════════════════════════════════════════════════════════
println("\n✓  All evals tests passed.")
