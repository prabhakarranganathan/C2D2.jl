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
include(joinpath(_MODELS, "kernel", "Params.jl"))

using .Fene
using .Drag
using .Spectrum
using .CoeffsRelax
using .Params

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

    # ── CU drag-interpolation scheme (2026-10-15, Aru→Ko handover) ───────────
    # Reference values: stand-alone Python (cu.py / smooth.py / compare.py) in
    # Aru's overlap-crossover note; NK = 5000, partialdraining, p_phi = 1.86.
    # Columns: h*K, E, phi, ζ/ζ_Z (RP), ζ/ζ_Z (CU).
    @testset "drag c2d2 CU scheme: RP/CU reference table" begin
        NK = 5000.0
        table = [
            (0.25,  1.0,  0.5,  1.000000,  1.139759),
            (0.25,  1.0,  1.0,  1.000000,  1.451590),
            (0.25,  1.0,  3.0,  3.007332,  3.210044),
            (0.25,  2.0,  0.1,  1.100697,  1.180349),
            (0.25,  8.0,  0.5,  3.030276,  3.088910),
            (0.25,  8.0,  8.0,  8.084718, 10.038675),
            (0.25,  8.0, 20.0, 20.840952, 22.117892),
            (0.25, 30.0,  0.01, 5.809616,  5.810402),
            (0.25, 30.0, 30.0, 32.449220, 40.128051),
            (0.01,  1.0,  0.5,  1.000000,  1.139759),
            (0.01,  1.0,  1.0,  1.000000,  1.451590),
            (0.01,  1.0,  3.0,  2.222668,  2.480391),
            (0.01,  2.0,  0.1,  0.935722,  1.003436),
            (0.01,  8.0,  0.5,  1.349312,  1.375421),
            (0.01,  8.0,  8.0,  3.599940,  4.469992),
            (0.01,  8.0, 20.0,  4.627897,  5.602053),
            (0.01, 30.0,  0.01, 0.880383,  0.880502),
            (0.01, 30.0, 30.0,  4.917321,  6.080963),
        ]
        for (hK, E, phi, z_rp, z_cu) in table
            zr, _ = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining)
            zc, d = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining;
                                    drag_interp=:cu, p_phi=1.86)
            @test zr ≈ z_rp rtol=1e-5
            @test zc ≈ z_cu rtol=1e-5
            # default kwargs == explicit :rp (bit-identical)
            zr2, _ = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:rp)
            @test zr2 == zr
            # diagnostics keep the RP regime definitions; G is the effective value
            @test d.blob_type == (E > phi ? :T : :C)
            @test d.G ≈ d.blob_drag_ratio * d.n_blob / zc rtol=1e-12
        end
        # :constant and :c1d2 ignore drag_interp
        for dm in (:constant, :c1d2)
            a, _ = drag_ratio_zimm(3.0, 2.0, 0.25, NK, dm, :partialdraining)
            b, _ = drag_ratio_zimm(3.0, 2.0, 0.25, NK, dm, :partialdraining; drag_interp=:cu)
            @test a == b
        end
    end

    @testset "drag c2d2 CU scheme: Λ = 1 at rest" begin
        NK = 5000.0; hK = 0.25
        for phi in (0.01, 0.5, 1.0, 3.0, 30.0)
            z1, _ = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu)
            spec  = build_spectrum_assessor(spectrum_model=:blob, Nm=1, NK=NK, hK_star=hK,
                                            phi=phi, drag_model=:c2d2,
                                            draining_model=:partialdraining,
                                            drag_interp=:cu, p_phi=1.86)
            @test z1 / spec.zeta0_ratio ≈ 1.0 rtol=1e-14
            _, Lambda = eval_spectrum(spec, 1.0, z1)
            @test Lambda ≈ 1.0 rtol=1e-14
            # equilibrium crossover: (1 + (φ a₀)^p)^(1/p), a₀ = α(NK/max(φ,1)²)/α_Z
            a0 = drag_draining_ratio(hK, NK / max(phi, 1.0)^2) / drag_draining_ratio(hK, NK)
            @test z1 ≈ (1 + (phi*a0)^1.86)^(1/1.86) rtol=1e-12
        end
    end

    @testset "drag c2d2 CU scheme: smooth across φ = E" begin
        # max jump in d ln ζ / d ln φ along φ at fixed E (h*K = 0.25), same grid as smooth.py
        NK = 5000.0; hK = 0.25
        lphi = range(log(1e-2), log(70.0); length=4000)
        for E in (3.0, 8.4, 24.0)
            lz = [log(drag_ratio_zimm(E, exp(lp), hK, NK, :c2d2, :partialdraining;
                                      drag_interp=:cu)[1]) for lp in lphi]
            slope = (lz[3:end] .- lz[1:end-2]) ./ (lphi[3:end] .- lphi[1:end-2])
            jump  = maximum(abs.(diff(slope)))
            # measured 2026-10-15: 0.0027 (E=3), 0.0060 (E=8.4), 0.0300 (E=24); RP: 3.2–5.1
            @test jump < 0.035
            # the RP scheme has the kink there (sanity: the test discriminates)
            lz_rp = [log(drag_ratio_zimm(E, exp(lp), hK, NK, :c2d2, :partialdraining)[1])
                     for lp in lphi]
            slope_rp = (lz_rp[3:end] .- lz_rp[1:end-2]) ./ (lphi[3:end] .- lphi[1:end-2])
            @test maximum(abs.(diff(slope_rp))) > 1.0
        end
    end

    @testset "drag c2d2 CU scheme: no dip below rest" begin
        NK = 5000.0; hK = 0.25
        Es = range(1.0, 8.0; length=1401)
        for phi in (0.001, 0.1, 0.5, 0.9, 2.0, 4.0)
            z1 = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu)[1]
            ratio = [drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining;
                                     drag_interp=:cu)[1] / z1 for E in Es]
            @test minimum(ratio) >= 1.0 - 1e-12
        end
    end

    @testset "drag c2d2 CU scheme: assessor has no kink split" begin
        NK = 5000.0; hK = 0.25; phi = 3.0
        a_rp = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining)
        a_cu = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu)
        @test a_rp.kinks == [phi]
        @test isempty(a_cu.kinks) && length(a_cu.xs) == 1
        @test a_cu.drag_interp === :cu && a_cu.p_phi == 1.86
        @test a_rp.drag_interp === :rp
        for E in [1.0, 1.5, 2.9, 3.0, 3.1, 5.0, 10.0, 30.0, 60.0]
            z_direct, _ = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu)
            @test eval_drag(a_cu, E) ≈ z_direct rtol=1e-4
        end
    end

    # ── U_1D relaxation factor g(s) (2026-10-17, Aru→Ko handover) ────────────
    @testset "CU g(s): definition, bit-identity when off, bare-ζ diagnostics" begin
        NK = 5000.0; hK = 0.25; p = 1.86; g0 = 0.44
        @test g1d_factor(0.0, g0, p) == 1.0
        @test g1d_factor(1e8, g0, p) ≈ g0 rtol=1e-7
        @test g1d_factor(3.0, 1.0, p) == 1.0
        @test g1d_factor(1.0, g0, p) ≈ g0 + (1-g0)/2^(1/p)
        @test parse_s_form("sum") === :sum && parse_s_form(" SUM ") === :sum
        @test parse_s_form("sum_hard") === :sum_hard
        @test_throws ErrorException parse_s_form("prod")
        @test DEFAULT_G0_1D == 1.0 && DEFAULT_S_FORM === :sum
        # model-anchored λ₀/λ_Z helper
        @test cu_lambda0_ratio(0.0, p) == 1.0
        @test cu_lambda0_ratio(1.0, p) ≈ 2^(1/p)
        @test cu_lambda0_ratio(100.0, p) ≈ 100.0 rtol=1e-3

        cells = ((1.0,0.5),(2.0,0.1),(8.0,8.0),(30.0,3.0),(30.0,30.0),(70.0,0.0),(1.0,0.0))
        for (E, phi) in cells
            # 1. g off (default or explicit) ≡ previous CU branch, exactly; RP ignores g
            z_old, d_old = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p)
            z_new, d_new = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p,
                                           g0_1D=1.0, s_form=:sum)
            @test z_new == z_old
            @test d_new.g1d == 1.0 && d_new.G == d_old.G
            zr0, dr0 = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining)
            zr1, _   = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; g0_1D=g0)
            @test zr1 == zr0 && dr0.g1d == 1.0 && isnan(dr0.s1d)

            # g on: ζ_eff = g(s)·ζ_bare with s = N_c + N_t − 2; diagnostics use the bare ζ.
            # :sum (default) → smooth N_c′ = min((1+φ^p)^(2/p), N_K); :sum_hard → clamp(φ²,1,N_K)
            Nt = clamp(E^2, 1.0, NK)
            for (form, Nc) in ((:sum, min((1 + phi^p)^(2/p), NK)), (:sum_hard, clamp(phi^2, 1.0, NK)))
                z_g, d_g = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p,
                                           g0_1D=g0, s_form=form)
                s = Nc + Nt - 2.0
                @test d_g.s1d ≈ s rtol=1e-14
                @test d_g.g1d ≈ g1d_factor(s, g0, p) rtol=1e-14
                @test z_g ≈ d_g.g1d * z_old rtol=1e-14
                @test d_g.G == d_old.G && d_g.n_blob == d_old.n_blob && d_g.blob_type == d_old.blob_type
                @test z_g <= z_old
            end
            # smooth N_c′ ≥ hard N_c (smaller g), equal only at φ = 0 and above the N_K cap
            z_s, d_s = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0)
            z_h, d_h = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0, s_form=:sum_hard)
            @test d_s.s1d >= d_h.s1d && z_s <= z_h
            phi == 0.0 && @test z_s == z_h
        end
        # :constant / :c1d2 ignore g
        for dm in (:constant, :c1d2)
            a, _ = drag_ratio_zimm(3.0, 2.0, hK, NK, dm, :partialdraining)
            b, _ = drag_ratio_zimm(3.0, 2.0, hK, NK, dm, :partialdraining; drag_interp=:cu, g0_1D=g0)
            @test a == b
        end
        @test_throws ErrorException build_drag_assessor(0.5, hK, NK, :c2d2, :partialdraining;
                                                        drag_interp=:cu, g0_1D=0.0)
    end

    @testset "CU g(s): dilute coil at rest unchanged; Λ = 1 at rest for all φ" begin
        NK = 5000.0; hK = 0.25; p = 1.86; g0 = 0.44
        # 3 (sum_hard, the 4a9e141 form): φ ≤ 1, E = 1 → N_c = N_t = 1 → s₀ = 0 → g = 1 → ζ₀,eff == ζ₀
        for phi in (0.0, 0.3, 1.0)
            zb, _ = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p)
            zg, d = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0, s_form=:sum_hard)
            @test zg == zb && d.s1d == 0.0 && d.g1d == 1.0
        end
        zb, _ = drag_ratio_zimm(1.0, 3.0, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p)
        zg, d = drag_ratio_zimm(1.0, 3.0, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0, s_form=:sum_hard)
        @test d.s1d == 8.0 && zg < zb
        @test zg ≈ g1d_factor(8.0, g0, p) * zb rtol=1e-14
        # 3 (sum, smooth N_c′): exact only at φ = 0; s₀ = (1+φ^p)^(2/p) − 1 is smooth through φ = 1
        zb0, _ = drag_ratio_zimm(1.0, 0.0, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p)
        zg0, d0 = drag_ratio_zimm(1.0, 0.0, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0)
        @test zg0 == zb0 && d0.s1d == 0.0 && d0.g1d == 1.0
        for phi in (0.3, 1.0, 3.0)
            _, d = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0)
            @test d.s1d ≈ (1 + phi^p)^(2/p) - 1 rtol=1e-14
        end
        s0 = [drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0)[2].s1d
              for phi in 0.9:0.01:1.1]
        @test all(diff(s0) .> 0) && maximum(abs.(diff(diff(s0)))) < 1e-3     # no corner at φ = 1

        # 2. Λ = 1 at rest for all φ with g on (through assessor, spectrum and LUT)
        for phi in (0.01, 0.5, 1.0, 3.0, 30.0)
            spec = build_spectrum_assessor(spectrum_model=:blob, Nm=3, NK=NK, hK_star=hK, phi=phi,
                                           drag_model=:c2d2, draining_model=:partialdraining,
                                           drag_interp=:cu, p_phi=p, g0_1D=g0, s_form=:sum)
            z1, _ = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0)
            @test z1 / spec.zeta0_ratio ≈ 1.0 rtol=1e-14
            _, Lambda = eval_spectrum(spec, 1.0, z1)
            @test Lambda ≈ 1.0 rtol=1e-14
            a = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining;
                                    drag_interp=:cu, p_phi=p, g0_1D=g0, s_form=:sum, n_points=4000)
            @test a.g0_1D == g0 && a.s_form === :sum
            @test eval_drag(a, 1.0) ≈ z1 rtol=1e-10
            z5, _ = drag_ratio_zimm(5.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p, g0_1D=g0)
            @test eval_drag(a, 5.0) ≈ z5 rtol=1e-5
            # Λ = [g(s)/g(s₀)]·ζ(E)/ζ₀ through the LUT
            zb5, _ = drag_ratio_zimm(5.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p)
            zb1, _ = drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:cu, p_phi=p)
            Nc = min((1 + phi^p)^(2/p), NK)                       # smooth N_c′ (default :sum)
            Lam_expected = g1d_factor(Nc + 25.0 - 2.0, g0, p) / g1d_factor(Nc - 1.0, g0, p) * zb5 / zb1
            _, Lam5 = eval_spectrum(spec, 5.0, z5)
            @test Lam5 ≈ Lam_expected rtol=1e-12
            # blob spectrum shape independent of g (G_eff built from the bare ζ)
            spec_b = build_spectrum_assessor(spectrum_model=:blob, Nm=3, NK=NK, hK_star=hK, phi=phi,
                                             drag_model=:c2d2, draining_model=:partialdraining,
                                             drag_interp=:cu, p_phi=p)
            lh_b, _ = eval_spectrum(spec_b, 5.0, zb5)
            lh_g, _ = eval_spectrum(spec,   5.0, z5)
            @test lh_b == lh_g
        end
    end

    @testset "C_phi in the stress prefactor; C2D2Params back-compat constructors" begin
        base = (3, 5000.0, 0.25, 3.0, 0.3, :fenepme, :c2d2, :partialdraining, :blob,
                1.0, 0.5, 0.0, 50, 2.0, true, 1.0, 1e-5, 1e-13, 1e-3, 1e-6, 1e-9,
                0.05, "output_test", "bhat")
        p24 = C2D2Params(base...)                               # pre-2026-10-15 arity
        p26 = C2D2Params(base..., :cu, 1.86)                    # 2026-10-15 arity
        @test p24.drag_interp === :rp && p24.g0_1D == 1.0 && p24.s_form === :sum && p24.C_phi == 1.0
        @test p26.drag_interp === :cu && p26.g0_1D == 1.0 && p26.C_phi == 1.0
        pC = C2D2Params(base..., :cu, 1.86, 1.0,  :sum_hard, 1.0)    # g off, C_phi = 1  (≡ p26)
        pA = C2D2Params(base..., :cu, 1.86, 0.44, :sum_hard, 1.0)    # g on (hard N_c: s₀ = φ² − 1 = 8)
        pB = C2D2Params(base..., :cu, 1.86, 0.44, :sum_hard, 0.5)    # g on, C_phi = 0.5
        c26, cC, cA, cB = build_runtime_ctx.((p26, pC, pA, pB))
        # 1. defaults bit-identical
        @test cC.poly_prefactor == c26.poly_prefactor && cC.zeta0_ratio == c26.zeta0_ratio
        # 4. C_phi = 0.5 doubles poly_prefactor, nothing else changes
        @test cB.poly_prefactor ≈ 2 * cA.poly_prefactor rtol=1e-15
        @test cB.zeta0_ratio == cA.zeta0_ratio && cB.U_zimm == cA.U_zimm
        @test eval_drag(cB.kernel.drag_lut, 4.0) == eval_drag(cA.kernel.drag_lut, 4.0)
        # g(s₀) enters the prefactor through ζ₀ (φ = 3 → s₀ = φ² − 1 = 8)
        @test cA.zeta0_ratio ≈ g1d_factor(8.0, 0.44, 1.86) * cC.zeta0_ratio rtol=1e-14
        @test cA.poly_prefactor ≈ g1d_factor(8.0, 0.44, 1.86) * cC.poly_prefactor rtol=1e-14
        @test cA.poly_prefactor < 0

        # 29-arg constructor (2026-10-17 am arity) fills De_mode = :input, De_Z = NaN
        @test pA.De_mode === :input && isnan(pA.De_Z)
        c29 = build_runtime_ctx(C2D2Params(base..., :cu, 1.86, 0.44, :sum_hard, 1.0, :input, NaN))
        @test c29.params.q_oes == 0.0                                      # 31-field arity fills q_oes = 0
        p32 = C2D2Params(base..., :oescui, 2.0, 1.0, :sum, 1.0, :input, NaN, 0.25)
        @test p32.q_oes == 0.25 && build_runtime_ctx(p32).kernel.drag_lut.q_oes == 0.25 &&
              build_runtime_ctx(p32).kernel.spectrum.q_oes == 0.25
        @test c29.poly_prefactor == cA.poly_prefactor && c29.lambda0_ratio == cA.zeta0_ratio

        # De_mode = :cu_fit: De = De_Z (1+φ^p)^(1/p); prefactor = −φ U/(C_phi De_Z), independent of h*_K and g0
        phi = base[4]; p = 1.86; De_Z = 0.2
        De0 = De_Z * (1 + phi^p)^(1/p)
        mk(hK, g0, Cphi) = C2D2Params(3, 5000.0, hK, phi, 0.3, :fenepme, :c2d2, :partialdraining, :blob,
                                      1.0, De0, 0.0, 50, 2.0, true, 1.0, 1e-5, 1e-13, 1e-3, 1e-6, 1e-9,
                                      0.05, "output_test", "bhat", :cu, p, g0, :sum, Cphi, :cu_fit, De_Z)
        cF1, cF2, cF3 = build_runtime_ctx.((mk(0.25, 0.5, 1.0), mk(0.025, 1.0, 1.0), mk(0.25, 0.5, 0.5)))
        @test cF1.params.De ≈ De_Z * cu_lambda0_ratio(phi, p) rtol=1e-15
        @test cF1.lambda0_ratio ≈ (1 + phi^p)^(1/p) rtol=1e-14
        @test cF1.poly_prefactor ≈ -(phi * cF1.U_zimm) / De_Z rtol=1e-15
        @test cF2.poly_prefactor ≈ -(phi * cF2.U_zimm) / De_Z rtol=1e-15     # g0 = 1: same prefactor
        @test cF1.zeta0_ratio != cF2.zeta0_ratio                             # … although ζ₀ differs (g, h*_K)
        @test cF3.poly_prefactor ≈ 2 * cF1.poly_prefactor rtol=1e-15
        # De_Z given in :input mode: De as given, prefactor tied to De_Z
        cI = build_runtime_ctx(C2D2Params(base..., :cu, p, 0.5, :sum, 1.0, :input, De_Z))
        @test cI.params.De == base[11]
        @test cI.poly_prefactor ≈ -(phi * cI.U_zimm) / De_Z rtol=1e-15
        @test cI.lambda0_ratio == base[11] / De_Z
    end

    # ── v2 family: OESCUI / OESI / BARECUI / BAREI (2026-10-21/22, Aru→Ko handovers) ──
    # Reference: Aru's oesi.py:zeta_v2 (copy in docs/validation/2026-10-22_oesi_v2/), full
    # double precision; partialdraining, p_phi = 2.0; D(A) shape factor.
    A_INF = 3π / 1.6366
    V2_REF   = joinpath(@__DIR__, "reference", "oesi_v2_reference_values.csv")
    OESI_REF = joinpath(@__DIR__, "reference", "oesi_reference_values.csv")   # 21-Oct table; RPI column still valid
    V2 = (:barei, :barecui, :oesi, :oescui)
    zv(E, phi, hK, NK, di; p=2.0, dm=:partialdraining) =
        drag_ratio_zimm(E, phi, hK, NK, :c2d2, dm; drag_interp=di, p_phi=p)

    @testset "v2: reference table (340 rows × 4 schemes, rtol 1e-10)" begin
        lines = readlines(V2_REF)
        @test split(lines[1], ",") == ["hK_star", "NK", "phi", "E", "regime",
                                       "zeta_bare", "zeta_bare_cui", "zeta_oes", "zeta_oes_cui"]
        @test length(lines) == 341
        for l in lines[2:end]
            f  = split(l, ",")
            hK = parse(Float64, f[1]); NK = parse(Float64, f[2])
            phi = parse(Float64, f[3]); E = parse(Float64, f[4])
            for (di, k) in ((:barei, 6), (:barecui, 7), (:oesi, 8), (:oescui, 9))
                z, d = zv(E, phi, hK, NK, di)
                @test z ≈ parse(Float64, f[k]) rtol=1e-10
                # regime column: dilute/weak → :T, strong → :C
                @test d.blob_type == (f[5] == "strong" ? :C : :T)
                @test d.blob_type == (E > phi ? :T : :C)
                @test d.g1d == 1.0 && isnan(d.s1d)
                # G keeps the blob spectrum consistent: G = ζ_b·n_blob/ζ for every scheme
                @test d.G ≈ d.blob_drag_ratio * d.n_blob / z rtol=1e-12
                # switched schemes: G = D(A) in the tension regime, 1 in the correlation regime
                if di in (:barei, :oesi)
                    d.blob_type == :C && @test d.G == 1.0
                    d.blob_type == :T && @test d.G ≈ oesi_shape_D(min(E^2, phi > 0 ? sqrt((1 + log(E / phi)) * E / phi) : Inf)) rtol=1e-12
                end
            end
            # the scheme default p_phi is 2.0, so the explicit value is redundant
            @test drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:oescui)[1] == zv(E, phi, hK, NK, :oescui)[1]
            @test drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:barecui)[1] == zv(E, phi, hK, NK, :barecui)[1]
        end
        # spot values of the 22-Oct handover (h*K = 0.025, NK = 5000; bare, bare+CU, OESI, OESCUI)
        for (E, phi, ref) in ((2.0, 0.0, (0.931934310668, 0.931934310668, 0.931934310668, 0.931934310668)),
                              (10.0, 1.0, (2.55991926212, 2.74830613808, 4.56729347679, 4.69300688369)),
                              (1.0, 1.0, (1.0, 1.41421356237, 1.07895500703, 1.52587280413)),
                              (3.0, 10.0, (6.04742769321, 6.58975917154, 10.7895500703, 11.2646209827)))
            for (di, r) in zip(V2, ref)
                @test zv(E, phi, 0.025, 5000.0, di)[1] ≈ r rtol=1e-11
            end
        end
        # :constant and :c1d2 ignore drag_interp
        for dm in (:constant, :c1d2), di in V2
            a, _ = drag_ratio_zimm(3.0, 2.0, 0.25, 5000.0, dm, :partialdraining)
            b, _ = drag_ratio_zimm(3.0, 2.0, 0.25, 5000.0, dm, :partialdraining; drag_interp=di)
            @test a == b
        end
    end

    @testset "RPI column of the 21-Oct table (144 rows) still = RP" begin
        lines = readlines(OESI_REF)
        @test split(lines[1], ",")[[1, 2, 3, 4, 7]] == ["hK_star", "NK", "phi", "E", "zeta_ratio_RPI"]
        @test length(lines) == 145
        for l in lines[2:end]
            f = split(l, ",")
            hK = parse(Float64, f[1]); NK = parse(Float64, f[2]); phi = parse(Float64, f[3]); E = parse(Float64, f[4])
            zr = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining)[1]                    # RP
            @test zr ≈ parse(Float64, f[7]) rtol=1e-10
            @test drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=parse_drag_interp("RPI"))[1] == zr
        end
    end

    @testset "v2: D(A) monotone, continuity at E = φ, E_b(φ) and φ = 1" begin
        NK = 5000.0; E_max = sqrt(NK)
        # E/D(E²) non-decreasing on [1, 100]; D(1) = 1; C¹ at A = 4
        Es = exp.(range(0.0, log(100.0); length=20001))
        r  = [E / oesi_shape_D(E^2) for E in Es]
        @test all(diff(r) .>= -1e-14)
        @test oesi_shape_D(1.0) == 1.0 && oesi_shape_D(4.0) == 2.0
        @test r[findfirst(>=(2.0), Es) - 1] ≈ 1.0 rtol=1e-6          # E/D = 1 up to E = 2
        @test (oesi_shape_D(4.0 + 1e-7) - oesi_shape_D(4.0 - 1e-7)) / 2e-7 ≈ 0.25 rtol=1e-6
        # continuity across φ = E (jump < 1e-9 for a φ-perturbation of 1e-12, i.e. no step; the
        # slope in φ is O(1)) at E ∈ {2, 5, 20, 70}, all four schemes, three h*K
        for hK in (0.25, 0.04, 0.025), di in V2, E in (2.0, 5.0, 20.0, 70.0)
            zl = zv(E, E * (1 - 1e-12), hK, NK, di)[1]     # E > φ: weak
            zr = zv(E, E * (1 + 1e-12), hK, NK, di)[1]     # E ≤ φ: strong
            @test abs(zr - zl) / zl < 1e-9
            @test abs(zr - zv(E, E, hK, NK, di)[1]) / zl < 1e-9
            # with oes off the two arms coincide exactly at φ = E (A = 1, D = 1, N_t = N_c)
            if di in (:barei, :barecui)
                @test zv(E * (1 + 1e-12), E, hK, NK, :barei)[1] ≈ zv(E, E, hK, NK, :barei)[1] rtol=1e-10
            end
        end
        # in E at fixed φ: across E = φ, across E_b(φ), and across φ = 1 at fixed E
        for hK in (0.17, 0.025), di in V2
            for phi in (1e-3, 0.1, 0.5, 2.0, 10.0)
                Eb = phi < 1 ? oesi_dilute_weak_E(phi, E_max) : phi
                @test isfinite(Eb) && 1.0 < Eb < E_max
                zl = zv(Eb * (1 - 1e-6), phi, hK, NK, di)[1]
                zr = zv(Eb * (1 + 1e-6), phi, hK, NK, di)[1]
                @test abs(zr - zl) / zl < 1e-4
            end
            for E in (1.0, 3.0, 30.0)
                zl = zv(E, 1 - 1e-6, hK, NK, di)[1]
                zr = zv(E, 1 + 1e-6, hK, NK, di)[1]
                @test abs(zr - zl) / zl < 1e-4
            end
        end
        # the dilute/weak boundary: φE³ = 1 + ln(E/φ), E > 1
        for phi in (1e-3, 0.01, 0.1, 0.5, 0.99)
            Eb = oesi_dilute_weak_E(phi, E_max)
            @test phi * Eb^3 ≈ 1 + log(Eb / phi) rtol=1e-10
            @test Eb^2 ≈ sqrt((1 + log(Eb / phi)) / (phi / Eb)) rtol=1e-10
        end
        @test isnan(oesi_dilute_weak_E(0.0, E_max)) && isnan(oesi_dilute_weak_E(1.0, E_max))
        @test isnan(oesi_dilute_weak_E(1e-9, E_max))      # dilute up to full stretch
    end

    @testset "v2: equilibrium and non-draining limits" begin
        NK = 5000.0; E_max = sqrt(NK)
        for hK in (0.25, 0.17, 0.025)
            aZ = drag_draining_ratio(hK, NK); r = A_INF / aZ
            # OESI at rest: 1 + (α_∞/α_Z − 1)φ (φ ≤ 1), (α_∞/α_Z)φ (1 < φ ≤ √NK)
            for phi in (0.0, 0.3, 0.7, 1.0)
                @test zv(1.0, phi, hK, NK, :oesi)[1] ≈ 1 + (r - 1) * phi rtol=1e-12
            end
            for phi in (1.5, 3.0, 10.0, E_max)
                @test zv(1.0, phi, hK, NK, :oesi)[1] ≈ r * phi rtol=1e-12
            end
            # BAREI at rest: the original blob model — 1 for φ ≤ 1, φ·α(h*K, NK/φ²)/α_Z above
            for phi in (0.0, 0.3, 1.0)
                @test zv(1.0, phi, hK, NK, :barei)[1] == 1.0
            end
            for phi in (1.5, 3.0, 10.0, 50.0)
                @test zv(1.0, phi, hK, NK, :barei)[1] ≈ phi * drag_draining_ratio(hK, NK / phi^2) / aZ rtol=1e-12
            end
            @test zv(1.0, E_max, hK, NK, :barei)[1] ≈ E_max * drag_draining_ratio(hK, 1.0) / aZ rtol=1e-12
            # CUI rest values at φ = 1: 2^(1/p)·(ζ_w = ζ_s)
            @test zv(1.0, 1.0, hK, NK, :oescui)[1] ≈ 2^(1/2) * r rtol=1e-12
            @test zv(1.0, 1.0, hK, NK, :barecui)[1] ≈ 2^(1/2) rtol=1e-12
            # strongly screened switched schemes: independent of E
            for E in (1.0, 2.0, 9.9), phi in (10.0, 100.0)
                @test zv(E, phi, hK, NK, :oesi)[1] ≈ r * min(phi, E_max) rtol=1e-12
                nC = min(phi^2, NK)
                @test zv(E, phi, hK, NK, :barei)[1] ≈ min(phi, E_max) * drag_draining_ratio(hK, NK / nC) / aZ rtol=1e-12
            end
            # Λ = 1 at rest through the spectrum assessor (scheme default p_phi both sides)
            for di in V2
                spec = build_spectrum_assessor(spectrum_model=:blob, Nm=1, NK=NK, hK_star=hK, phi=0.3,
                                               drag_model=:c2d2, draining_model=:partialdraining,
                                               drag_interp=di)
                z1 = drag_ratio_zimm(1.0, 0.3, hK, NK, :c2d2, :partialdraining; drag_interp=di)[1]
                @test z1 / spec.zeta0_ratio == 1.0
                @test spec.p_phi == default_p_phi(di)
            end
            # non-draining: α ≡ α_∞, oes is a no-op; ζ₀/ζ_Z = max(1, φ) exactly for the switched schemes
            for phi in (0.0, 0.3, 1.0, 3.0, 30.0)
                @test zv(1.0, phi, hK, NK, :oesi; dm=:nondraining)[1] == max(1.0, phi)
                @test zv(1.0, phi, hK, NK, :barei; dm=:nondraining)[1] == max(1.0, phi)
                for E in (1.0, 4.0, 40.0)
                    @test zv(E, phi, hK, NK, :oesi; dm=:nondraining)[1] == zv(E, phi, hK, NK, :barei; dm=:nondraining)[1]
                    @test zv(E, phi, hK, NK, :oescui; dm=:nondraining)[1] == zv(E, phi, hK, NK, :barecui; dm=:nondraining)[1]
                end
            end
        end
    end

    @testset "CUI schemes: Rouse cap, p_phi, dilute limit" begin
        NK = 5000.0; E_max = sqrt(NK)
        for hK in (0.17, 0.025)
            aZ = drag_draining_ratio(hK, NK); r = A_INF / aZ; r1 = drag_draining_ratio(hK, 1.0) / aZ
            # every E gives the Rouse drag at φ = √NK: (α_∞/α_Z)√NK (OESCUI), (α(h*K,1)/α_Z)√NK (BARECUI)
            for E in (1.0, 3.0, 30.0, E_max)
                @test zv(E, E_max, hK, NK, :oescui)[1] ≈ r * E_max rtol=1e-12
                @test zv(E, E_max, hK, NK, :barecui)[1] ≈ r1 * E_max rtol=1e-12
            end
            # once the cap onset E_cap(φ) lies below √NK the fully stretched chain sits on the cap,
            # while the coil (E = 1) never does for φ < √NK
            for (di, cap) in ((:oescui, r * E_max), (:barecui, r1 * E_max)), phi in (30.0, 45.0, 60.0)
                E_cap = Drag._v2_cap_E(phi, hK, NK, :partialdraining, di === :oescui, 2.0)
                isfinite(E_cap) && @test zv(E_max, phi, hK, NK, di)[1] ≈ cap rtol=1e-12
                isfinite(E_cap) && @test zv(E_cap * 0.99, phi, hK, NK, di)[1] < cap * (1 - 1e-8)
                @test zv(1.0, phi, hK, NK, di)[1] < cap
            end
            @test isfinite(Drag._v2_cap_E(60.0, 0.025, NK, :partialdraining, true, 2.0))
            # dilute limit unchanged by the blend
            for E in (1.0, 3.0, 30.0)
                @test zv(E, 0.0, hK, NK, :oescui)[1] == zv(E, 0.0, hK, NK, :oesi)[1]
                @test zv(E, 0.0, hK, NK, :barecui)[1] == zv(E, 0.0, hK, NK, :barei)[1]
                @test zv(E, 0.0, hK, NK, :barei)[1] == zv(E, 0.0, hK, NK, :oesi)[1]     # no overlap, no OES
            end
            # blend ≥ both arms, ≤ cap
            for E in (1.0, 2.0, 7.0, 30.0), phi in (0.1, 1.0, 5.0, 40.0)
                zo = zv(E, phi, hK, NK, :oesi)[1]; zc = zv(E, phi, hK, NK, :oescui)[1]
                @test zc >= zo * (1 - 1e-12) && zc <= r * E_max * (1 + 1e-12)
                zb = zv(E, phi, hK, NK, :barei)[1]; zbc = zv(E, phi, hK, NK, :barecui)[1]
                @test zbc >= zb * (1 - 1e-12) && zbc <= r1 * E_max * (1 + 1e-12)
            end
        end
        # p_phi is consulted (p → ∞ gives max(ζ_w, ζ_s)); defaults
        hK = 0.025; aZ = drag_draining_ratio(hK, NK); r = A_INF / aZ
        @test zv(1.0, 1.0, hK, NK, :oescui; p=400.0)[1] ≈ r rtol=1e-2
        @test zv(1.0, 1.0, hK, NK, :barecui; p=400.0)[1] ≈ 1.0 rtol=1e-2
        @test zv(1.0, 1.0, hK, NK, :oescui; p=1.86)[1] ≈ 1.566200717 rtol=1e-9     # the 21-Oct value
        @test DEFAULT_P_PHI == 1.86 && DEFAULT_P_PHI_OESCUI == 2.0
        @test default_p_phi(:oescui) == 2.0 && default_p_phi(:barecui) == 2.0
        @test default_p_phi(:cu) == 1.86 && default_p_phi(:rp) == 1.86 && default_p_phi(:oesi) == 1.86 && default_p_phi(:barei) == 1.86
        for di in (:oescui, :barecui)
            @test_throws ErrorException build_drag_assessor(1.0, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=0.0)
            @test_throws ErrorException build_drag_assessor(1.0, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=NaN)
            @test build_drag_assessor(1.0, hK, NK, :c2d2, :partialdraining; drag_interp=di).p_phi == 2.0
        end
    end

    @testset "v2: LUT assessor vs direct" begin
        NK = 5000.0; hK = 0.025; E_max = sqrt(NK)
        for di in V2, phi in (0.0, 0.01, 0.3, 1.0, 3.0, 10.0, 30.0, 45.0, 60.0)
            a = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining; drag_interp=di)
            @test a.drag_interp === di
            cui = di in (:oescui, :barecui)
            E_cap = cui ? Drag._v2_cap_E(phi, hK, NK, :partialdraining, di === :oescui, 2.0) : NaN
            if phi == 0.0 || phi == 1.0 || phi >= E_max
                @test isempty(a.kinks)
            elseif phi > 1.0
                if !isfinite(E_cap)                      # no cap below E_max: one corner at E = φ
                    @test a.kinks == [phi]
                elseif E_cap < phi                       # capped before E = φ: the cap is the only corner
                    @test a.kinks == [E_cap]
                else                                     # both corners: E = φ then the cap onset
                    @test a.kinks == [phi, E_cap]
                end
            else
                @test length(a.kinks) == 1 && a.kinks[1] ≈ oesi_dilute_weak_E(phi, E_max)
            end
            # every corner is an exact grid node of the pieces on both sides; pieces tile [1, E_max]
            @test length(a.xs) == length(a.kinks) + 1 == length(a.ys)
            @test issorted(a.kinks) && a.xs[1][1] == 0.0 && a.xs[end][end] ≈ log10(E_max)
            for (i, k) in enumerate(a.kinks)
                @test a.xs[i][end] == log10(k) == a.xs[i+1][1]
            end
            # dense sweep + corner neighbourhoods, default 400-pt LUT, rtol 1e-4 as for CU/RPDP
            ks = a.kinks
            Es = vcat(exp.(range(0.0, log(E_max); length=600)),
                      [k * m for k in ks for m in (0.98, 0.999, 1.0, 1.001, 1.02)])
            worst = maximum(abs(eval_drag(a, E) - zv(E, phi, hK, NK, di)[1]) / zv(E, phi, hK, NK, di)[1] for E in Es)
            @test worst < 1e-4
            # derivative is finite everywhere and the three-piece evaluator is continuous at the corners
            for k in ks
                @test isfinite(eval_drag_deriv(a, k))
                @test eval_drag(a, k * (1 - 1e-9)) ≈ eval_drag(a, k * (1 + 1e-9)) rtol=1e-6
            end
        end
        # Rouse-cap corners (N_K = 5000, h*K = 0.025, p = 2): OESCUI φ = 45 has E_cap > φ (two
        # corners), φ = 60 has E_cap < φ (one); BARECUI's lower cap engages far earlier (φ = 30: E_cap ≈ 8.4)
        for (di, phi, two) in ((:oescui, 45.0, true), (:oescui, 60.0, false), (:barecui, 30.0, false), (:barecui, 45.0, false))
            a = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining; drag_interp=di)
            cap = (di === :oescui ? A_INF : drag_draining_ratio(hK, 1.0)) / drag_draining_ratio(hK, NK) * E_max
            @test length(a.kinks) == (two ? 2 : 1)
            Ec = a.kinks[end]
            @test 1.0 < Ec < E_max
            two ? (@test Ec > phi) : (@test Ec < phi)
            @test eval_drag(a, Ec) ≈ cap rtol=1e-10
            @test eval_drag(a, Ec * 0.99) < cap * (1 - 1e-6)
            @test eval_drag(a, Ec * 1.5) ≈ cap rtol=1e-10
        end
        @test isnan(Drag._v2_cap_E(3.0, hK, NK, :partialdraining, true, 2.0))       # cap not reached
        @test isnan(Drag._v2_cap_E(E_max, hK, NK, :partialdraining, true, 2.0))     # capped from E = 1
    end

    # ── q_oes: stretch-delayed OES dressing (2026-10-22 pm, Aru→Ko handover) ──
    # ψ = min(φ/E^q, 1) in OESI/OESCUI; default 0 is bit-identical (checked by the dump of
    # acceptance item 5, not here).  Reference: oesi.py:zeta_v2(..., q=0.25) columns of
    # oesi_v2q_reference_values.csv.
    V2Q_REF = joinpath(@__DIR__, "reference", "oesi_v2q_reference_values.csv")
    @testset "q_oes: reference table, spot values, E = 1 invariance, continuity" begin
        @test DEFAULT_Q_OES == 0.0
        lines = readlines(V2Q_REF)
        hdr = split(lines[1], ",")
        @test hdr[1:9] == ["hK_star", "NK", "phi", "E", "regime", "zeta_bare", "zeta_bare_cui", "zeta_oes", "zeta_oes_cui"]
        @test hdr[10:11] == ["zeta_oes_q0.25", "zeta_oes_cui_q0.25"]
        @test length(lines) == 341
        for l in lines[2:end]
            f  = split(l, ",")
            hK = parse(Float64, f[1]); NK = parse(Float64, f[2]); phi = parse(Float64, f[3]); E = parse(Float64, f[4])
            for (di, k0, kq) in ((:oesi, 8, 10), (:oescui, 9, 11))
                zq = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=2.0, q_oes=0.25)[1]
                z0 = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=2.0, q_oes=0.0)[1]
                @test zq ≈ parse(Float64, f[kq]) rtol=1e-10
                @test z0 ≈ parse(Float64, f[k0]) rtol=1e-10
                # q = 0 is the kwarg default
                @test z0 == drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di)[1]
                # q lowers the dressing (ψ ≤ ψ₀): unchanged where ψ = 1 (φ ≥ E^q) or φ = 0, changed
                # otherwise (the sign depends on h*K: at h*K = 0.25 the KR fit for small blobs exceeds
                # α_∞, so less dressing *raises* ζ there)
                if phi == 0.0 || phi >= E^0.25
                    @test zq == z0
                elseif E > 1.0
                    @test zq != z0
                end
            end
            # bare schemes ignore q_oes
            for di in (:barei, :barecui)
                @test drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=0.25)[1] ==
                      drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di)[1]
            end
        end
        # spot values (OESI, OESCUI), q = 0.25, p = 2
        for (E, phi, hK, NK, ro, rc) in ((10.0, 1.0, 0.025, 5000.0, 3.68874873815, 3.84330732058),
                                         (10.0, 3.0, 0.025, 5000.0, 6.55363259489, 7.3094045827),
                                         (sqrt(5000.0), 1.0, 0.025, 5000.0, 9.39808187991, 9.45981431787),
                                         (3.0, 2.0, 0.017, 11186.0, 2.6907799746, 3.45146113296))
            @test drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:oesi, p_phi=2.0, q_oes=0.25)[1] ≈ ro rtol=1e-11
            @test drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=:oescui, p_phi=2.0, q_oes=0.25)[1] ≈ rc rtol=1e-11
        end
        # at E = 1 the drag equals the q = 0 value exactly, for all φ (E^q = 1)
        NK = 5000.0; E_max = sqrt(NK)
        for hK in (0.25, 0.025), di in (:oesi, :oescui), phi in (0.0, 0.3, 1.0, 1.5, 2.0, 10.0, E_max, 200.0)
            @test drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=0.25)[1] ==
                  drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=0.0)[1]
            # so p_phi = 2 at rest and Λ = 1 at rest stand
            spec = build_spectrum_assessor(spectrum_model=:blob, Nm=1, NK=NK, hK_star=hK, phi=phi,
                                           drag_model=:c2d2, draining_model=:partialdraining,
                                           drag_interp=di, q_oes=0.25)
            @test spec.q_oes == 0.25
            @test drag_ratio_zimm(1.0, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=0.25)[1] / spec.zeta0_ratio == 1.0
        end
        # continuity across φ = E (1e-12 perturbation) at E ∈ {2, 5, 20, 70}, q ∈ {0.25, 1}
        for hK in (0.25, 0.04, 0.025), di in (:oesi, :oescui), q in (0.25, 1.0), E in (2.0, 5.0, 20.0, 70.0)
            zl = drag_ratio_zimm(E, E * (1 - 1e-12), hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=q)[1]
            zr = drag_ratio_zimm(E, E * (1 + 1e-12), hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=q)[1]
            @test abs(zr - zl) / zl < 1e-9
        end
        # dressing completes at φ = E^q: ψ = 1 for φ ≥ E^q, i.e. the OES drag equals its φ-independent value
        hK = 0.025; aZ = drag_draining_ratio(hK, NK)
        for E in (3.0, 10.0, 50.0), q in (0.25, 0.5)
            zfull = drag_ratio_zimm(E, E^q, hK, NK, :c2d2, :partialdraining; drag_interp=:oesi, q_oes=q)[1]
            # (same tension-blob prefactor α_∞ as at φ = E, only A differs: the undressed-α part is gone)
            d = drag_ratio_zimm(E, E^q, hK, NK, :c2d2, :partialdraining; drag_interp=:oesi, q_oes=q)[2]
            @test d.blob_drag_ratio ≈ (A_INF / E) / aZ rtol=1e-12
            # just below E^q the dressing is incomplete
            d2 = drag_ratio_zimm(E, E^q * (1 - 1e-6), hK, NK, :c2d2, :partialdraining; drag_interp=:oesi, q_oes=q)[2]
            @test d2.blob_drag_ratio < d.blob_drag_ratio
        end
        # validation: q ∉ [0, 1] rejected for the OES schemes, ignored by the others
        for di in (:oesi, :oescui)
            @test_throws ErrorException build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=-0.1)
            @test_throws ErrorException build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=1.5)
            @test_throws ErrorException build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=NaN)
            @test build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=1.0).q_oes == 1.0
        end
        @test build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=:barei, q_oes=7.0).q_oes == 7.0
        # E_q = φ^(1/q)
        @test oesi_dressing_E(2.0, 0.25, E_max) == 16.0
        @test isnan(oesi_dressing_E(2.0, 0.0, E_max)) && isnan(oesi_dressing_E(1.0, 0.25, E_max)) && isnan(oesi_dressing_E(3.0, 0.25, E_max))
        @test oesi_dressing_E(2.85, 0.25, E_max) ≈ 2.85^4 && isnan(oesi_dressing_E(2.9, 0.25, E_max))   # 5000^(1/8) ≈ 2.8997
    end

    @testset "q_oes: LUT corners (E = φ, E_q, cap) and accuracy" begin
        NK = 5000.0; hK = 0.025; E_max = sqrt(NK); q = 0.25
        zq(E, phi, di) = drag_ratio_zimm(E, phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, p_phi=2.0, q_oes=q)[1]
        for di in (:oesi, :oescui), phi in (0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 3.0, 10.0, 45.0, 60.0)
            a = build_drag_assessor(phi, hK, NK, :c2d2, :partialdraining; drag_interp=di, q_oes=q)
            @test a.q_oes == q
            E_q   = oesi_dressing_E(phi, q, E_max)
            E_cap = di === :oescui ? Drag._v2_cap_E(phi, hK, NK, :partialdraining, true, 2.0, q) : NaN
            if phi < 1.0
                @test length(a.kinks) == 1 && a.kinks[1] ≈ oesi_dilute_weak_E(phi, E_max)
            elseif phi == 1.0
                @test isempty(a.kinks)
            elseif isfinite(E_q)                         # 1 < φ < 5000^(1/8) ≈ 2.9: E = φ and E_q = φ⁴
                @test a.kinks == [phi, E_q]
            elseif isfinite(E_cap) && E_cap < phi        # φ = 60: cap only
                @test a.kinks == [E_cap]
            elseif isfinite(E_cap)                       # φ = 45 (OESCUI): E = φ and the cap
                @test a.kinks == [phi, E_cap]
            else
                @test a.kinks == [phi]
            end
            for (i, k) in enumerate(a.kinks)
                @test a.xs[i][end] == log10(k) == a.xs[i+1][1]
            end
            Es = vcat(exp.(range(0.0, log(E_max); length=600)),
                      [k * m for k in a.kinks for m in (0.98, 0.999, 1.0, 1.001, 1.02)])
            worst = maximum(abs(eval_drag(a, E) - zq(E, phi, di)) / zq(E, phi, di) for E in Es)
            @test worst < 1e-4
            for k in a.kinks
                @test isfinite(eval_drag_deriv(a, k))
                @test eval_drag(a, k * (1 - 1e-9)) ≈ eval_drag(a, k * (1 + 1e-9)) rtol=1e-6
            end
        end
        # the slope really does change at E_q (ψ leaves 1): one-sided LUT derivatives differ
        a = build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=:oesi, q_oes=q)
        @test a.kinks == [2.0, 16.0]
        dl = eval_drag_deriv(a, 16.0 * (1 - 1e-6)); dr = eval_drag_deriv(a, 16.0 * (1 + 1e-6))
        @test abs(dr - dl) > 1e-3 * abs(dl)
        # q = 1: E_q = φ, coincident corners are merged (no empty piece)
        a1 = build_drag_assessor(2.0, hK, NK, :c2d2, :partialdraining; drag_interp=:oesi, q_oes=1.0)
        @test a1.kinks == [2.0] && length(a1.xs) == 2
    end

    @testset "parse_drag_interp" begin
        @test parse_drag_interp("OESI") === :oesi && parse_drag_interp(" oesi ") === :oesi
        @test parse_drag_interp("OESCUI") === :oescui && parse_drag_interp("oescui") === :oescui
        @test parse_drag_interp("BAREI") === :barei && parse_drag_interp("barei") === :barei
        @test parse_drag_interp("BARECUI") === :barecui && parse_drag_interp(" BareCUI ") === :barecui
        @test parse_drag_interp("RPI") === :rp && parse_drag_interp("rpi") === :rp
        @test parse_drag_interp("RPDP") === :rpdp
        @test parse_drag_interp("RP") === :rp
        @test parse_drag_interp("rp") === :rp
        @test parse_drag_interp("CU") === :cu
        @test parse_drag_interp(" cu ") === :cu
        @test_throws ErrorException parse_drag_interp("blend")
        @test_throws ErrorException parse_drag_interp("BARE")
        @test DEFAULT_P_PHI == 1.86
        @test_throws ErrorException build_drag_assessor(1.0, 0.25, 5000.0, :c2d2,
                                                        :partialdraining; drag_interp=:cu, p_phi=0.0)
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
