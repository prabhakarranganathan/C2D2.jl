"""
    test_pmf.jl

Tests for the steady planar-mixed-flow (PMF) branch of the steady model pack
(simple shear at r = 1), added 2026-10-16 per Aru's handover
docs/logs/aru-to-ko/2026-10-16_planar-mixed-flow-steady.md.

Test groups
  1. StyextFlows kinematics: poles, μ-maps, analytic T_p partials vs FD.
  2. r = 0 regression: PMF(r=0) ≡ STYEXT planar extension α = (1, −1, 0)
     (fixed-Y Newton, fixed-Wi Newton, PAC manifold).
  3. r = 1 vs Python full-tensor reference CSVs (Aru, c2d2_tensor.py):
     η_p, Ψ₁, Y to relative 1e-6 at the CSV Wi grid (Wi_csv = λ₀γ̇ = 2·Wi).
  4. Hookean (UCM) analytic: η_p = Σλ̂_p, Ψ₁ = 2Σλ̂_p², Ψ₂ = 0.
  5. King–McKinley identity (Nm = 1): eigenvalues of the steady shear tensor
     equal the planar-extension tensor at ε̇_eff = e₁′·D·e₁′, and the STYEXT
     planar solve at Wi_eff has the same closure state Y.
  6. Front-end: Wi_convention = "shear" scaling; alphas ignored for pmf.

Run with (repo root):
  julia --project=julia julia/experiments/test/test_pmf.jl
"""

using Test
using LinearAlgebra

const _MODELS = joinpath(@__DIR__, "..", "..", "models")

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
using .SteadyWiSweep
using .StyextRunContext

const INPUTS = joinpath(@__DIR__, "..", "pmf", "inputs")
const REF    = joinpath(@__DIR__, "reference")

# ── Helpers ───────────────────────────────────────────────────────────────────

"""Load a numeric CSV with '#' comment lines; returns (header, matrix)."""
function _load_csv(path)
    lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), readlines(path))
    hdr   = String.(strip.(split(lines[1], ',')))
    data  = [parse(Float64, strip(v)) for l in lines[2:end] for v in split(l, ',')]
    return hdr, permutedims(reshape(data, length(hdr), :))
end
_col(hdr, name) = (i = findfirst(==(name), hdr); i === nothing ? error("no column $name") : i)
_relerr(a, b) = abs(a - b) / max(abs(b), 1e-300)

"""Strain-axes stresses from a PMF C2D2Obs (flow-frame storage)."""
function _strain_frame(o::C2D2Obs)
    half = 0.5 * (o.tau_xx + o.tau_yy)
    return (t11 = half + o.tau_xy, t22 = half - o.tau_xy, t33 = o.tau_zz,
            t12 = 0.5 * (o.tau_yy - o.tau_xx))
end

"""Fixed-Y Newton continuation along a Y grid; returns Vector of (X, Y, mu, obs)."""
function _fixed_Y_path(ctx, solver_opts, Ygrid)
    Na  = ctx.Na
    X   = Na > 0 ? fill(ctx.Q0_sq, Na) : Float64[]
    mu  = 1e-3
    out = []
    for Y in Ygrid
        z0 = pack_ztilde(X, Y, mu, Na)
        z, st = newton_fixed_Y(ctx, z0, Y, solver_opts)
        @test st.converged
        X, _, mu = unpack_ztilde(z, Na)
        push!(out, (X = copy(X), Y = Y, mu = mu, obs = eval_state_obs(ctx, X, Y, mu)))
    end
    return out
end

# ═════════════════════════════════════════════════════════════════════════════
# 1. Kinematics
# ═════════════════════════════════════════════════════════════════════════════

@testset "PMF kinematics (StyextFlows)" begin
    f0, f05, f1, f2 = PlanarMixedFlow(0.0), PlanarMixedFlow(0.5), PlanarMixedFlow(1.0), PlanarMixedFlow(2.0)
    ext_planar = PrincipalExtension((1.0, -1.0, 0.0))
    θ = [1.0, 2.0, 5.0]

    @test has_pole(f0) && has_pole(f05)
    @test !has_pole(f1) && !has_pole(f2)
    @test wi_pole(f0, θ) == wi_pole(ext_planar, θ) == 0.5
    @test wi_pole(f05, θ) ≈ 1.0 / (2 * sqrt(0.75))
    @test wi_pole(f1, θ) == Inf && wi_pole(f2, θ) == Inf
    @test pole_mode_index(f05, [3.0, 1.0, 2.0]) == 2
    @test_throws ErrorException PlanarMixedFlow(-0.1)

    @testset "μ-map roundtrip + derivatives" begin
        for (f, Wp) in ((f05, 0.7), (f1, Inf))
            for mu in (0.0, 0.3, 2.0, 7.0)
                Wi, dWi_dmu, dWi_dWp = wi_map(f, mu, Wp)
                @test mu_from_wi(f, Wi, Wp) ≈ mu atol = 1e-12
                h = 1e-6
                Wi_p, _, _ = wi_map(f, mu + h, Wp)
                Wi_m, _, _ = wi_map(f, mu - h, Wp)
                @test dWi_dmu ≈ (Wi_p - Wi_m) / (2h) rtol = 1e-6
                if has_pole(f)
                    Wi_q, _, _ = wi_map(f, mu, Wp + h)
                    @test dWi_dWp ≈ (Wi_q - Wi) / h rtol = 1e-6
                else
                    @test dWi_dWp == 0.0
                end
            end
        end
        @test wi_shear(0.25) == 0.5
    end

    @testset "T_p kernel: r=0 ≡ planar extension; partials vs FD" begin
        Nm = 4
        θv = [1.0, 2.3, 4.1, 9.0];  σv = [0.33, 0.7, 1.2, 2.9]
        Tp  = zeros(Nm); dW = zeros(Nm); dθ = zeros(Nm)
        Tp2 = zeros(Nm); dW2 = zeros(Nm); dθ2 = zeros(Nm)
        for Wi in (0.0, 0.1, 0.3, 0.45)
            md1 = StyextModelPack._modal_Tp!(Tp, dW, dθ, θv, σv, f0, Wi)
            md2 = StyextModelPack._modal_Tp!(Tp2, dW2, dθ2, θv, σv, ext_planar, Wi)
            @test Tp ≈ Tp2 rtol = 1e-13
            @test dW ≈ dW2 rtol = 1e-12 atol = 1e-15
            @test dθ ≈ dθ2 rtol = 1e-12
            @test sign(md1) == sign(md2)
        end
        # analytic partials against FD for r = 0.5 and r = 1
        for f in (f05, f1, f2), Wi in (0.2, 1.0, 10.0)
            StyextModelPack._modal_Tp!(Tp, dW, dθ, θv, σv, f, Wi)
            h = 1e-6
            StyextModelPack._modal_Tp!(Tp2, dW2, dθ2, θv, σv, f, Wi + h); Tp_p = copy(Tp2)
            StyextModelPack._modal_Tp!(Tp2, dW2, dθ2, θv, σv, f, Wi - h); Tp_m = copy(Tp2)
            @test dW ≈ (Tp_p .- Tp_m) ./ (2h) rtol = 1e-6
            StyextModelPack._modal_Tp!(Tp2, dW2, dθ2, θv .+ h, σv, f, Wi); Tp_p = copy(Tp2)
            StyextModelPack._modal_Tp!(Tp2, dW2, dθ2, θv .- h, σv, f, Wi); Tp_m = copy(Tp2)
            @test dθ ≈ (Tp_p .- Tp_m) ./ (2h) rtol = 1e-6
            # trace identity: A + B + D == T_p
            A, B, C, D = modal_tensors(f, θv, σv, Wi)
            @test A .+ B .+ D ≈ Tp rtol = 1e-13
            # positive definiteness of the in-plane block (report: all Wi for r ≥ 1)
            if !has_pole(f)
                @test all(A .> 0) && all(A .* B .- C .^ 2 .> 0)
            end
        end
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# 2. r = 0 regression against STYEXT planar extension
# ═════════════════════════════════════════════════════════════════════════════

@testset "r=0 regression: PMF(r=0) ≡ STYEXT planar α=(1,-1,0)" begin
    pP, ctxP, soP, moP = load_styext_context(joinpath(INPUTS, "pmf_r0_regression.toml"))
    pE, ctxE, soE, moE = load_styext_context(joinpath(INPUTS, "styext_planar_regression.toml"))
    @test pP.flow_kind === :pmf && pE.flow_kind === :styext
    @test ctxP.flow isa PlanarMixedFlow && ctxE.flow isa PrincipalExtension
    @test ctxP.alphas == ctxE.alphas == (1.0, -1.0, 0.0)
    @test ctxP.Ymin == ctxE.Ymin && ctxP.Ymax == ctxE.Ymax

    @testset "fixed-Y Newton" begin
        Ygrid = ctxP.Ymin .* [1.001, 1.01, 1.1, 1.5, 3.0, 10.0, 50.0, 200.0]
        Ygrid = filter(<(ctxP.Ymax), Ygrid)
        pathP = _fixed_Y_path(ctxP, soP, Ygrid)
        pathE = _fixed_Y_path(ctxE, soE, Ygrid)
        for (a, b) in zip(pathP, pathE)
            @test a.obs.Wi    ≈ b.obs.Wi    rtol = 1e-9
            @test a.X         ≈ b.X         rtol = 1e-9
            @test a.obs.etaE1 ≈ b.obs.etaE1 rtol = 1e-8
            @test a.obs.etaE2 ≈ b.obs.etaE2 rtol = 1e-8
            s = _strain_frame(a.obs)
            @test s.t11 ≈ b.obs.tau_xx rtol = 1e-8
            @test s.t22 ≈ b.obs.tau_yy rtol = 1e-8
            @test s.t33 ≈ b.obs.tau_zz rtol = 1e-8
            @test abs(s.t12) < 1e-10 * max(abs(s.t11), 1.0)        # no shear stress at r = 0
            @test a.obs.Psi1 == 0.0 || abs(a.obs.Psi1) < 1e-10      # flow-frame N1 = −2τ₁₂ = 0
        end
    end

    @testset "fixed-Wi Newton" begin
        # Coiled branch below the coil–stretch fold (Wi_pole(eq) = 0.5; the
        # branch ends at Wi ≈ 0.456 for this case — see the fold check below).
        Wis = [0.05, 0.1, 0.2, 0.3, 0.4]
        tP = sweep_fixed_Wi(ctxP, Wis, soP)
        tE = sweep_fixed_Wi(ctxE, Wis, soE)
        @test length(tP) == length(tE) == length(Wis) + 1
        @test all(p -> p.status_str in ("eq", "FWi-cnv"), tP)
        @test all(p -> p.status_str in ("eq", "FWi-cnv"), tE)
        for (a, b) in zip(tP[2:end], tE[2:end])
            @test a.Y         ≈ b.Y         rtol = 1e-9
            @test a.X         ≈ b.X         rtol = 1e-9
            @test a.obs.etaE1 ≈ b.obs.etaE1 rtol = 1e-8
            @test a.Wi_pole   ≈ b.Wi_pole   rtol = 1e-12
        end
        # Beyond the fold the coiled branch has no steady state at fixed Wi:
        # both flows must fail at the same Wi with the same last admissible pole.
        fP = sweep_fixed_Wi(ctxP, [0.4, 0.45], soP; max_bisect = 12)
        fE = sweep_fixed_Wi(ctxE, [0.4, 0.45], soE; max_bisect = 12)
        @test fP[end].status_str == "FWi-fail" == fE[end].status_str
        @test fP[end].Wi_pole ≈ fE[end].Wi_pole rtol = 1e-10
        @test 0.4 < fP[end].Wi_pole < 0.5
    end

    @testset "PAC manifold (interpolated Wi(Y) curves)" begin
        tP = build_manifold(ctxP, soP, moP)
        tE = build_manifold(ctxE, soE, moE)
        @test length(tP) > 10 && length(tE) > 10
        YE  = [p.Y for p in tE];  WE = [p.obs.Wi for p in tE]
        # both curves monotone in Y → interpolate E's Wi at P's Y (log-log)
        for p in tP[2:end]
            (p.Y <= YE[2] || p.Y >= YE[end]) && continue
            j = searchsortedfirst(YE, p.Y)
            w = (log(p.Y) - log(YE[j-1])) / (log(YE[j]) - log(YE[j-1]))
            Wi_interp = exp((1 - w) * log(WE[j-1]) + w * log(WE[j]))
            @test p.obs.Wi ≈ Wi_interp rtol = 2e-3
        end
        println("  r=0 regression: PMF manifold $(length(tP)) pts (Wi_end=$(round(tP[end].obs.Wi; sigdigits=6))), " *
                "STYEXT planar $(length(tE)) pts (Wi_end=$(round(tE[end].obs.Wi; sigdigits=6)))")
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# 3. r = 1 vs Python full-tensor reference
# ═════════════════════════════════════════════════════════════════════════════

# (case, rtol). Constant drag agrees to ~3e-11 (the PMF algebra and the
# multimode closure are exact vs the Python port). The c2d2 cases are limited
# by the drag look-up table: with the global default of 400 PCHIP points the
# error is ~1e-5 (at E ≈ 1.003, the drag onset); the TOMLs set 4000 points,
# which gives ~1e-7 (floor ~6e-8 at 40 000 points). See the 400-pt sub-test.
const REF_CASES = [
    ("pmf_shear_N1_fenepm_c2d2",         1e-6),
    ("pmf_shear_N10_rouse_fenepm_const", 1e-9),
    ("pmf_shear_N10_zimm_fenepm_c2d2",   1e-6),
]

@testset "r=1 vs Python reference: $(case)" for (case, rtol) in REF_CASES
    toml = joinpath(INPUTS, "$case.toml")
    csv  = joinpath(REF,    "$case.csv")
    params, ctx, so, mo = load_styext_context(toml)
    @test params.flow_kind === :pmf && params.r == 1.0 && params.Wi_convention === :shear
    @test ctx.flow isa PlanarMixedFlow && !has_pole(ctx.flow)

    hdr, data = _load_csv(csv)
    Wi_csv  = data[:, _col(hdr, "Wi")]            # λ₀γ̇ (lab shear rate)
    Wi_rep  = Wi_csv ./ 2                          # report Wi (principal strain rate)
    traj = sweep_fixed_Wi(ctx, Wi_rep, so; include_equilibrium = false)
    @test length(traj) == length(Wi_rep)
    @test all(p -> p.status_str == "FWi-cnv", traj)

    e_eta = 0.0; e_psi = 0.0; e_Y = 0.0; e_txy = 0.0; psi2_max = 0.0
    for (i, p) in enumerate(traj)
        o = p.obs
        @test o.Wi ≈ Wi_rep[i]
        e_eta = max(e_eta, _relerr(o.eta_p, data[i, _col(hdr, "eta_p")]))
        e_psi = max(e_psi, _relerr(o.Psi1,  data[i, _col(hdr, "Psi1")]))
        e_Y   = max(e_Y,   _relerr(o.E2,    data[i, _col(hdr, "Y")]))
        # Python stress is tensile-positive; Julia is −Σ(3fM − I) ⇒ opposite sign
        e_txy = max(e_txy, _relerr(-o.tau_xy, data[i, _col(hdr, "tau_xy")]))
        psi2_max = max(psi2_max, abs(o.Psi2) / max(abs(o.Psi1), 1e-300))
        @test o.eta_p > 0 && o.Psi1 > 0
    end
    @test e_eta < rtol
    @test e_psi < rtol
    @test e_Y   < rtol
    @test e_txy < rtol
    @test psi2_max < 1e-10          # Ψ₂ ≡ 0 analytically at r = 1; residual is cancellation roundoff (~3e-12 at Nm = 10)
    println("  $case: $(length(traj)) pts  max rel err  eta_p=$(round(e_eta; sigdigits=3))  " *
            "Psi1=$(round(e_psi; sigdigits=3))  Y=$(round(e_Y; sigdigits=3))  tau_xy=$(round(e_txy; sigdigits=3))  |Psi2/Psi1|max=$(round(psi2_max; sigdigits=2))")
end

@testset "r=1 vs Python reference: default 400-pt drag LUT (documented limit)" begin
    # Same N1 c2d2 case with the global-default LUT resolution: agreement is
    # ~1e-5, dominated by PCHIP interpolation of ζ(E) near E = 1. Kept as a
    # regression guard on the magnitude, not as a precision claim.
    src = read(joinpath(INPUTS, "pmf_shear_N1_fenepm_c2d2.toml"), String)
    tmp = joinpath(mktempdir(), "n1_lut400.toml")
    write(tmp, replace(src, r"drag_lut_n_points\s*=\s*\d+" => "drag_lut_n_points = 400"))
    params, ctx, so, _ = load_styext_context(tmp)
    @test params.drag_lut_n_points == 400
    hdr, data = _load_csv(joinpath(REF, "pmf_shear_N1_fenepm_c2d2.csv"))
    traj = sweep_fixed_Wi(ctx, data[:, _col(hdr, "Wi")] ./ 2, so; include_equilibrium = false)
    e_eta = maximum(_relerr(p.obs.eta_p, data[i, _col(hdr, "eta_p")]) for (i, p) in enumerate(traj))
    e_Y   = maximum(_relerr(p.obs.E2,    data[i, _col(hdr, "Y")])     for (i, p) in enumerate(traj))
    @test e_eta < 5e-5
    @test e_Y   < 1e-6
    println("  400-pt LUT: max rel err eta_p=$(round(e_eta; sigdigits=3))  Y=$(round(e_Y; sigdigits=3))")
end

# ═════════════════════════════════════════════════════════════════════════════
# 4. Hookean (UCM) analytic
# ═════════════════════════════════════════════════════════════════════════════

@testset "Hookean steady shear: multimode UCM analytic" begin
    for nm in (1, 5)
        params, ctx, so, mo = load_styext_context(joinpath(INPUTS, "pmf_shear_hookean_N$(nm)_const.toml"))
        @test params.fe_model === :hookean && ctx.Na == 0
        lhat = (1:nm) .^ (-2.0)
        traj = sweep_fixed_Wi(ctx, params.Wi_sweep, so; include_equilibrium = false)
        @test length(traj) == length(params.Wi_sweep)
        for p in traj
            @test p.status_str == "FWi-cnv"
            @test p.obs.eta_p ≈ sum(lhat)          rtol = 1e-10
            @test p.obs.Psi1  ≈ 2 * sum(lhat .^ 2) rtol = 1e-10
            @test abs(p.obs.Psi2) < 1e-10 * p.obs.Psi1    # Ψ₂ ≡ 0; cancellation roundoff
            # UCM N1 = 2 η_p λ γ̇² ⇒ Ψ₁ = 2 η_p λ for Nm = 1
            nm == 1 && @test p.obs.Psi1 ≈ 2 * p.obs.eta_p * ctx.lambda0 rtol = 1e-10
            # Y = Σ S_p tr M_p with tr M_p = 1 + 8 λ̂_p² Wi²/3 ... check via E2 > Ymin
            @test p.Y > ctx.Ymin
        end
        println("  Hookean Nm=$nm: eta_p=$(traj[1].obs.eta_p)  Psi1=$(traj[1].obs.Psi1)  (analytic $(sum(lhat)), $(2*sum(lhat.^2)))")
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# 5. King–McKinley identity (Nm = 1)
# ═════════════════════════════════════════════════════════════════════════════

@testset "King–McKinley identity (Nm=1, FENE-PM, c2d2)" begin
    pS, ctxS, soS, _ = load_styext_context(joinpath(INPUTS, "pmf_shear_N1_fenepm_c2d2.toml"))
    pE, ctxE, soE, _ = load_styext_context(joinpath(INPUTS, "styext_planar_N1_fenepm_c2d2.toml"))
    Wis = [0.05, 0.5, 5.0, 50.0]                 # report Wi (strain-rate)
    traj = sweep_fixed_Wi(ctxS, Wis, soS; include_equilibrium = false)
    @test all(p -> p.status_str == "FWi-cnv", traj)
    for p in traj
        θ, σ, _ = StyextModelPack._eval_ts!(ctxS, p.X, p.Y)
        A, B, C, D = modal_tensors(ctxS.flow, θ, σ, p.obs.Wi)
        M2 = [A[1] C[1]; C[1] B[1]]
        ev = eigen(Symmetric(M2))
        m_min, m_max = ev.values
        e1 = ev.vectors[:, 2]                      # principal (most stretched) axis
        eps_eff = p.obs.Wi * (e1[1]^2 - e1[2]^2)   # e₁′·D·e₁′, D = Wi·diag(1,−1) in strain frame
        @test 0 < eps_eff < p.obs.Wi
        # (a) algebraic identity at the shared closure state
        @test m_max ≈ σ[1] / (θ[1] - 2 * eps_eff) rtol = 1e-10
        @test m_min ≈ σ[1] / (θ[1] + 2 * eps_eff) rtol = 1e-10
        @test D[1]  ≈ σ[1] / θ[1]                  rtol = 1e-14
        # (b) the STYEXT planar steady state at Wi_eff has the same closure state
        zE, stE = newton_fixed_Wi(ctxE, pack_ztilde(p.X, p.Y, 0.0, ctxE.Na), eps_eff, soE)
        # Seeded at the shear state the planar residual is already at roundoff
        # (Y ~ 1e2–1e3 near the pole), which can sit just above tol_res = 1e-10.
        @test stE.converged || stE.residual_norm < 1e-8 * max(p.Y, 1.0)
        XE, YE, _ = unpack_ztilde(zE, ctxE.Na)
        @test YE ≈ p.Y rtol = 1e-8
        @test XE ≈ p.X rtol = 1e-8
    end
    println("  K–M identity verified at Wi = $Wis")
end

# ═════════════════════════════════════════════════════════════════════════════
# 5b. X_stop termination (FENE-PM, Nm > 1): Y saturates below E∞², X → Q∞²
# ═════════════════════════════════════════════════════════════════════════════

@testset "X_stop: multimode FENE-PM terminates on the X closure" begin
    params, ctx, so, mo = load_styext_context(joinpath(INPUTS, "pmf_shear_N10_rouse_fenepm_const.toml"))
    @test mo.X_stop ≈ params.max_frac_stretch_sq * params.Qinf_sq
    # PAC in shear with no Wi cap: must stop on X_stop, not run to max_steps
    mo2 = ManifoldOpts(ds0 = mo.ds0, ds_min = mo.ds_min, ds_max = mo.ds_max, max_steps = 20000,
                       Wi_max = Inf, tol_FY_rel_hard = mo.tol_FY_rel_hard,
                       target_newton = mo.target_newton, target_rho = mo.target_rho, X_stop = mo.X_stop)
    traj, reason = build_manifold(ctx, so, mo2; return_reason = true)
    @test reason == "X_stop"
    @test traj[end].X[1] ≈ mo.X_stop rtol = 1e-3
    @test traj[end].Y < ctx.Ymax                      # Y never reaches the E∞² cap
    # fixed-Wi sweep: X_stop ends it with a labelled point
    # shear saturates slowly (X/Q∞² ≈ 0.96 at λ₀γ̇ = 1e4), so sweep to 1e7
    tr = sweep_fixed_Wi(ctx, exp10.(range(-2, 7; length = 91)) ./ 2, so; X_stop = mo.X_stop)
    @test tr[end].status_str == "X_stop"
    @test all(p -> p.status_str in ("eq", "FWi-cnv", "X_stop"), tr)
    # linear limit matches the Wi → 0 end of the sweep
    eta0, Psi10 = linear_limit_shear(ctx)
    @test eta0  ≈ sum((1:10) .^ -2.0)          rtol = 1e-12
    @test Psi10 ≈ 2 * sum((1:10) .^ -4.0)      rtol = 1e-12
    @test tr[2].obs.eta_p ≈ eta0 rtol = 1e-4   # λ₀γ̇ = 1e-2
    println("  X_stop: PAC $(length(traj)) pts, reason=$reason, X_end/Q∞²=$(round(traj[end].X[1]/params.Qinf_sq; digits=5)); sweep ends $(tr[end].status_str) at Wi_shear=$(round(wi_shear(tr[end].obs.Wi); sigdigits=4))")
end

# ═════════════════════════════════════════════════════════════════════════════
# 6. Front end
# ═════════════════════════════════════════════════════════════════════════════

@testset "Front end: Wi_convention and flow kind" begin
    params, ctx, so, mo = load_styext_context(joinpath(INPUTS, "pmf_shear_N1_fenepm_c2d2.toml"))
    @test params.Wi_convention === :shear
    @test length(params.Wi_sweep) == 26
    @test params.Wi_sweep[1]   ≈ 0.5 * 1e-2 rtol = 1e-12     # λ₀γ̇ = 1e-2 → Wi = 5e-3
    @test params.Wi_sweep[end] ≈ 0.5 * 1e3  rtol = 1e-12
    @test mo.Wi_max == 500.0                                  # limits.Wi_max = 1e3 (shear) → 500 (strain)
    @test params.drag_lut_n_points == 4000
    @test params.Qinf_sq == params.NK / params.Nm           # STYEXT FENE-PM convention

    # equilibrium observables: shear fields finite and zero
    o0 = eval_state_obs(ctx, [ctx.Q0_sq], ctx.Ymin, 0.0)
    @test o0.Wi == 0.0 && abs(o0.tau_xy) < 1e-14 && abs(o0.eta_p) < 1e-14
    # extension contexts leave the shear fields NaN
    pE, ctxE, soE, _ = load_styext_context(joinpath(INPUTS, "styext_planar_regression.toml"))
    oE = eval_state_obs(ctxE, [ctxE.Q0_sq], ctxE.Ymin, 0.0)
    @test isnan(oE.tau_xy) && isnan(oE.eta_p) && isnan(oE.Psi1)
    @test isnan(pE.r) && isempty(pE.Wi_sweep)
end

println("\nAll PMF tests complete.")
