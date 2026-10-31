"""
    StyextModelPack.jl

Model-pack for the steady homogeneous-flow PAC solver: steady extension
(STYEXT) and, since 2026-10-16, steady planar mixed flows (PMF; simple shear
at r = 1). The flow kinematics are carried by `ctx.flow`
(`StyextFlows.PrincipalExtension` | `StyextFlows.PlanarMixedFlow`); everything
else — closure scalars, PAC/fixed-Y/fixed-Wi residuals, Newton and the
manifold builder — is shared.

Mirrors MATLAB's
  styext_modelpack_core.m / styext_FENE_PM_modelpack.m / styext_FENE_PME_modelpack.m
for the extensional branch.

Steady modal tensors
--------------------
Extension (κ = Wi·diag α):   M_pα = σ_p / (θ_p − 2 Wi α_α)       (diagonal)
PMF (κ = Wi·κ̂(r)), report `eq:pmf-components-r`, strain-axes frame:
    A_p = σ_p(θ_p² + 2θ_p Wi + 4r²Wi²) / (θ_p D_p)
    B_p = σ_p(θ_p² − 2θ_p Wi + 4r²Wi²) / (θ_p D_p)
    C_p = −4 σ_p r Wi² / (θ_p D_p)
    D_p = σ_p/θ_p,            D_p := θ_p² − 4(1 − r²)Wi²  (common denominator)
and the modal trace (`eq:pmf-trace-r`)
    T_p = (σ_p/θ_p)(3θ_p² + 4(3r² − 1)Wi²) / (θ_p² − 4(1 − r²)Wi²).
At r = 0 these reduce exactly to planar extension with α = (1, −1, 0).

Normalisation convention
------------------------
MATLAB's coeffs_relaxation uses r = √Y / E₀ as the drag/spectrum LUT argument
(where E₀ = √Nm for Q₀²=1).  Julia's CoeffsKernel uses E = Z.E directly.
To match MATLAB for Nm > 1 we set Z.E = √Y / E₀ before calling eval_coeffs!.

FENE pole convention: Qinf_sq is supplied by the caller. STYEXT/PMF use
Q∞² = Q₀²·NK for FENE-PME and Q₀²·NK/Nm (per spring) for FENE-PM/P; the VELB
path (Params.jl) uses NK throughout. The two silently differ for Nm > 1.

Stress sign: `Stress.polymer_stress` gives τ/(c kBT) = −Σ_p(3 f_p M_p − I)
(compressive-positive, STYEXT convention). The material functions below carry
the compensating minus sign so η_E, η_p, Ψ₁ are positive in their usual sense.

State-vector layout
-------------------
FENE-PME / Hookean (Na=0) :  ztilde = [Y; μ]              length 2
FENE-PM  (Na=1)           :  ztilde = [X; Y; μ]           length 3
FENE-P   (Na=Nm)          :  ztilde = [X₁…Xₙ; Y; μ]       length Nm+2

Public API
----------
  StyextMPCtx         — assembled model-pack context
  build_mp_ctx(...)   — constructor (keyword `flow` selects the kinematics)
  pack_ztilde         — (X, Y, μ) → vector
  unpack_ztilde       — vector → (X, Y, μ)
  rj_pac              — PAC residual + Jacobian
  rj_fixed_Y          — fixed-Y residual + Jacobian
  rj_fixed_Wi         — fixed-Wi residual + Jacobian  (unknowns [X; Y])
  eval_state_obs      — C2D2Obs from current (X, Y, μ)
  get_Wi_pole         — Wi_pole at (X, Y)  (Inf for pole-free flows)
  modal_tensors       — steady modal tensors at given (θ_p, σ_p, Wi)
"""
module StyextModelPack

using LinearAlgebra: norm, dot

using ..Fene:       AbstractFeneModel, FenePME, FenePM, FeneP, fene_peterlin
using ..CoeffsRelax: CoeffsKernel, ClosureZ, eval_coeffs!
using ..StyextFlows: AbstractSteadyFlow, PrincipalExtension, PlanarMixedFlow,
                     has_pole, wi_pole, pole_mode_index, wi_map, mu_from_wi, wi_shear
import ..StyextFlows: dwi_pole_dtheta
using ..Stress:     polymer_stress
using ..StateObs:   C2D2Obs

export StyextMPCtx, build_mp_ctx,
       pack_ztilde, unpack_ztilde,
       rj_pac, rj_fixed_Y, rj_fixed_Wi, eval_state_obs, get_Wi_pole,
       modal_tensors, eval_modal_tensors, linear_limit_shear

# ── Context struct ────────────────────────────────────────────────────────────

"""
    StyextMPCtx

Assembled context for the steady model pack.

Fields
------
kernel   : CoeffsKernel (FENE + drag LUT + spectrum; build once, reuse)
Sp       : Nm-vector of Rouse end-to-end weights  (Y-closure weights wY)
alphas   : NTuple{3,Float64} — principal flow-rate ratios (αx, αy, αz).
           For a PlanarMixedFlow this is the strain-axes frame (1, −1, 0).
amax     : maximum of alphas (extensional axis rate; 1 for PMF)
Na       : number of X auxiliaries (0=PME/Hookean, 1=PM, Nm=P)
Nm       : number of modes
Ymin     : lower bound on Y = E₀² = Nm Q₀²
Ymax     : upper bound on Y = E∞²
E0       : √Ymin (= √Nm for Q₀²=1) — drag/spectrum LUT normalisation
Q0_sq    : Q₀² (= 1.0 in our dimensionless convention)
Qinf_sq  : FENE pole (Q₀²·NK for PME; Q₀²·NK/Nm for PM/P)
lambda0  : longest relaxation time [time units]
etas     : solvent viscosity
ckBT     : c·k_B·T
h_fd     : FD step for d/dY numerical derivatives (default 1e-6)
flow     : AbstractSteadyFlow — kinematics (PrincipalExtension | PlanarMixedFlow)
"""
struct StyextMPCtx
    kernel   :: CoeffsKernel
    Sp       :: Vector{Float64}
    alphas   :: NTuple{3, Float64}
    amax     :: Float64
    Na       :: Int
    Nm       :: Int
    Ymin     :: Float64
    Ymax     :: Float64
    E0       :: Float64
    Q0_sq    :: Float64
    Qinf_sq  :: Float64
    lambda0  :: Float64
    etas     :: Float64
    ckBT     :: Float64
    h_fd     :: Float64
    flow     :: AbstractSteadyFlow
end

"""
    build_mp_ctx(kernel, Sp, alphas, Na, Nm, E0_sq, Einf_sq,
                 Q0_sq, Qinf_sq, lambda0, etas, ckBT;
                 flow = PrincipalExtension(alphas)) -> StyextMPCtx

`flow` selects the kinematics. Without it the context is the historical STYEXT
one (principal extension with `alphas`). For `PlanarMixedFlow` the stored
`alphas` are overwritten with the strain-axes frame (1, −1, 0).
"""
function build_mp_ctx(kernel   :: CoeffsKernel,
                      Sp       :: Vector{Float64},
                      alphas   :: NTuple{3, Float64},
                      Na       :: Int,
                      Nm       :: Int,
                      E0_sq    :: Float64,
                      Einf_sq  :: Float64,
                      Q0_sq    :: Float64,
                      Qinf_sq  :: Float64,
                      lambda0  :: Float64,
                      etas     :: Float64,
                      ckBT     :: Float64;
                      flow     :: Union{Nothing, AbstractSteadyFlow} = nothing)::StyextMPCtx
    if flow === nothing
        flow = PrincipalExtension(alphas)        # validates amax > 0
    end
    if flow isa PrincipalExtension
        alphas_ctx = flow.alphas
        amax       = flow.amax
    else
        alphas_ctx = (1.0, -1.0, 0.0)
        amax       = 1.0
    end
    E0 = sqrt(E0_sq)
    return StyextMPCtx(kernel, Sp, alphas_ctx, amax,
                       Na, Nm, E0_sq, Einf_sq,
                       E0, Q0_sq, Qinf_sq,
                       lambda0, etas, ckBT, 1e-6, flow)
end

# ── Pack / unpack ─────────────────────────────────────────────────────────────

"""
    pack_ztilde(X, Y, mu, Na) -> Vector{Float64}

Pack (X, Y, μ) into a flat continuation vector ztilde = [X; Y; μ].
"""
function pack_ztilde(X::Vector{Float64}, Y::Float64, mu::Float64, Na::Int)::Vector{Float64}
    v = Vector{Float64}(undef, Na + 2)
    if Na > 0
        v[1:Na] .= X
    end
    v[Na + 1] = Y
    v[Na + 2] = mu
    return v
end

"""
    unpack_ztilde(v, Na) -> (X, Y, mu)

Inverse of pack_ztilde.  X is a fresh copy (length Na; empty for Na=0).
"""
function unpack_ztilde(v::AbstractVector{Float64}, Na::Int)
    length(v) == Na + 2 ||
        error("unpack_ztilde: length(v)=$(length(v)) ≠ Na+2=$(Na+2)")
    X  = Na > 0 ? Vector{Float64}(v[1:Na]) : Float64[]
    Y  = Float64(v[Na + 1])
    mu = Float64(v[Na + 2])
    return X, Y, mu
end

# ── Low-level θ / σ evaluation ───────────────────────────────────────────────

"""
    _eval_ts!(ctx, X, Y) -> (theta_p, sigma_p, fp)

Compute θ_p and σ_p by calling eval_coeffs! on the kernel.
Sets Z.E = √Y / E₀ (MATLAB-compatible normalisation).
Returns COPIES of the scratch vectors so the caller can modify them.
"""
function _eval_ts!(ctx::StyextMPCtx,
                   X::Vector{Float64}, Y::Float64)
    E_norm = sqrt(max(Y, 0.0)) / ctx.E0
    Z = ClosureZ(X, Y, E_norm)    # ClosureZ(Xa, Y, E) — explicit E
    eval_coeffs!(ctx.kernel, Z)
    sc = ctx.kernel.scratch
    return copy(sc.theta_p), copy(sc.sigma_p), copy(sc.fp)
end

# ── Clamp Y to the valid interval ─────────────────────────────────────────────

@inline _clamp_Y(Y::Float64, ctx::StyextMPCtx) = clamp(Y, ctx.Ymin, ctx.Ymax)

# ── Modal trace T_p and its partials (flow-kinematics kernel) ─────────────────

"""
    _modal_Tp!(Tp, dTp_dWi, dTp_dth, theta_p, sigma_p, flow, Wi) -> min_denom

Fill, per mode p, the steady modal trace T_p = tr M_p and its partials
∂T_p/∂Wi (θ_p, σ_p fixed) and ∂T_p/∂θ_p (σ_p, Wi fixed). Returns the smallest
kinematic denominator over all modes (≤ 0 ⇒ inadmissible):
  extension : min over (p, α) of  D_pα = θ_p − 2 Wi α_α
  PMF       : min over p of       D_p  = θ_p² − 4(1 − r²) Wi²
"""
function _modal_Tp!(Tp::Vector{Float64}, dTp_dWi::Vector{Float64}, dTp_dth::Vector{Float64},
                    theta_p::Vector{Float64}, sigma_p::Vector{Float64},
                    f::PrincipalExtension, Wi::Float64)::Float64
    Nm = length(theta_p)
    min_d = Inf
    @inbounds for p in 1:Nm
        tp = theta_p[p];  sp = sigma_p[p]
        s1 = 0.0;  s2 = 0.0;  sa = 0.0
        for α in 1:3
            a  = f.alphas[α]
            d  = tp - 2.0 * Wi * a
            id = 1.0 / d
            s1 += id
            s2 += id * id
            sa += a * id * id
            min_d = min(min_d, d)
        end
        Tp[p]      = sp * s1
        dTp_dWi[p] = sp * 2.0 * sa
        dTp_dth[p] = -sp * s2
    end
    return min_d
end

function _modal_Tp!(Tp::Vector{Float64}, dTp_dWi::Vector{Float64}, dTp_dth::Vector{Float64},
                    theta_p::Vector{Float64}, sigma_p::Vector{Float64},
                    f::PlanarMixedFlow, Wi::Float64)::Float64
    Nm  = length(theta_p)
    r2  = f.r^2
    W2  = Wi^2
    cD  = 4.0 * (1.0 - r2)          # D = θ² − cD·Wi²
    cN  = 4.0 * (3.0 * r2 - 1.0)    # N = 3θ² + cN·Wi²
    min_d = Inf
    @inbounds for p in 1:Nm
        tp  = theta_p[p];  sp = sigma_p[p]
        th2 = tp * tp
        D   = th2 - cD * W2
        N   = 3.0 * th2 + cN * W2
        Tp[p]      = sp * N / (tp * D)
        dTp_dWi[p] = 16.0 * sp * tp * Wi / (D * D)
        dTp_dth[p] = sp * (6.0 * th2 * D - N * (D + 2.0 * th2)) / (th2 * D * D)
        min_d = min(min_d, D)
    end
    return min_d
end

# ── Steady modal tensors ──────────────────────────────────────────────────────

"""
    modal_tensors(flow::PrincipalExtension, theta_p, sigma_p, Wi) -> M_diag (Nm×3)
    modal_tensors(flow::PlanarMixedFlow,    theta_p, sigma_p, Wi) -> (A, B, C, D)

Steady modal conformation tensors at the given closure coefficients.
For PMF the components are in the strain-axes frame (report
`eq:pmf-components-r`): M_p = [A C 0; C B 0; 0 0 D].
"""
function modal_tensors(f::PrincipalExtension, theta_p::Vector{Float64},
                       sigma_p::Vector{Float64}, Wi::Float64)
    Nm = length(theta_p)
    M_diag = Matrix{Float64}(undef, Nm, 3)
    @inbounds for p in 1:Nm, α in 1:3
        M_diag[p, α] = sigma_p[p] / (theta_p[p] - 2.0 * Wi * f.alphas[α])
    end
    return M_diag
end

function modal_tensors(f::PlanarMixedFlow, theta_p::Vector{Float64},
                       sigma_p::Vector{Float64}, Wi::Float64)
    Nm = length(theta_p)
    A = Vector{Float64}(undef, Nm); B = similar(A); C = similar(A); D = similar(A)
    r  = f.r
    W2 = Wi^2
    @inbounds for p in 1:Nm
        tp  = theta_p[p];  sp = sigma_p[p]
        th2 = tp * tp
        den = tp * (th2 - 4.0 * (1.0 - r^2) * W2)
        A[p] = sp * (th2 + 2.0 * tp * Wi + 4.0 * r^2 * W2) / den
        B[p] = sp * (th2 - 2.0 * tp * Wi + 4.0 * r^2 * W2) / den
        C[p] = -4.0 * sp * r * W2 / den
        D[p] = sp / tp
    end
    return A, B, C, D
end

# ── Weighted T_p sum with all partial derivatives ─────────────────────────────

"""
    _Tp_weighted_all(ctx, X, Y, Wi, w, theta_p, sigma_p)
                   -> (Tsum, dTsum_dX, dTsum_dY, dTsum_dWi, min_denom)

Compute w · T_p and the three partial derivatives needed for the Jacobian.
theta_p and sigma_p are pre-computed at (X, Y) and passed in for efficiency.

- dTsum/dX : analytic (Na-vector; empty for Na=0)
- dTsum/dY : FD  (scalar; Wi held fixed)
- dTsum/dWi: analytic (scalar)
"""
function _Tp_weighted_all(ctx       :: StyextMPCtx,
                           X         :: Vector{Float64},
                           Y         :: Float64,
                           Wi        :: Float64,
                           w         :: Vector{Float64},
                           theta_p   :: Vector{Float64},
                           sigma_p   :: Vector{Float64})
    Nm = ctx.Nm
    Na = ctx.Na
    Tp      = Vector{Float64}(undef, Nm)
    dTp_dWi = Vector{Float64}(undef, Nm)
    dTp_dth = Vector{Float64}(undef, Nm)
    min_d = _modal_Tp!(Tp, dTp_dWi, dTp_dth, theta_p, sigma_p, ctx.flow, Wi)

    Tsum      = dot(w, Tp)
    dTsum_dWi = dot(w, dTp_dWi)

    # ── dTsum/dX (analytic; Na=0 → empty; Na=1 → scalar) ────────────────────
    dTsum_dX = zeros(Na)
    if Na == 1
        # dθ_p/dX = df/dX · 3·σ_p  (FENE-PM: f depends only on X[1])
        _, df_dX = fene_peterlin(X[1], ctx.Q0_sq, ctx.Qinf_sq)
        @inbounds for p in 1:Nm
            dtheta_dX_p = df_dX * 3.0 * sigma_p[p]
            dTsum_dX[1] += w[p] * dTp_dth[p] * dtheta_dX_p
        end
    elseif Na > 1
        error("_Tp_weighted_all: analytic dX only implemented for Na=0 and Na=1 (FENE-P not yet supported)")
    end

    # ── dTsum/dY (FD, Wi held fixed) ─────────────────────────────────────────
    h = ctx.h_fd
    f_Tsum = (YY::Float64) -> begin
        tp2, sp2, _ = _eval_ts!(ctx, X, _clamp_Y(YY, ctx))
        Tp2 = Vector{Float64}(undef, Nm)
        _modal_Tp!(Tp2, dTp_dWi, dTp_dth, tp2, sp2, ctx.flow, Wi)   # scratch reuse
        dot(w, Tp2)
    end
    dTsum_dY = _fd1_2nd(f_Tsum, Y, ctx.Ymin, ctx.Ymax; h=h)

    return Tsum, dTsum_dX, dTsum_dY, dTsum_dWi, min_d
end

# ── 2nd-order finite difference ───────────────────────────────────────────────

"""
    _fd1_2nd(f, y, ymin, ymax; h) -> df/dy

Second-order FD derivative, mirroring MATLAB's fd1_2nd.m.
Central if far from boundaries; 1-sided otherwise.
"""
function _fd1_2nd(f::Function, y::Float64, ymin::Float64, ymax::Float64;
                  h::Float64 = 1e-6)
    eps_bd = 2.0 * h
    if y - ymin > eps_bd && ymax - y > eps_bd
        return (f(y + h) - f(y - h)) / (2.0 * h)
    elseif y - ymin <= eps_bd
        return (-3.0*f(y) + 4.0*f(y + h) - f(y + 2.0*h)) / (2.0 * h)
    else
        return (3.0*f(y) - 4.0*f(y - h) + f(y - 2.0*h)) / (2.0 * h)
    end
end

# ── Wi_pole derivatives ───────────────────────────────────────────────────────

"""dWi_pole/dY via FD at (X, Y). Zero for pole-free flows."""
function _dWi_pole_dY_fd(ctx::StyextMPCtx, X::Vector{Float64}, Y::Float64)::Float64
    has_pole(ctx.flow) || return 0.0
    f = (YY::Float64) -> begin
        tp, _, _ = _eval_ts!(ctx, X, _clamp_Y(YY, ctx))
        wi_pole(ctx.flow, tp)
    end
    return _fd1_2nd(f, Y, ctx.Ymin, ctx.Ymax; h=ctx.h_fd)
end

"""dWi_pole/dX analytically for FENE-PM (Na=1): = (dθ_p*/dX[1]) · dWi_pole/dθ."""
function _dWi_pole_dX_PM(ctx::StyextMPCtx,
                          X::Vector{Float64}, Y::Float64,
                          theta_p::Vector{Float64},
                          sigma_p::Vector{Float64})::Vector{Float64}
    has_pole(ctx.flow) || return [0.0]
    _, df_dX = fene_peterlin(X[1], ctx.Q0_sq, ctx.Qinf_sq)
    pstar = pole_mode_index(ctx.flow, theta_p)
    # dθ_p*/dX = df/dX · 3 · σ_p*
    dtheta_dX = df_dX * 3.0 * sigma_p[pstar]
    return [dtheta_dX * dwi_pole_dtheta(ctx.flow)]
end

# ── Weight vectors per closure model ─────────────────────────────────────────

@inline function _weights(ctx::StyextMPCtx)
    wY = ctx.Sp
    wX = ctx.Na == 1 ? fill(1.0 / ctx.Nm, ctx.Nm) : Float64[]
    return wX, wY
end

# ── PAC residual + Jacobian ───────────────────────────────────────────────────

"""
    rj_pac(ctx, z, zprev, zpred, t_prev) -> (F, J, info)

PAC residual and Jacobian in the μ continuation coordinate.

z = ztilde = [X(1:Na); Y; μ]  (length Na+2)

Residuals:
  FX(1:Na) : wX · T_p − X   (X-closure, only when Na>0)
  FY       : wY · T_p − Y   (E²-closure)
  F_arc    : ([Y;μ] − [Y_pred;μ_pred])ᵀ t̂   (arc constraint, 2-D)

info : NamedTuple with Wi, Wi_pole, min_denom, mu.
"""
function rj_pac(ctx   :: StyextMPCtx,
                z     :: AbstractVector{Float64},
                zprev :: AbstractVector{Float64},
                zpred :: AbstractVector{Float64},
                t_prev :: AbstractVector{Float64})

    Na  = ctx.Na
    X, Y_raw, mu = unpack_ztilde(z, Na)
    Y = _clamp_Y(Y_raw, ctx)

    wX, wY = _weights(ctx)

    # ── θ_p / σ_p at (X, Y) ──────────────────────────────────────────────────
    theta_p, sigma_p, _ = _eval_ts!(ctx, X, Y)
    Wi_pole = wi_pole(ctx.flow, theta_p)

    # ── Wi_pole derivatives ───────────────────────────────────────────────────
    dWi_pole_dY = _dWi_pole_dY_fd(ctx, X, Y)
    dWi_pole_dX = Na > 0 ? _dWi_pole_dX_PM(ctx, X, Y, theta_p, sigma_p) : Float64[]

    # ── μ → Wi chain-rule ────────────────────────────────────────────────────
    Wi, dWi_dmu, dWi_dWp = wi_map(ctx.flow, mu, Wi_pole)
    dWi_dY  = dWi_dWp * dWi_pole_dY
    dWi_dX  = Na > 0 ? dWi_dWp .* dWi_pole_dX : Float64[]

    # ── Weighted closure sums + all partials ──────────────────────────────────
    SsumY, dSsumY_dX, dSsumY_dY, dSsumY_dWi, min_d =
        _Tp_weighted_all(ctx, X, Y, Wi, wY, theta_p, sigma_p)

    if Na > 0
        SsumX, dSsumX_dX, dSsumX_dY, dSsumX_dWi, _ =
            _Tp_weighted_all(ctx, X, Y, Wi, wX, theta_p, sigma_p)
    end

    # ── Residuals ─────────────────────────────────────────────────────────────
    FY = SsumY - Y
    FX = Na > 0 ? SsumX - X[1] : 0.0   # scalar (Na=1) or unused

    # ── Arc constraint (projected to Y–μ plane) ───────────────────────────────
    zp2 = SVector2(zpred[Na+1], zpred[Na+2])
    t2v = SVector2(t_prev[Na+1], t_prev[Na+2])
    n2  = _sv2_norm(t2v)
    t2  = n2 < 1e-8 ? SVector2(1.0, 0.0) : t2v / n2
    Farc = _sv2_dot(SVector2(Y, mu) - zp2, t2)

    # ── Assemble F and J ──────────────────────────────────────────────────────
    nz = Na + 2
    F  = Vector{Float64}(undef, nz)
    J  = zeros(Float64, nz, nz)

    if Na > 0   # FENE-PM: 3×3
        colX  = 1;    colY  = 2;  colMu = 3   # (Na=1 specific)
        rowX  = 1;    rowY  = 2;  rowArc = 3

        F[rowX]   = FX
        F[rowY]   = FY
        F[rowArc] = Farc

        J[rowX, colX]  = dSsumX_dX[1] + dSsumX_dWi * dWi_dX[1] - 1.0
        J[rowX, colY]  = dSsumX_dY    + dSsumX_dWi * dWi_dY
        J[rowX, colMu] = dSsumX_dWi   * dWi_dmu

        J[rowY, colX]  = dSsumY_dX[1] + dSsumY_dWi * dWi_dX[1]
        J[rowY, colY]  = dSsumY_dY    + dSsumY_dWi * dWi_dY   - 1.0
        J[rowY, colMu] = dSsumY_dWi   * dWi_dmu

        J[rowArc, colX]  = 0.0
        J[rowArc, colY]  = t2[1]
        J[rowArc, colMu] = t2[2]

    else        # FENE-PME / Hookean: 2×2
        F[1] = FY
        F[2] = Farc

        J[1, 1] = dSsumY_dY    + dSsumY_dWi * dWi_dY   - 1.0
        J[1, 2] = dSsumY_dWi   * dWi_dmu
        J[2, 1] = t2[1]
        J[2, 2] = t2[2]
    end

    info = (Wi=Wi, Wi_pole=Wi_pole, min_denom=min_d, mu=mu)
    return F, J, info
end

# ── Fixed-Y residual + Jacobian ───────────────────────────────────────────────

"""
    rj_fixed_Y(ctx, z, Y_fixed) -> (F, J, info)

Fixed-Y Newton: solve for (X, μ) with Y pinned to Y_fixed.

z = [X(1:Na); Y_slot; μ]   (Y_slot is overwritten internally)
F = [FX(1:Na); FY]  (length Na+1)
J is (Na+1)×(Na+1) wrt [X(1:Na); μ].
"""
function rj_fixed_Y(ctx    :: StyextMPCtx,
                    z      :: AbstractVector{Float64},
                    Y_fixed :: Float64)

    Na  = ctx.Na
    X, _, mu = unpack_ztilde(z, Na)
    Y = _clamp_Y(Y_fixed, ctx)

    wX, wY = _weights(ctx)

    # ── θ_p / σ_p at (X, Y_fixed) ────────────────────────────────────────────
    theta_p, sigma_p, _ = _eval_ts!(ctx, X, Y)
    Wi_pole = wi_pole(ctx.flow, theta_p)

    dWi_pole_dX = Na > 0 ? _dWi_pole_dX_PM(ctx, X, Y, theta_p, sigma_p) : Float64[]

    Wi, dWi_dmu, dWi_dWp = wi_map(ctx.flow, mu, Wi_pole)
    dWi_dX  = Na > 0 ? dWi_dWp .* dWi_pole_dX : Float64[]

    # ── Closure sums (no dY needed for fixed-Y Newton) ────────────────────────
    SsumY, dSsumY_dX, _, dSsumY_dWi, min_d =
        _Tp_weighted_all(ctx, X, Y, Wi, wY, theta_p, sigma_p)
    if Na > 0
        SsumX, dSsumX_dX, _, dSsumX_dWi, _ =
            _Tp_weighted_all(ctx, X, Y, Wi, wX, theta_p, sigma_p)
    end

    # ── Residuals ─────────────────────────────────────────────────────────────
    nF = Na + 1
    F  = Vector{Float64}(undef, nF)
    J  = zeros(Float64, nF, nF)

    if Na > 0   # FENE-PM: 2×2 (in [X; mu] space)
        F[1]     = SsumX - X[1]   # FX (Na=1: scalar)
        F[Na+1]  = SsumY - Y

        colX  = 1;  colMu = Na+1   # = 2 for Na=1

        J[1,   colX]  = dSsumX_dX[1] + dSsumX_dWi * dWi_dX[1] - 1.0
        J[1,   colMu] = dSsumX_dWi   * dWi_dmu
        J[Na+1, colX]  = dSsumY_dX[1] + dSsumY_dWi * dWi_dX[1]
        J[Na+1, colMu] = dSsumY_dWi   * dWi_dmu

    else        # FENE-PME / Hookean: 1×1 (in [mu] space)
        F[1]    = SsumY - Y
        J[1, 1] = dSsumY_dWi * dWi_dmu
    end

    info = (Wi=Wi, Wi_pole=Wi_pole, min_denom=min_d, mu=mu)
    return F, J, info
end

# ── Fixed-Wi residual + Jacobian ──────────────────────────────────────────────

"""
    rj_fixed_Wi(ctx, z, Wi) -> (F, J, info)

Fixed-Wi Newton: solve the closure equations for the state Z = (X, Y) at a
prescribed Wi (direct Wi sweeps; the natural mode for pole-free flows such as
simple shear, where the report proves a unique steady state at every Wi).

z = [X(1:Na); Y]        (length Na+1)
F = [FX(1:Na); FY]      (length Na+1)
J is (Na+1)×(Na+1) wrt [X(1:Na); Y].

For flows with a pole, Wi ≥ Wi_pole(Z) is inadmissible: `info.min_denom ≤ 0`
and the Newton driver aborts (`min_denom_fail`).
"""
function rj_fixed_Wi(ctx :: StyextMPCtx,
                     z   :: AbstractVector{Float64},
                     Wi  :: Float64)
    Na = ctx.Na
    length(z) == Na + 1 || error("rj_fixed_Wi: length(z)=$(length(z)) ≠ Na+1=$(Na+1)")
    X = Na > 0 ? Vector{Float64}(z[1:Na]) : Float64[]
    Y = _clamp_Y(Float64(z[Na+1]), ctx)

    wX, wY = _weights(ctx)

    theta_p, sigma_p, _ = _eval_ts!(ctx, X, Y)
    Wi_pole = wi_pole(ctx.flow, theta_p)

    SsumY, dSsumY_dX, dSsumY_dY, _, min_d =
        _Tp_weighted_all(ctx, X, Y, Wi, wY, theta_p, sigma_p)
    if Na > 0
        SsumX, dSsumX_dX, dSsumX_dY, _, _ =
            _Tp_weighted_all(ctx, X, Y, Wi, wX, theta_p, sigma_p)
    end

    nF = Na + 1
    F  = Vector{Float64}(undef, nF)
    J  = zeros(Float64, nF, nF)

    if Na > 0   # FENE-PM: 2×2 in [X; Y]
        F[1]    = SsumX - X[1]
        F[Na+1] = SsumY - Y
        J[1,    1]    = dSsumX_dX[1] - 1.0
        J[1,    Na+1] = dSsumX_dY
        J[Na+1, 1]    = dSsumY_dX[1]
        J[Na+1, Na+1] = dSsumY_dY - 1.0
    else        # FENE-PME / Hookean: 1×1 in [Y]
        F[1]    = SsumY - Y
        J[1, 1] = dSsumY_dY - 1.0
    end

    mu = (min_d > 0.0 && (!(has_pole(ctx.flow)) || Wi < Wi_pole)) ?
             mu_from_wi(ctx.flow, Wi, Wi_pole) : NaN
    info = (Wi=Wi, Wi_pole=Wi_pole, min_denom=min_d, mu=mu)
    return F, J, info
end

# ── State observables ─────────────────────────────────────────────────────────

"""
    eval_state_obs(ctx, X, Y, mu) -> C2D2Obs

Compute canonical observables from a manifold point (X, Y, μ).

Extension (mirrors MATLAB's styext_state_obs.m):
  - etaE = -(τ_ext - τ_comp) / max(Wi/λ₀, ε)·λ₀
  - Tr   = etaE · c·k_B·T · λ₀ / (3 ηₛ)
  - shear fields (tau_xy, eta_p, Psi1, Psi2) are NaN.

Planar mixed flow (strain-axes components A_p, B_p, C_p, D_p from
`modal_tensors`, then rotated by 45° to the Fuller–Leal flow frame
v_x = (1+r)Wi·y, v_y = (1−r)Wi·x, which at r = 1 is the lab shear frame):
  - tau_xx, tau_yy, tau_zz, tau_xy : flow-frame τ/(c kBT), STYEXT sign
  - etaE1 = −(τ₁₁ − τ₂₂)/Wi, etaE2 = −(τ₁₁ − τ₃₃)/Wi  (strain frame; equals
    STYEXT planar extension α=(1,−1,0) at r = 0), Tr1/Tr2 likewise
  - eta_p = −τ_xy/Wi_shear  (≡ dissipation definition −τ:D/(2D:D))
  - Psi1  = −(τ_xx − τ_yy)/Wi_shear²,  Psi2 = −(τ_yy − τ_zz)/Wi_shear²
  with Wi_shear = 2·Wi = λ₀√(2D:D) = λ₀γ̇.  Psi2 ≡ 0 analytically at r = 1.

At equilibrium (Wi = 0, τ = 0): material functions are 0 (stored as -0.0,
matching MATLAB's eps floor).
"""
function eval_state_obs(ctx :: StyextMPCtx,
                        X   :: Vector{Float64},
                        Y   :: Float64,
                        mu  :: Float64)::C2D2Obs
    Y = _clamp_Y(Y, ctx)

    # --- θ, σ, fp at (X, Y) ---
    theta_p, sigma_p, fp = _eval_ts!(ctx, X, Y)
    Wi_pole  = wi_pole(ctx.flow, theta_p)
    Wi, _, _ = wi_map(ctx.flow, mu, Wi_pole)

    return _obs_from_state(ctx.flow, ctx, theta_p, sigma_p, fp, Wi, Y)
end

function _obs_from_state(f::PrincipalExtension, ctx::StyextMPCtx,
                         theta_p, sigma_p, fp, Wi::Float64, Y::Float64)::C2D2Obs
    M_diag = modal_tensors(f, theta_p, sigma_p, Wi)
    tau    = polymer_stress(M_diag, fp)

    # --- Extensional viscosity with MATLAB's eps-floor ---
    alphas_vec = [f.alphas[1], f.alphas[2], f.alphas[3]]
    i_ext  = argmax(alphas_vec)
    i_comp = setdiff(1:3, i_ext)
    taus   = (tau.xx, tau.yy, tau.zz)
    tau_e  = taus[i_ext]
    tau_c1 = taus[i_comp[1]]
    tau_c2 = taus[i_comp[2]]

    # strainRate = max(Wi/lambda0, ε)  [MATLAB: max(Wi/lambda0, eps)]
    strain_rate = max(Wi / ctx.lambda0, eps(Float64))
    etaE1 = -(tau_e - tau_c1) / (strain_rate * ctx.lambda0)
    etaE2 = -(tau_e - tau_c2) / (strain_rate * ctx.lambda0)
    Tr1   = etaE1 * ctx.ckBT * ctx.lambda0 / (3.0 * ctx.etas)
    Tr2   = etaE2 * ctx.ckBT * ctx.lambda0 / (3.0 * ctx.etas)

    return C2D2Obs(Wi, Y, etaE1, etaE2, Tr1, Tr2,
                   tau.xx, tau.yy, tau.zz, :ok)
end

function _obs_from_state(f::PlanarMixedFlow, ctx::StyextMPCtx,
                         theta_p, sigma_p, fp, Wi::Float64, Y::Float64)::C2D2Obs
    A, B, C, D = modal_tensors(f, theta_p, sigma_p, Wi)

    # --- strain-axes stress τ/(c kBT) = −Σ_p(3 f_p M_p − I)  (STYEXT sign) ---
    t11 = 0.0; t22 = 0.0; t33 = 0.0; t12 = 0.0
    @inbounds for p in eachindex(A)
        t11 -= 3.0 * fp[p] * A[p] - 1.0
        t22 -= 3.0 * fp[p] * B[p] - 1.0
        t33 -= 3.0 * fp[p] * D[p] - 1.0
        t12 -= 3.0 * fp[p] * C[p]
    end

    # --- rotate 45° to the flow frame (lab shear frame at r = 1) ---
    half = 0.5 * (t11 + t22)
    txx  = half - t12
    tyy  = half + t12
    txy  = 0.5 * (t11 - t22)
    tzz  = t33

    # --- strain-frame planar extensional viscosities (STYEXT definition) ---
    strain_rate = max(Wi / ctx.lambda0, eps(Float64))
    etaE1 = -(t11 - t22) / (strain_rate * ctx.lambda0)
    etaE2 = -(t11 - t33) / (strain_rate * ctx.lambda0)
    Tr1   = etaE1 * ctx.ckBT * ctx.lambda0 / (3.0 * ctx.etas)
    Tr2   = etaE2 * ctx.ckBT * ctx.lambda0 / (3.0 * ctx.etas)

    # --- shear material functions on the shear-rate measure γ̇ = 2 Wi/λ₀ ---
    gdot_l = max(wi_shear(Wi), eps(Float64))      # λ₀γ̇ with eps floor
    eta_p  = -txy / gdot_l
    Psi1   = -(txx - tyy) / gdot_l^2
    Psi2   = -(tyy - tzz) / gdot_l^2

    return C2D2Obs(Wi, Y, etaE1, etaE2, Tr1, Tr2,
                   txx, tyy, tzz, :ok,
                   txy, eta_p, Psi1, Psi2)
end

# ── Tiny 2-vector to avoid allocations in arc computation ─────────────────────

struct SVector2
    x :: Float64
    y :: Float64
end
Base.:-(a::SVector2, b::SVector2) = SVector2(a.x-b.x, a.y-b.y)
Base.:/(a::SVector2, s::Float64)  = SVector2(a.x/s, a.y/s)
_sv2_dot(a::SVector2, b::SVector2) = a.x*b.x + a.y*b.y
_sv2_norm(a::SVector2) = sqrt(a.x^2 + a.y^2)
Base.getindex(a::SVector2, i::Int) = i == 1 ? a.x : a.y

"""
    eval_modal_tensors(ctx, X, Y, mu) -> NamedTuple

Closure coefficients and steady modal tensors at a manifold point:
`(theta_p, sigma_p, fp, Wi, Wi_pole, M)` with `M = modal_tensors(ctx.flow, …)`
(Nm×3 diagonal for extension; `(A, B, C, D)` strain-axes tuple for PMF).
Diagnostic helper for per-mode output (orientation angles, mode traces).
"""
function eval_modal_tensors(ctx::StyextMPCtx, X::Vector{Float64}, Y::Float64, mu::Float64)
    Y = _clamp_Y(Y, ctx)
    theta_p, sigma_p, fp = _eval_ts!(ctx, X, Y)
    Wi_pole  = wi_pole(ctx.flow, theta_p)
    Wi, _, _ = wi_map(ctx.flow, mu, Wi_pole)
    return (theta_p = theta_p, sigma_p = sigma_p, fp = fp, Wi = Wi, Wi_pole = Wi_pole,
            M = modal_tensors(ctx.flow, theta_p, sigma_p, Wi))
end

"""
    linear_limit_shear(ctx) -> (eta0, Psi10)

Zero-shear-rate polymer viscosity and first normal-stress coefficient from the
equilibrium closure coefficients (X = Q₀², Y = E₀²; for the blob spectrum these
depend on φ). Per mode, in units of c kBT λ₀ and c kBT λ₀²:
    η₀    = Σ_p 3 f_p σ_p / θ_p²    (= Σ λ̂_p/f_p when Λ = 1)
    Ψ₁,₀  = Σ_p 6 f_p σ_p / θ_p³    (= Σ 2 λ̂_p²/f_p²)
These are the Wi → 0 limits of `eta_p` and `Psi1` in `eval_state_obs` for a
PlanarMixedFlow at r = 1, and are flow-independent (pure LVE).
"""
function linear_limit_shear(ctx::StyextMPCtx)
    X0 = ctx.Na > 0 ? fill(ctx.Q0_sq, ctx.Na) : Float64[]
    theta_p, sigma_p, fp = _eval_ts!(ctx, X0, ctx.Ymin)
    eta0 = 0.0; Psi10 = 0.0
    @inbounds for p in eachindex(theta_p)
        eta0  += 3.0 * fp[p] * sigma_p[p] / theta_p[p]^2
        Psi10 += 6.0 * fp[p] * sigma_p[p] / theta_p[p]^3
    end
    return eta0, Psi10
end

"""
    get_Wi_pole(ctx, X, Y) -> Float64

Return Wi_pole at the given state (Inf for pole-free flows, e.g. PMF r ≥ 1).
"""
function get_Wi_pole(ctx::StyextMPCtx, X::Vector{Float64}, Y::Float64)::Float64
    theta_p, _, _ = _eval_ts!(ctx, X, _clamp_Y(Y, ctx))
    return wi_pole(ctx.flow, theta_p)
end

end # module StyextModelPack
