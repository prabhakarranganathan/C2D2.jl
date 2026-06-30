"""
    StyextModelPack.jl

Model-pack for the steady-extension PAC solver.

Mirrors MATLAB's
  styext_modelpack_core.m / styext_FENE_PM_modelpack.m / styext_FENE_PME_modelpack.m

Provides the PAC and fixed-Y residual/Jacobians that drive the continuation.

Normalisation convention
------------------------
MATLAB's coeffs_relaxation uses r = √Y / E₀ as the drag/spectrum LUT argument
(where E₀ = √Nm for Q₀²=1).  Julia's CoeffsKernel uses E = Z.E directly.
To match MATLAB for Nm > 1 we set Z.E = √Y / E₀ before calling eval_coeffs!.

State-vector layout
-------------------
FENE-PME (Na=0)  :  ztilde = [Y; μ]              length 2
FENE-PM  (Na=1)  :  ztilde = [X; Y; μ]           length 3
FENE-P   (Na=Nm) :  ztilde = [X₁…Xₙ; Y; μ]       length Nm+2

Public API
----------
  StyextMPCtx         — assembled model-pack context
  build_mp_ctx(...)   — constructor
  pack_ztilde         — (X, Y, μ) → vector
  unpack_ztilde       — vector → (X, Y, μ)
  rj_pac              — PAC residual + Jacobian
  rj_fixed_Y          — fixed-Y residual + Jacobian
  eval_state_obs      — C2D2Obs from current (X, Y, μ)
"""
module StyextModelPack

using LinearAlgebra: norm, dot

using ..Fene:       AbstractFeneModel, FenePME, FenePM, FeneP, fene_peterlin
using ..CoeffsRelax: CoeffsKernel, ClosureZ, eval_coeffs!
using ..StyextFlows: Wi_from_mu
using ..Stress:     polymer_stress
using ..StateObs:   C2D2Obs

export StyextMPCtx, build_mp_ctx,
       pack_ztilde, unpack_ztilde,
       rj_pac, rj_fixed_Y, eval_state_obs, get_Wi_pole

# ── Context struct ────────────────────────────────────────────────────────────

"""
    StyextMPCtx

Assembled context for the STYEXT model pack.

Fields
------
kernel   : CoeffsKernel (FENE + drag LUT + spectrum; build once, reuse)
Sp       : Nm-vector of Rouse end-to-end weights  (Y-closure weights wY)
alphas   : NTuple{3,Float64} — principal flow-rate ratios (αx, αy, αz)
amax     : maximum of alphas (extensional axis rate)
Na       : number of X auxiliaries (0=PME, 1=PM, Nm=P)
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
end

"""
    build_mp_ctx(kernel, Sp, alphas, Na, Nm, E0_sq, Einf_sq,
                 Q0_sq, Qinf_sq, lambda0, etas, ckBT) -> StyextMPCtx
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
                      ckBT     :: Float64)::StyextMPCtx
    amax = maximum(alphas)
    amax > 0 || error("build_mp_ctx: amax = $amax ≤ 0; alphas must have a positive element")
    E0 = sqrt(E0_sq)
    return StyextMPCtx(kernel, Sp, alphas, amax,
                       Na, Nm, E0_sq, Einf_sq,
                       E0, Q0_sq, Qinf_sq,
                       lambda0, etas, ckBT, 1e-6)
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

# ── D, inv-D and T_p ─────────────────────────────────────────────────────────

"""
    _Tp!(out, theta_p, sigma_p, alphas, Wi) -> (D, invD, Tsum_invD2)

Fill T_p vector in-place: T_p = σ_p · Σ_α 1/D_pα.
Also returns D (Nm×3) and invD (Nm×3) for Jacobian use,
and a vector sum_invD2 (Nm) = Σ_α 1/D_pα².
"""
function _Tp!(Tp_out::Vector{Float64},
               theta_p::Vector{Float64}, sigma_p::Vector{Float64},
               alphas::NTuple{3, Float64}, Wi::Float64)
    Nm = length(theta_p)
    D        = Matrix{Float64}(undef, Nm, 3)
    invD     = Matrix{Float64}(undef, Nm, 3)
    sum_invD  = Vector{Float64}(undef, Nm)
    sum_invD2 = Vector{Float64}(undef, Nm)

    @inbounds for p in 1:Nm
        tp = theta_p[p]
        s1 = 0.0;  s2 = 0.0
        for α in 1:3
            d        = tp - 2.0 * Wi * alphas[α]
            id       = 1.0 / d
            D[p, α]  = d
            invD[p, α] = id
            s1 += id
            s2 += id * id
        end
        sum_invD[p]  = s1
        sum_invD2[p] = s2
        Tp_out[p]    = sigma_p[p] * s1
    end
    return D, invD, sum_invD, sum_invD2
end

# ── Weighted T_p sum with all partial derivatives ─────────────────────────────

"""
    _Tp_weighted_all(ctx, X, Y, Wi, w, theta_p, sigma_p)
                   -> (Tsum, dTsum_dX, dTsum_dY, dTsum_dWi)

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
    Tp = Vector{Float64}(undef, Nm)
    D, invD, _, sum_invD2 = _Tp!(Tp, theta_p, sigma_p, ctx.alphas, Wi)

    Tsum = dot(w, Tp)

    # ── dTsum/dWi (analytic) ─────────────────────────────────────────────────
    # dT_p/dWi = σ_p · 2 · Σ_α α_α / D_pα²
    dTsum_dWi = 0.0
    @inbounds for p in 1:Nm
        s = 0.0
        for α in 1:3
            s += ctx.alphas[α] * invD[p, α]^2
        end
        dTsum_dWi += w[p] * sigma_p[p] * 2.0 * s
    end

    # ── dTsum/dX (analytic; Na=0 → empty; Na=1 → scalar) ────────────────────
    dTsum_dX = zeros(Na)
    if Na == 1
        # dθ_p/dX = df/dX · 3·σ_p  (FENE-PM: f depends only on X[1])
        _, df_dX = fene_peterlin(X[1], ctx.Q0_sq, ctx.Qinf_sq)
        @inbounds for p in 1:Nm
            dtheta_dX_p = df_dX * 3.0 * sigma_p[p]
            dT_dX_p     = -sigma_p[p] * sum_invD2[p] * dtheta_dX_p
            dTsum_dX[1] += w[p] * dT_dX_p
        end
    elseif Na > 1
        error("_Tp_weighted_all: analytic dX only implemented for Na=0 and Na=1 (FENE-P not yet supported)")
    end

    # ── dTsum/dY (FD, Wi held fixed) ─────────────────────────────────────────
    h = ctx.h_fd
    f_Tsum = (YY::Float64) -> begin
        tp2, sp2, _ = _eval_ts!(ctx, X, _clamp_Y(YY, ctx))
        Tp2 = Vector{Float64}(undef, Nm)
        _Tp!(Tp2, tp2, sp2, ctx.alphas, Wi)
        dot(w, Tp2)
    end
    dTsum_dY = _fd1_2nd(f_Tsum, Y, ctx.Ymin, ctx.Ymax; h=h)

    return Tsum, dTsum_dX, dTsum_dY, dTsum_dWi
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

"""dWi_pole/dY via FD at (X, Y) with Wi_pole = θ₁(X,Y)/(2 amax)."""
function _dWi_pole_dY_fd(ctx::StyextMPCtx, X::Vector{Float64}, Y::Float64)::Float64
    f = (YY::Float64) -> begin
        tp, _, _ = _eval_ts!(ctx, X, _clamp_Y(YY, ctx))
        tp[1] / (2.0 * ctx.amax)
    end
    return _fd1_2nd(f, Y, ctx.Ymin, ctx.Ymax; h=ctx.h_fd)
end

"""dWi_pole/dX analytically for FENE-PM (Na=1): = (dθ₁/dX[1]) / (2 amax)."""
function _dWi_pole_dX_PM(ctx::StyextMPCtx,
                          X::Vector{Float64}, Y::Float64,
                          sigma_p::Vector{Float64})::Vector{Float64}
    _, df_dX = fene_peterlin(X[1], ctx.Q0_sq, ctx.Qinf_sq)
    # dθ₁/dX = df/dX · 3 · σ₁
    dtheta1_dX = df_dX * 3.0 * sigma_p[1]
    return [dtheta1_dX / (2.0 * ctx.amax)]
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
    Nm  = ctx.Nm
    X, Y_raw, mu = unpack_ztilde(z, Na)
    Y = _clamp_Y(Y_raw, ctx)

    wX, wY = _weights(ctx)

    # ── θ_p / σ_p at (X, Y) ──────────────────────────────────────────────────
    theta_p, sigma_p, _ = _eval_ts!(ctx, X, Y)
    Wi_pole = theta_p[1] / (2.0 * ctx.amax)

    # ── Wi_pole derivatives ───────────────────────────────────────────────────
    dWi_pole_dY = _dWi_pole_dY_fd(ctx, X, Y)
    dWi_pole_dX = Na > 0 ? _dWi_pole_dX_PM(ctx, X, Y, sigma_p) : Float64[]

    # ── μ → Wi chain-rule ────────────────────────────────────────────────────
    Wi      = Wi_from_mu(mu, Wi_pole)
    dWi_dmu = Wi_pole * exp(-mu)
    em      = 1.0 - exp(-mu)           # 1 − e^{−μ}
    dWi_dY  = em * dWi_pole_dY
    dWi_dX  = Na > 0 ? em .* dWi_pole_dX : Float64[]

    # ── Weighted closure sums + all partials ──────────────────────────────────
    SsumY, dSsumY_dX, dSsumY_dY, dSsumY_dWi =
        _Tp_weighted_all(ctx, X, Y, Wi, wY, theta_p, sigma_p)

    if Na > 0
        SsumX, dSsumX_dX, dSsumX_dY, dSsumX_dWi =
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

    else        # FENE-PME: 2×2
        F[1] = FY
        F[2] = Farc

        J[1, 1] = dSsumY_dY    + dSsumY_dWi * dWi_dY   - 1.0
        J[1, 2] = dSsumY_dWi   * dWi_dmu
        J[2, 1] = t2[1]
        J[2, 2] = t2[2]
    end

    # ── min_denom diagnostic ──────────────────────────────────────────────────
    Tp_diag = Vector{Float64}(undef, Nm)
    D_diag, _, _, _ = _Tp!(Tp_diag, theta_p, sigma_p, ctx.alphas, Wi)
    min_d = minimum(D_diag)

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
    Nm  = ctx.Nm
    X, _, mu = unpack_ztilde(z, Na)
    Y = _clamp_Y(Y_fixed, ctx)

    wX, wY = _weights(ctx)

    # ── θ_p / σ_p at (X, Y_fixed) ────────────────────────────────────────────
    theta_p, sigma_p, _ = _eval_ts!(ctx, X, Y)
    Wi_pole = theta_p[1] / (2.0 * ctx.amax)

    dWi_pole_dX = Na > 0 ? _dWi_pole_dX_PM(ctx, X, Y, sigma_p) : Float64[]

    Wi      = Wi_from_mu(mu, Wi_pole)
    dWi_dmu = Wi_pole * exp(-mu)
    em      = 1.0 - exp(-mu)
    dWi_dX  = Na > 0 ? em .* dWi_pole_dX : Float64[]

    # ── Closure sums (no dY needed for fixed-Y Newton) ────────────────────────
    # Reuse _Tp_weighted_all but the dY component is not used in J here.
    SsumY, dSsumY_dX, _, dSsumY_dWi =
        _Tp_weighted_all(ctx, X, Y, Wi, wY, theta_p, sigma_p)
    if Na > 0
        SsumX, dSsumX_dX, _, dSsumX_dWi =
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

    else        # FENE-PME: 1×1 (in [mu] space)
        F[1]    = SsumY - Y
        J[1, 1] = dSsumY_dWi * dWi_dmu
    end

    # ── Diagnostics ───────────────────────────────────────────────────────────
    Tp_diag = Vector{Float64}(undef, Nm)
    D_diag, _, _, _ = _Tp!(Tp_diag, theta_p, sigma_p, ctx.alphas, Wi)
    min_d = minimum(D_diag)

    info = (Wi=Wi, Wi_pole=Wi_pole, min_denom=min_d, mu=mu)
    return F, J, info
end

# ── State observables ─────────────────────────────────────────────────────────

"""
    eval_state_obs(ctx, X, Y, mu) -> C2D2Obs

Compute canonical observables from a manifold point (X, Y, μ).

Mirrors MATLAB's styext_state_obs.m:
  - etaE = -(τ_ext - τ_comp) / max(Wi/λ₀, ε)·λ₀
  - Tr   = etaE · c·k_B·T · λ₀ / (3 ηₛ)

At equilibrium (Wi = 0, τ = 0): etaE = 0 (→ stored as -0.0, matching MATLAB).
"""
function eval_state_obs(ctx :: StyextMPCtx,
                        X   :: Vector{Float64},
                        Y   :: Float64,
                        mu  :: Float64)::C2D2Obs
    Y = _clamp_Y(Y, ctx)

    # --- θ, σ, fp at (X, Y) ---
    theta_p, sigma_p, fp = _eval_ts!(ctx, X, Y)
    Wi_pole = theta_p[1] / (2.0 * ctx.amax)
    Wi      = Wi_from_mu(mu, Wi_pole)

    # --- M_diag = σ_p / D_pα  (Nm×3) ---
    Nm = ctx.Nm
    Tp = Vector{Float64}(undef, Nm)
    D, invD, _, _ = _Tp!(Tp, theta_p, sigma_p, ctx.alphas, Wi)
    M_diag = Matrix{Float64}(undef, Nm, 3)
    @inbounds for p in 1:Nm
        for α in 1:3
            M_diag[p, α] = sigma_p[p] * invD[p, α]
        end
    end

    # --- Polymer stress ---
    tau = polymer_stress(M_diag, fp)

    # --- Extensional viscosity with MATLAB's eps-floor ---
    alphas_vec = [ctx.alphas[1], ctx.alphas[2], ctx.alphas[3]]
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
    get_Wi_pole(ctx, X, Y) -> Float64

Return Wi_pole = θ₁(X,Y) / (2 amax) at the given state.
"""
function get_Wi_pole(ctx::StyextMPCtx, X::Vector{Float64}, Y::Float64)::Float64
    theta_p, _, _ = _eval_ts!(ctx, X, _clamp_Y(Y, ctx))
    return theta_p[1] / (2.0 * ctx.amax)
end

end # module StyextModelPack
