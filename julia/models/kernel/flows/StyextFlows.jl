"""
    StyextFlows.jl

Steady homogeneous-flow kinematics and continuation-coordinate maps for the
steady PAC solver (STYEXT and, since 2026-10-16, planar mixed flows / PMF).

Mirrors MATLAB's
  styext_Wi_from_mu.m / styext_mu_from_Wi.m
and extends them with a flow-kinematics type.

Flow types
----------
  PrincipalExtension(alphas)   — κ = Wi·diag(α₁, α₂, α₃); the classic STYEXT
                                 flow (uniaxial / biaxial / planar extension).
  PlanarMixedFlow(r)           — κ = Wi·κ̂(r), κ̂(r) = [1 r 0; −r −1 0; 0 0 0]
                                 (tech report App. "Planar mixed flows",
                                 `a:M2CD2-ShrFlow`). r = vorticity/strain-rate
                                 ratio: r = 0 planar extension, r = 1 simple
                                 shear, r > 1 rotation-dominated.

Wi convention for PlanarMixedFlow
---------------------------------
`Wi` is the report-appendix Wi: λ₀ times the principal strain rate (the ±1
eigenvalues of κ̂ are unit). The shear-rate measure γ̇ = √(2 D:D) is
r-independent and equals 2·Wi/λ₀, so the *conventional shear Weissenberg
number* λ₀γ̇ is `Wi_shear = 2·Wi`. Both are reported; see `wi_shear`.

Continuation coordinate μ
-------------------------
Flows with a finite-Wi pole (extension; PMF with r < 1) use the logistic map

    Wi = Wi_pole · (1 − exp(−μ)),      μ ∈ (0, ∞) ↔ Wi ∈ (0, Wi_pole)

Flows without a pole (PMF with r ≥ 1, e.g. simple shear) use

    Wi = exp(μ) − 1,                   μ ∈ (0, ∞) ↔ Wi ∈ (0, ∞)

and `wi_pole` returns Inf for them. `wi_map` returns Wi together with the
two partial derivatives the PAC Jacobian needs.

Public API
----------
  AbstractSteadyFlow, PrincipalExtension, PlanarMixedFlow
  has_pole(flow)                        -> Bool
  wi_pole(flow, theta_p)                -> Wi_pole (Inf if none)
  pole_mode_index(flow, theta_p)        -> index p* whose θ sets the pole
  wi_map(flow, mu, Wi_pole)             -> (Wi, dWi/dμ, dWi/dWi_pole)
  mu_from_wi(flow, Wi, Wi_pole)         -> μ
  wi_shear(Wi)                          -> 2·Wi   (λ₀√(2D:D) for PMF)
  Wi_from_mu(mu, Wi_pole), mu_from_Wi(Wi, Wi_pole)   — legacy logistic map
"""
module StyextFlows

export AbstractSteadyFlow, PrincipalExtension, PlanarMixedFlow,
       has_pole, wi_pole, pole_mode_index, wi_map, mu_from_wi, wi_shear,
       Wi_from_mu, mu_from_Wi

# ── Flow kinematics types ─────────────────────────────────────────────────────

abstract type AbstractSteadyFlow end

"""
    PrincipalExtension(alphas)

Shear-free extension with principal rates α = (αx, αy, αz), κ = Wi·diag(α).
Requires max(α) > 0. `amax` is the extensional-axis rate; Wi_pole = θ_p*/(2 amax).
"""
struct PrincipalExtension <: AbstractSteadyFlow
    alphas :: NTuple{3, Float64}
    amax   :: Float64
end

function PrincipalExtension(alphas::NTuple{3, Float64})
    amax = maximum(alphas)
    amax > 0 || error("PrincipalExtension: amax = $amax ≤ 0; alphas must have a positive element")
    return PrincipalExtension(alphas, amax)
end
PrincipalExtension(alphas::AbstractVector) =
    PrincipalExtension((Float64(alphas[1]), Float64(alphas[2]), Float64(alphas[3])))

"""
    PlanarMixedFlow(r)

Planar mixed flow κ = Wi·κ̂(r), κ̂(r) = [1 r 0; −r −1 0; 0 0 0], r ≥ 0.
In the strain-axes frame D = Wi·diag(1, −1, 0) for every r; the vorticity is
2 r Wi. Rotating by 45° gives the Fuller–Leal "flow frame"
κ_lab = Wi·[0 (1+r) 0; (1−r) 0 0; 0 0 0], i.e. v_x = (1+r)Wi·y, v_y = (1−r)Wi·x,
which for r = 1 is simple shear v_x = γ̇ y with γ̇ = 2 Wi.
"""
struct PlanarMixedFlow <: AbstractSteadyFlow
    r :: Float64
    function PlanarMixedFlow(r::Real)
        isfinite(r) && r >= 0 || error("PlanarMixedFlow: r must be finite and ≥ 0, got $r")
        new(Float64(r))
    end
end

# ── Pole bookkeeping ──────────────────────────────────────────────────────────

has_pole(::PrincipalExtension) = true
has_pole(f::PlanarMixedFlow)   = f.r < 1.0

"""
    pole_mode_index(flow, theta_p) -> Int

Index of the mode whose θ_p sets Wi_pole. Extension keeps the historical
STYEXT choice p* = 1 (bit-identical to the pre-PMF code). PMF uses argmin θ_p
(the report's θ_min); for monotone spectra (Rouse, Zimm, blob) this is p = 1
as well.
"""
pole_mode_index(::PrincipalExtension, theta_p::AbstractVector{Float64}) = 1
pole_mode_index(::PlanarMixedFlow,    theta_p::AbstractVector{Float64}) = argmin(theta_p)

"""
    wi_pole(flow, theta_p) -> Float64

Finite-Wi pole of the steady modal tensors at the closure state that produced
`theta_p`:
  extension : θ_p*/(2 amax)
  PMF r<1   : θ_p*/(2√(1−r²))      (report `eq:pmf-pole`)
  PMF r≥1   : Inf (no kinematic pole)
"""
function wi_pole(f::PrincipalExtension, theta_p::AbstractVector{Float64})::Float64
    return theta_p[pole_mode_index(f, theta_p)] / (2.0 * f.amax)
end
function wi_pole(f::PlanarMixedFlow, theta_p::AbstractVector{Float64})::Float64
    has_pole(f) || return Inf
    return theta_p[pole_mode_index(f, theta_p)] / (2.0 * sqrt(1.0 - f.r^2))
end

"""dWi_pole/dθ_p* — the scalar factor multiplying dθ_p*/d(X or Y)."""
dwi_pole_dtheta(f::PrincipalExtension) = 1.0 / (2.0 * f.amax)
dwi_pole_dtheta(f::PlanarMixedFlow)    = has_pole(f) ? 1.0 / (2.0 * sqrt(1.0 - f.r^2)) : 0.0

# ── μ ↔ Wi maps ──────────────────────────────────────────────────────────────

"""
    wi_map(flow, mu, Wi_pole) -> (Wi, dWi_dmu, dWi_dWipole)

Continuation map μ → Wi with the partials the PAC Jacobian needs.
Logistic map when the flow has a pole; Wi = exp(μ) − 1 otherwise.
"""
@inline function wi_map(flow::AbstractSteadyFlow, mu::Float64, Wi_pole::Float64)
    if has_pole(flow)
        em = exp(-mu)
        return Wi_pole * (1.0 - em), Wi_pole * em, 1.0 - em
    else
        e = exp(mu)
        return e - 1.0, e, 0.0
    end
end

"""
    mu_from_wi(flow, Wi, Wi_pole) -> μ

Inverse of `wi_map` (Wi_pole ignored for pole-free flows).
"""
function mu_from_wi(flow::AbstractSteadyFlow, Wi::Float64, Wi_pole::Float64)::Float64
    if has_pole(flow)
        return mu_from_Wi(Wi, Wi_pole)
    else
        Wi >= 0.0 || error("mu_from_wi: Wi must be ≥ 0 for a pole-free flow, got $Wi")
        return log1p(Wi)
    end
end

"""
    wi_shear(Wi) -> 2·Wi

Conventional shear Weissenberg number λ₀γ̇ = λ₀√(2 D:D) for a PlanarMixedFlow
at report-appendix Wi (r-independent).
"""
@inline wi_shear(Wi::Float64) = 2.0 * Wi

# ── Legacy logistic map (STYEXT) ──────────────────────────────────────────────

"""
    Wi_from_mu(mu, Wi_pole) -> Wi

Map continuation coordinate μ → Wi.

    Wi = Wi_pole · (1 − exp(−μ))
"""
@inline function Wi_from_mu(mu::Float64, Wi_pole::Float64)::Float64
    return Wi_pole * (1.0 - exp(-mu))
end

"""
    mu_from_Wi(Wi, Wi_pole) -> mu

Map Wi → continuation coordinate μ, guarding against Wi → Wi_pole.

    μ = −log(1 − Wi_safe / Wi_pole)

Wi is clamped to (1 − 1e-12) · Wi_pole to avoid log(0).
"""
function mu_from_Wi(Wi::Float64, Wi_pole::Float64)::Float64
    Wi_pole > 0.0 || error("mu_from_Wi: Wi_pole must be positive, got $Wi_pole")
    Wi_safe = min(Wi, (1.0 - 1e-12) * Wi_pole)
    ratio   = max(0.0, Wi_safe / Wi_pole)
    return -log(1.0 - ratio)
end

end # module StyextFlows
