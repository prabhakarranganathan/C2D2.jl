"""
    UsyextFlows.jl

Flow protocol structs and evaluators for the unsteady extension (USYEXT) solver:
how the Weissenberg number Wi = λ₁,₀ ε̇ is set at each instant (technical report
Sec. 4.5.1 for capillary thinning).

Three protocols are supported (TOML `[flow] kind`):
  ConstantWiFlow      — "constant": Wi fixed at a user-specified value (start-up of
                        steady extension).
  CaberFlow           — "caber": Wi from the one-dimensional viscocapillary stress
                        balance of a thinning filament (capillary-breakup extensional
                        rheometry, CaBER, in the viscous limit Oh ≫ 1).
  CaberInertialFlow   — "caber_inertial": the same balance in the inertiocapillary
                        limit Oh ≪ 1, with the Ohnesorge number as input.

Time unit: λ₁,₀.  The Deborah number De compares λ₁,₀ with the macroscopic flow
time: the viscocapillary time t_v = η_s R₀/γ for "caber" and the Rayleigh time
t_R = √(ρR₀³/γ) for "caber_inertial" (R₀ the initial filament radius, γ the
surface tension).

Public API
----------
  ConstantWiFlow, CaberFlow, CaberInertialFlow   — flow contexts
  eval_flow(flow, t, strain, N1_p)               — returns a NamedTuple
                                                   (Wi, De, R, psi, X, N1_p, K_til)
"""
module UsyextFlows

export ConstantWiFlow, CaberFlow, CaberInertialFlow, eval_flow

# ── Constant-Wi flow ──────────────────────────────────────────────────────────

"""
    ConstantWiFlow

Flow context for extension at constant Weissenberg number Wi = λ₁,₀ ε̇.

Fields
------
Wi_const : constant Wi
De       : Deborah number λ₁,₀/t_flow; only converts between strain and time
           in the output (t = strain·De/Wi) for this protocol
alphas   : NTuple{3} of principal flow-rate ratios (uniaxial: (−½, −½, 1))
i_ext    : index of the extensional axis (argmax alphas)
i_comp   : index of the primary compressive axis (argmin alphas)
"""
struct ConstantWiFlow
    Wi_const :: Float64
    De       :: Float64
    alphas   :: NTuple{3, Float64}
    i_ext    :: Int
    i_comp   :: Int
end

function ConstantWiFlow(De::Float64, Wi::Float64,
                        alphas::NTuple{3, Float64})::ConstantWiFlow
    i_ext  = argmax(collect(alphas))
    i_comp = argmin(collect(alphas))
    return ConstantWiFlow(Wi, De, alphas, i_ext, i_comp)
end

# ── CaBER flow ────────────────────────────────────────────────────────────────

"""
    CaberFlow

Flow context for capillary thinning (CaBER) in the viscocapillary limit Oh ≫ 1.

The one-dimensional stress balance of a thinning filament (report Sec. 4.5.1)
gives Wi from the current filament radius R = exp(−ε/2) (relative to R₀) and
the polymer first normal stress difference N1_p = (τ_zz − τ_rr)/(c k_BT):

    R   = exp(−ε/2)
    U_R = U₁ / De                          # polymer modulus over capillary pressure, c*k_BT R₀/γ
    ψ,X = _x_switch(Xv, Xe, U_R, φ, N1_p, R)
    Wi  = ⅓ [ (2X−1)/R · De  +  φ U₁ N1_p ]

Here `U_1` = U₁ = λ₁,₀ k_BT/(η_s E₀³) is the dimensionless group that compares the
polymer contribution to the viscosity with the solvent viscosity (→ 0.325·λ₁,₀/λ_Z
for a non-draining coil).  The product φ U₁ is written to the CaBER diagnostic
CSV as `K_til`.  X is the end-plate (geometry) factor of the thinning filament,
carried from its Newtonian value Xv to the elastic value Xe as the polymer stress
takes over the capillary pressure (`_x_switch`).

Fields
------
De       : Deborah number λ₁,₀/t_v, t_v = η_s R₀/γ the viscocapillary time
alphas   : NTuple{3} of principal flow-rate ratios
i_ext    : index of extensional axis (argmax alphas)
i_comp   : index of primary compressive axis
Xv       : viscous-limit X = 0.7127  (constant, Entov–Hinch / Papageorgiou)
Xe       : elastic-limit X = 1.5     (constant)
phi      : overlap ratio φ = c/c* as seen by the stress balance (φ/C_φ)
U_1      : the polymer prefactor U₁ defined above
"""
struct CaberFlow
    De     :: Float64
    alphas :: NTuple{3, Float64}
    i_ext  :: Int
    i_comp :: Int
    Xv     :: Float64
    Xe     :: Float64
    phi    :: Float64
    U_1    :: Float64
end

function CaberFlow(De::Float64, alphas::NTuple{3, Float64},
                   phi::Float64, U_1::Float64)::CaberFlow
    i_ext  = argmax(collect(alphas))
    i_comp = argmin(collect(alphas))
    return CaberFlow(De, alphas, i_ext, i_comp,
                     0.7127, 1.5,   # Xv, Xe (fixed constants)
                     phi, U_1)
end

# ── CaBER flow: inertio-capillary limit (Oh ≪ 1) ─────────────────────────────

"""
    CaberInertialFlow

Flow context for capillary thinning (CaBER) in the **inertiocapillary** limit
Oh ≪ 1 (report Sec. 4.5.1, "Inertiocapillary limit").  Companion to `CaberFlow`
(the viscocapillary limit Oh ≫ 1); the two closed-form limits are separate
structs so that the viscous path is left untouched.

The stress balance gives Wi from the current filament radius R = exp(−ε/2) and
the polymer first normal stress difference N1_p.  The polymer prefactor is
formed at every evaluation from the physical inputs:

    R       = exp(−ε/2)
    U_R_eff = Oh · U₁ / De_R                # = U₁/De_v, the polymer modulus over the capillary pressure
    ψ,X     = _x_switch(XR, Xe, U_R_eff, φ, N1_p, R)
    arg     = (2X−1)/R  +  φ U_R_eff N1_p   # capillary minus polymer normal stress
    Wi      = (2√2 · De_R / R) · √(max(arg, 0))

The Ohnesorge number Oh = η_s/√(ρ R₀ γ) is the input; U_R is derived from it.
U_R is not an independent constant: the relaxation time is carried by the
solvent (λ₁,₀ = U₁ η_s E₀³/k_BT), so U_R = c* k_BT R₀/γ = U₁/De_v = Oh·U₁/De_R
(De_v = De_R/Oh because t_v = Oh·t_R).  A De_R sweep at fixed Oh therefore has
U_R_eff ∝ 1/De_R by design: it traces the locus of one real fluid on one
device as R₀ changes.  Oh = 0 is the degenerate limit without polymer arrest
(the polymer term vanishes and the filament pinches off as a Newtonian
inviscid thread); arrest needs roughly Oh ≳ 1/N_K.  The diagnostic group
K̃ = φ U_R_eff is returned in the NamedTuple for the CaBER CSV.

Pure-solvent check (N1_p → 0): Wi = 2√2 De_R √(2X_R−1) R^(−3/2), i.e.
Wi ∝ R^(−1/β) with β = 2/3, and the pre-elastic prefactor H_R = √(8(2X_R−1))
(Wi₀ = H_R De_R at R = 1).  With the self-similar prefactor A ≈ 0.47 measured by
Gaillard et al., 2X_R−1 = 2A³/9 ≈ 0.0231, so X_R ≈ 0.5115 and H_R ≈ 0.43.

Fields
------
De_R     : Rayleigh-time Deborah number λ₁,₀/t_R, t_R = √(ρ R₀³/γ)
alphas   : NTuple{3} of principal flow-rate ratios
i_ext    : index of extensional axis (argmax alphas)
i_comp   : index of primary compressive axis
XR       : inertiocapillary Newtonian end-plate factor X_R (default 0.5115)
Xe       : elastic-limit X = 1.5
phi      : overlap ratio φ = c/c* as seen by the stress balance (φ/C_φ)
Oh       : Ohnesorge number (the inertial input; 0 ⇒ no polymer arrest)
U_1      : the polymer prefactor U₁ = λ₁,₀ k_BT/(η_s E₀³) (see `CaberFlow`)
"""
struct CaberInertialFlow
    De_R   :: Float64
    alphas :: NTuple{3, Float64}
    i_ext  :: Int
    i_comp :: Int
    XR     :: Float64
    Xe     :: Float64
    phi    :: Float64
    Oh     :: Float64
    U_1    :: Float64
end

function CaberInertialFlow(De_R::Float64, alphas::NTuple{3, Float64},
                           phi::Float64, Oh::Float64, U_1::Float64;
                           XR::Float64=0.5115, Xe::Float64=1.5)::CaberInertialFlow
    i_ext  = argmax(collect(alphas))
    i_comp = argmin(collect(alphas))
    return CaberInertialFlow(De_R, alphas, i_ext, i_comp, XR, Xe, phi, Oh, U_1)
end

# ── Shared X switch ───────────────────────────────────────────────────────────

"""
    _x_switch(X0, Xe, U_R, phi, N1_p, R) -> (psi, X)

End-plate-factor switch shared by both CaBER limits (report Sec. 4.5.1, the
linear switching law). `X0` is the low-stress (Newtonian) end-plate factor, `Xv`
in the viscocapillary regime and `XR` in the inertiocapillary regime, and `Xe`
the elastic limit (3/2).

The switch variable ψ is the polymer normal stress over the capillary pressure,
ψ = clamp(−φ U_R N1_p R, 0, 2), where U_R = U₁/De in the viscous regime and
Oh·U₁/De_R in the inertial regime. X is carried linearly from X0 to Xe as ψ runs
from 0 to 2 and stays at Xe beyond. N1_p < 0 in extension, so ψ ≥ 0.
"""
@inline function _x_switch(X0::Float64, Xe::Float64, U_R::Float64,
                           phi::Float64, N1_p::Float64, R::Float64)
    psi = clamp(-phi * U_R * N1_p * R, 0.0, 2.0)
    X   = X0 + 0.5 * (Xe - X0) * psi
    return psi, X
end

# ── Flow evaluators ───────────────────────────────────────────────────────────

"""
    eval_flow(flow, t, strain, N1_p) -> NamedTuple

Evaluate the flow state at the current time, Hencky strain and polymer N1_p.

Returns a NamedTuple with the fields Wi, De, R, psi, X, N1_p, K_til (R, psi, X
and K_til are NaN for `ConstantWiFlow`; K_til = φ U₁ or φ U_R_eff is the
polymer-stress prefactor of the balance, written to the CaBER diagnostic CSV).
"""
function eval_flow(flow::ConstantWiFlow, t::Float64, strain::Float64, N1_p::Float64)
    return (
        Wi    = flow.Wi_const,
        De    = flow.De,
        R     = NaN,
        psi   = NaN,
        X     = NaN,
        N1_p  = N1_p,
        K_til = NaN,
    )
end

function eval_flow(flow::CaberFlow, t::Float64, strain::Float64, N1_p::Float64)
    R   = exp(-0.5 * strain)

    # Polymer modulus over capillary pressure, for the X switch.
    U_R = flow.U_1 / flow.De                       # = U₁/De_v
    psi, X = _x_switch(flow.Xv, flow.Xe, U_R, flow.phi, N1_p, R)

    K_til = flow.phi * flow.U_1                     # derived temporary (diagnostic)
    Wi = (1.0/3.0) * ( ((2.0*X - 1.0) / R) * flow.De  +  K_til * N1_p )

    return (
        Wi    = Wi,
        De    = flow.De,
        R     = R,
        psi   = psi,
        X     = X,
        N1_p  = N1_p,
        K_til = K_til,
    )
end

function eval_flow(flow::CaberInertialFlow, t::Float64, strain::Float64, N1_p::Float64)
    R = exp(-0.5 * strain)

    # U_R is not an independent input: U_R = Oh·U₁/De_R (see the struct docstring).
    # Formed per evaluation; ∝ 1/De_R across a fixed-Oh sweep (by design).
    U_R_eff = flow.Oh * flow.U_1 / flow.De_R

    psi, X = _x_switch(flow.XR, flow.Xe, U_R_eff, flow.phi, N1_p, R)

    # arg = (2X-1)/R − U_R_eff φ Ñ₁ₚ ;  Ñ₁ₚ = −N1_p ⇒ +φ U_R_eff N1_p
    K_til = flow.phi * U_R_eff                      # derived temporary (diagnostic)
    arg = (2.0*X - 1.0) / R + K_til * N1_p
    Wi  = (2.0 * sqrt(2.0) * flow.De_R / R) * sqrt(max(arg, 0.0))

    return (
        Wi    = Wi,
        De    = flow.De_R,
        R     = R,
        psi   = psi,
        X     = X,
        N1_p  = N1_p,
        K_til = K_til,
    )
end

end # module UsyextFlows
