"""
    UsyextFlows.jl

Flow protocol structs and evaluators for the unsteady extension (USYEXT) solver.

Mirrors MATLAB's
  usyext_flowpack_constant_Wi.m
  usyext_flowpack_caber.m

Two protocols are supported:
  ConstantWiFlow — Wi fixed at a user-specified value; De given externally.
  CaberFlow      — Wi computed from the 0-D capillary/stress balance
                   (Entov–Hinch model for CaBER thinning).

Public API
----------
  ConstantWiFlow   — constant-Wi flow context
  CaberFlow        — CaBER flow context
  FlowState        — output of eval_flow (plain NamedTuple)
  eval_flow        — dispatch on flow type; returns a FlowState NamedTuple
"""
module UsyextFlows

export ConstantWiFlow, CaberFlow, CaberInertialFlow, eval_flow

# ── Constant-Wi flow ──────────────────────────────────────────────────────────

"""
    ConstantWiFlow

Flow context for constant Weissenberg number extension.

Fields
------
Wi_const : constant Wi
De       : Deborah number (= λ₀ / t_cap)
alphas   : NTuple{3} of principal flow-rate ratios
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

Flow context for CaBER capillary-thinning extension.

The 0-D viscocapillary stress balance (TnW paper v4 §`sec:mfsbe`, Eq.
`e:Wi-visc`) gives Wi from the current filament radius R = exp(-ε/2) and the
first normal stress difference N1_p, in the paper's notation (`U_1`, not the
old `K̃`):

    R   = exp(-ε/2)
    U_R = U_1 / De                         # capillary-normalized modulus ratio (= U_1/De_{0v})
    ψ,X = _x_switch(Xv, Xe, U_R, φ, N1_p, R)
    Wi  = ⅓ [ (2X-1)/R · De  +  φ U_1 N1_p ]

Here `U_1 = λ₁ kT /(η_s R_eo³)` is the viscocapillary polymer prefactor (paper
Eq. for `U_1`; → 0.325 as N_K→∞). The old code group `K̃ = φ U_1` is now only a
derived temporary (returned in the diagnostic NamedTuple for the CaBER CSV).

Fields
------
De       : Deborah number  (= λ₀ / t_cap = De_{0v})
alphas   : NTuple{3} of principal flow-rate ratios
i_ext    : index of extensional axis (argmax alphas)
i_comp   : index of primary compressive axis
Xv       : viscous-limit X = 0.7127  (constant; see Entov-Hinch)
Xe       : elastic-limit X = 1.5     (constant)
phi      : concentration φ = c/c*
U_1      : viscocapillary polymer prefactor  λ₁ kT /(η_s R_eo³)
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

Flow context for CaBER capillary thinning in the **inertio-capillary** regime
(Oh ≪ 1).  Companion to `CaberFlow` (the viscocapillary Oh ≫ 1 limit); the two
closed-form limits are implemented as separate structs so the viscous path is
left untouched.

The 0-D inertio-capillary stress balance (TnW paper v4 §`sec:mfsbe`, Eq.
`e:Wi-inert`) gives Wi from the current filament radius R = exp(-ε/2) and the
first normal stress difference N1_p.  The polymer prefactor is formed PER
EVALUATION from the physical inputs (2026-06-12 Oh-primary pivot):

    R       = exp(-ε/2)
    U_R_eff = Oh · U_1 / De_R               # = U_1/De_0v — the device locus
    ψ,X     = _x_switch(XR, Xe, U_R_eff, φ, N1_p, R)
    arg     = (2X-1)/R  +  φ U_R_eff N1_p   # capillary minus polymer normal stress
    Wi      = (2√2 · De_R / R) · √(max(arg, 0))

`U_R` is NOT an independent constant: the relaxation time is carried by the
solvent (`λ₁ = U_1 η_s R_eo³/kT`), so `U_R = R₀c*kT/γ = U_1/De_0v =
Oh·U_1/De_R`. A De_R sweep at fixed Oh therefore has `U_R_eff ∝ 1/De_R` —
the prefactor varies across the sweep BY DESIGN (it traces the device locus
of one real fluid; the fluid group `C = U_1·Oh/De_R^{1/3} = ℓ_λ/ℓ_R` is
R₀-independent).  `Oh = 0` is the degenerate no-arrest limit (the polymer
term vanishes → Keller–Miksis pinch-off); inertio-ELASTO-capillary thinning
is the finite-Oh regime, with arrest requiring roughly `Oh ≳ 1/N_K` (the
balance needs `Ñ₁ₚ* ~ 1/Oh`, capped at `~N_K` by finite extensibility).
The diagnostic group `K̃ = φ U_R_eff` is returned in the NamedTuple for the
CaBER CSV.

Pure-solvent check (N1_p → 0): `Wi = 2√2 De_R √(2X_R-1) R^{-3/2}`, i.e.
`Wi ∝ R^{-1/β}` with β = 2/3, and the pre-elastic prefactor
`H_R = √(8(2X_R-1))` (Wi₀ = H_R De_R at R = 1).  With Gaillard's measured
self-similar prefactor A ≈ 0.47, `2X_R-1 = 2A³/9 ≈ 0.0231` ⟹ X_R ≈ 0.5115,
H_R ≈ 0.43.

Fields
------
De_R     : Rayleigh-time Deborah number (= λ₀ / t_R,  t_R = √(ρ R₀³/γ))
alphas   : NTuple{3} of principal flow-rate ratios
i_ext    : index of extensional axis (argmax alphas)
i_comp   : index of primary compressive axis
XR       : inertio-capillary Newtonian end-plate factor X_R (default 0.5115)
Xe       : elastic-limit X = 1.5
phi      : concentration φ = c/c*
Oh       : Ohnesorge number (primary inertial input; 0 ⇒ no polymer arrest)
U_1      : viscocapillary polymer prefactor λ₁kT/(η_s R_eo³) (≈0.325 as N_K→∞)
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

Regime-neutral end-plate-factor switch shared by both CaBER limits. `X0` is the
low-stress (Newtonian) end-plate factor — `Xv` in the viscocapillary regime,
`XR` in the inertio-capillary regime — and `Xe` the elastic limit (3/2).

The switch variable ψ is the capillary-normalized polymer/capillary stress ratio
`ψ = clamp(−φ U_R N1_p R, 0, 2)`, where `U_R` is the capillary-normalized modulus
ratio (`U_R = U_1/De` in the viscous regime, the supplied `U_R` in the inertial
regime). X is carried linearly from X0 to Xe as ψ runs 0→2. N1_p < 0 in
extension, so ψ ≥ 0.
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

Evaluate the flow state at the current time, strain, and polymer N1_p.

Returns a NamedTuple with at least fields: Wi, De, R, psi, X, N1_p, K_til.
(`K_til = φ U` is a derived diagnostic temporary; the structs store U_1 / U_R.)
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

    # Capillary-normalized modulus ratio for the (regime-neutral) X switch.
    U_R = flow.U_1 / flow.De                       # = U_1/De_{0v}
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

    # Device-locus prefactor: U_R is not independent — U_R = Oh·U_1/De_R.
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
