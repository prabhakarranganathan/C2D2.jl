"""
    Params.jl + RuntimeCtx.jl

C2D2Params  : portable, serialisable parameter struct (loaded from TOML).
RuntimeCtx  : runtime-only assembled context (built from C2D2Params).

C2D2Params mirrors the TOML file sections.
RuntimeCtx holds the CoeffsKernel, LUTs, and derived scalar parameters.
It is never saved to disk — rebuild from C2D2Params if needed.
"""
module Params

using ..Fene:      build_fene_model, na_for_model, _rouse_weight_matrix,
                   FenePME, FenePM, FeneP
using ..Drag:      build_drag_assessor, DragAssessor, drag_draining_ratio
using ..Spectrum:  build_spectrum_assessor, SpectrumAssessor
using ..CoeffsRelax: build_coeffs_kernel, CoeffsKernel, closure_from_M_diag

using LinearAlgebra: dot
import TOML

export C2D2Params, RuntimeCtx, build_runtime_ctx
export load_params_toml, from_bhat

# ── C2D2Params ────────────────────────────────────────────────────────────────

"""
    C2D2Params

All user-facing parameters for a C2D2 liquid bridge simulation.
Loaded from a TOML file; all fields are plain numbers, strings, or bools.
No function handles. Safe to serialise.

TOML section mapping
--------------------
[chain]         : Nm, NK, hK_star
[concentration] : phi
[model]         : fe_model, drag_model, draining_model, spectrum_model
[flow]          : Oh, De
[grid]          : N_cells, z_max, half_bridge
[time]          : t_max, dt_init, dt_min, dt_max, rtol, atol
[output]        : save_interval, output_dir
"""
struct C2D2Params
    # Chain
    Nm              :: Int
    NK              :: Float64
    hK_star         :: Float64

    # Concentration
    phi             :: Float64      # c/c*  — concentration ratio, used in constitutive kernel

    # Initial bridge geometry (Spiegelberg IC only)
    # ic_volfrac sets the shape parameter for the Spiegelberg initial profile.
    # It is INDEPENDENT of phi (c/c*): you may want a well-stretched chain concentration
    # but a mildly necked initial bridge, or vice versa.
    # For ic_type = "bhat" this field is ignored.
    # Default: equals phi (backward-compatible with code that used phi for both).
    ic_volfrac      :: Float64      # Spiegelberg bridge shape parameter (NOT c/c*)

    # Constitutive model selectors
    fe_model        :: Symbol       # :fenepme | :fenepm | :fenep
    drag_model      :: Symbol       # :constant | :c1d2 | :c2d2
    draining_model  :: Symbol       # :nondraining | :partialdraining
    spectrum_model  :: Symbol       # :rouse_scaling | :zimm_scaling | :blob | ...

    # Flow
    Oh              :: Float64      # Ohnesorge number
    De              :: Float64      # Deborah number
    gravity         :: Float64      # dimensionless gravity G = ρgR/γ (Bond-like); 0 = off

    # Grid
    N_cells         :: Int          # number of FVM cells
    z_max           :: Float64      # half-bridge length
    half_bridge     :: Bool         # always true for now

    # Time integration
    t_max           :: Float64
    dt_init         :: Float64
    dt_min          :: Float64
    dt_max          :: Float64
    rtol            :: Float64
    atol            :: Float64

    # Output
    save_interval   :: Float64
    output_dir      :: String

    # Initial condition selector
    # "spiegelberg" (default) — Spiegelberg/JC bridge profile using ic_volfrac
    # "bhat"                  — Bhat et al. 2010 cosine profile: R = 1 - 0.8 sin(πz/(2z_max))
    ic_type         :: String
end

"""
    load_params_toml(path) -> C2D2Params

Load parameters from a TOML file.
"""
function load_params_toml(path::String)::C2D2Params
    d = TOML.parsefile(path)

    get_sym(section, key) = Symbol(d[section][key])
    get_f(section, key)   = Float64(d[section][key])
    get_i(section, key)   = Int(d[section][key])
    get_b(section, key)   = Bool(d[section][key])
    get_s(section, key)   = String(d[section][key])

    phi      = get_f("concentration", "phi")
    out_sec  = get(d, "output", Dict())
    flow_sec = get(d, "flow", Dict())
    ic_type = get(out_sec, "ic_type", "spiegelberg")
    # ic_volfrac: explicit field, or fall back to phi (backward-compatible).
    # For Bhat IC cases phi is the concentration and the bridge shape is fixed
    # analytically; ic_volfrac is unused there but still stored.
    ic_volfrac = Float64(get(out_sec, "ic_volfrac", phi))
    # gravity: dimensionless G = ρgR/γ; defaults to 0 (gravity-free / horizontal bridge)
    gravity = Float64(get(flow_sec, "gravity", 0.0))

    return C2D2Params(
        get_i("chain",         "Nm"),
        get_f("chain",         "NK"),
        get_f("chain",         "hK_star"),
        phi,
        ic_volfrac,
        get_sym("model",       "fe_model"),
        get_sym("model",       "drag_model"),
        get_sym("model",       "draining_model"),
        get_sym("model",       "spectrum_model"),
        get_f("flow",          "Oh"),
        get_f("flow",          "De"),
        gravity,
        get_i("grid",          "N_cells"),
        get_f("grid",          "z_max"),
        get_b("grid",          "half_bridge"),
        get_f("time",          "t_max"),
        get_f("time",          "dt_init"),
        get_f("time",          "dt_min"),
        get_f("time",          "dt_max"),
        get_f("time",          "rtol"),
        get_f("time",          "atol"),
        get_f("output",        "save_interval"),
        get_s("output",        "output_dir"),
        ic_type,
    )
end

# ── RuntimeCtx ────────────────────────────────────────────────────────────────

"""
    RuntimeCtx

Runtime-only context built from C2D2Params.
Contains all precomputed objects needed during time integration.
Never saved to disk.

Fields
------
params        : the source C2D2Params
kernel        : CoeffsKernel (FENE + drag LUT + spectrum assessor assembled)
Sp            : Nm-vector of end-to-end mode weights S_p
Q0_sq         : dimensionless equilibrium spring length squared (= 1 in our scaling)
Qinf_sq       : dimensionless FENE pole (= NK for PME/PM, NK/Nm for P variant)
Na            : number of X auxiliary variables
zeta0_ratio   : equilibrium drag ratio ζ₀/ζ_Z
U_zimm        : Zimm pre-factor U = U_∞·α/α_∞ (= 0.325 for non-draining theta coil)
poly_prefactor: = −φ·U_zimm·(ζ₀/ζ_Z)/De_v  (= −χ/3 from FVM note Eq. 42)
"""
struct RuntimeCtx
    params          :: C2D2Params
    kernel          :: CoeffsKernel
    Sp              :: Vector{Float64}
    Q0_sq           :: Float64
    Qinf_sq         :: Float64
    Na              :: Int
    zeta0_ratio     :: Float64
    U_zimm          :: Float64   # Zimm pre-factor U = U_∞ · α/α_∞  (0.325 for non-draining)
    poly_prefactor  :: Float64   # = −φ · U_zimm · (ζ₀/ζ_Z) / De_v  (χ/3 with correct sign)
end

"""
    build_runtime_ctx(p::C2D2Params) -> RuntimeCtx

Assemble the full runtime context from parameters.
This is the only place where LUTs are built and assessors are wired together.
"""
function build_runtime_ctx(p::C2D2Params)::RuntimeCtx
    # ── Input sanity checks ───────────────────────────────────────────────────
    p.Oh      > 0.0 || error("Oh must be positive, got $(p.Oh)")
    p.De      > 0.0 || error("De must be positive, got $(p.De)")
    p.phi     >= 0.0 || error("phi (c/c*) must be non-negative, got $(p.phi)")
    p.NK      > 1.0 || error("NK must be > 1 (FENE extensibility), got $(p.NK)")
    p.Nm      >= 1  || error("Nm must be >= 1, got $(p.Nm)")
    p.N_cells >= 2  || error("N_cells must be >= 2, got $(p.N_cells)")
    p.z_max   > 0.0 || error("z_max must be positive, got $(p.z_max)")
    p.gravity >= 0.0 || @warn "gravity < 0 (unusual for a vertical bridge); proceeding"
    if (p.fe_model === :hookean || p.fe_model === :oldroyd_b) && p.phi > 1.0
        @warn "Hookean spring law with phi=$(p.phi) > 1 (c/c* > 1); ensure this is intentional."
    end
    if p.De < 1e-3
        @warn "De=$(p.De) < 1e-3: the M-relaxation eigenvalue λ ≈ −1/De = $(round(-1/p.De; sigdigits=1)) " *
              "is on the explicit side of KenCarp4, requiring dt ≲ $(round(3.5*p.De; sigdigits=1)) " *
              "for stability.  Production runs should use De ≥ 1e-3.  " *
              "For a purely Newtonian baseline (phi=0) use De=1e-3 (or any value ≥ 1e-3); " *
              "De has no physical effect when poly_prefactor=0."
    end

    Nm = p.Nm

    # ---- Dimensionless FENE parameters ----
    # In our dimensionless scaling (Q0²= 1 in modal coordinates via the Nm convention):
    # The report uses Q0² = 1, E0² = Nm, Q∞² = NK/Nm for each mode
    # For PME: chi = Y/Nm,  Qinf_sq per mode = NK (since Y_max ≈ Nm*(NK/Nm) = NK)
    # We store the spring-level Qinf: Qinf_sq = NK (contour-length squared, scaled by E0²)
    Q0_sq   = 1.0             # in units of Q0² (equilibrium spring length)
    Qinf_sq = Float64(p.NK)   # pole in units of Q0²

    # ---- Rouse end-to-end weights S_p ----
    Sp = _rouse_weights(Nm)

    # ---- FENE model ----
    fene = build_fene_model(p.fe_model, Nm, Q0_sq, Qinf_sq)

    # ---- Drag LUT ----
    drag_lut = build_drag_assessor(p.phi, p.hK_star, Float64(p.NK),
                                   p.drag_model, p.draining_model)

    # ---- Spectrum assessor ----
    spectrum = build_spectrum_assessor(
        spectrum_model = p.spectrum_model,
        Nm             = Nm,
        NK             = Float64(p.NK),
        hK_star        = p.hK_star,
        phi            = p.phi,
        drag_model     = p.drag_model,
        draining_model = p.draining_model,
    )

    # ---- Assembled coefficients kernel ----
    kernel = build_coeffs_kernel(fene, drag_lut, spectrum)

    Na    = na_for_model(p.fe_model, Nm)
    zeta0 = spectrum.zeta0_ratio

    # ---- Zimm pre-factor U_zimm = U_∞ · (α / α_∞) ----------------------------
    # U_∞ ≈ 0.325 is the non-draining theta-chain constant (Kirkwood-Riseman).
    # α is the draining factor for the full chain; α_∞ is its non-draining limit.
    # For :nondraining the ratio is 1.0 by construction.
    U_INF      = 0.325   # non-draining theta-chain constant
    alpha_inf  = drag_draining_ratio(p.hK_star, Inf)   # ≈ 5.77
    alpha_coil = p.draining_model === :nondraining ?
                     alpha_inf :
                     drag_draining_ratio(p.hK_star, Float64(p.NK))
    U_zimm = U_INF * alpha_coil / alpha_inf

    # ---- Polymer stress prefactor in the filament momentum equation -----------
    # From FVM note (Ardekani form), the pressure-like variable is:
    #   Π = AK(A) + χ · A · f · (M_zz − M_rr)
    # where χ = 3 φ U_zimm (ζ₀/ζ_Z) / De_v  (FVM note Eq. 42).
    #
    # In the code, tau_diff = −3 Σ_p f_p (M_zz_p − M_rr_p), so the polymer
    # contribution to ∂Π/∂z is −(χ/3) · ∂(A · tau_diff)/∂z.
    # Hence poly_prefactor = −χ/3 = −φ · U_zimm · (ζ₀/ζ_Z) / De_v.
    #
    # Sign convention: poly_prefactor is NEGATIVE for a polymer that partially
    # resists thinning (which is correct for the viscoelastic cases in Bhat 2010).
    poly_prefactor = -(p.phi * U_zimm * zeta0) / p.De

    return RuntimeCtx(p, kernel, Sp, Q0_sq, Qinf_sq, Na, zeta0, U_zimm, poly_prefactor)
end

# ── Rouse weights ─────────────────────────────────────────────────────────────

"""
    _rouse_weights(Nm) -> Vector{Float64}

End-to-end weights S_p = Σ_i Π[i,p]² for the Rouse orthogonal matrix.
Only odd modes are nonzero for linear chains with free ends.
  S_p = (2/(Nm+1)) * cot²(pπ / 2(Nm+1))   for odd p
"""
function _rouse_weights(Nm::Int)::Vector{Float64}
    # S_p = (2/(Nm+1)) * cot²(pπ/(2(Nm+1)))  for odd p   (report eq. 2.4)
    # S_p = 0                                  for even p
    # Sum over odd p = Nm  (exact identity for free-ended Rouse chain)
    Sp = zeros(Nm)
    for p in 1:Nm
        if isodd(p)
            theta_p = p * π / (2*(Nm+1))
            Sp[p] = (2.0 / (Nm+1)) * cot(theta_p)^2
        end
        # even modes: Sp[p] = 0 (already initialised)
    end
    return Sp
end

# ── Bhat parameter conversion ─────────────────────────────────────────────────

"""
    from_bhat(Oh_Bhat, beta, De_Bhat; Lambda=1.5) -> NamedTuple

Convert Bhat et al. (2010) parameter convention to C2D2 code parameters.

**Bhat convention**

| Symbol   | Definition                          |
|----------|-------------------------------------|
| Oh_Bhat  | η₀ / √(ρ R₀ γ)  (total-solution)   |
| beta     | η_s / η₀  (solvent fraction)        |
| De_Bhat  | λ / t_c  (total-viscosity Deborah)  |
| Lambda   | L / R₀  (full-bridge aspect ratio)  |

**C2D2 code convention (Oh_Bhat < 1 regime — inertia-dominated)**

When `Oh_Bhat < 1`, Bhat's time is already the Rayleigh time `t_R = √(ρR₀³/γ)`,
which is *also* the C2D2 code time.  No conversion factor is needed for `De`.

For Oldroyd-B, `chi` is the fundamental polymer coupling — it is set by `beta`,
`Oh`, and `De` alone, with no need for a physical polymer concentration:

```
chi = 3 · Oh_code · (1−β) / (β · De_code)
    = 3 · Oh_Bhat · (1−β) / De_Bhat          (both forms are equivalent)
```

The C2D2 code uses `phi` (c/c*) as its coupling parameter, so `phi` is back-
computed from `chi` rather than from a chain model:

```
Oh    = beta · Oh_Bhat                    (solvent Ohnesorge)
De    = De_Bhat                           (De_Bhat is already Rayleigh-time De)
chi   = 3 · Oh_Bhat · (1−β) / De_Bhat    (fundamental Oldroyd-B coupling)
phi   = chi · De / (3 · U_zimm)           (proxy for chi; not a physical c/c*)
z_max = Lambda / 2                        (half-bridge length)
```

Note that `U_zimm` cancels when the stress is computed:
`phi · U_zimm / De = chi / 3 = Oh_Bhat · (1−β) / De_Bhat`

**Oh_Bhat > 1 regime (viscosity-dominated):** Bhat's time is the viscous time
`t_v = η₀ R₀ / γ`, related by `t_v = Oh_Bhat · t_R`.  In that case:
```
De    = De_Bhat / Oh_Bhat        (viscous → Rayleigh-time conversion)
chi   = 3 · Oh_Bhat · (1−β) / De_Bhat  (same Bhat formula; De_Bhat viscous-time)
```
This function implements only the Oh_Bhat < 1 regime.

This function assumes `draining_model = :nondraining` and `drag_model = :constant`.
For partial-draining cases build `RuntimeCtx` directly and read `ctx.poly_prefactor`.

**Returns** `(Oh, De, phi, z_max, chi)` as a `NamedTuple`.
"""
function from_bhat(Oh_Bhat::Float64, beta::Float64, De_Bhat::Float64;
                   Lambda::Float64 = 1.5)
    0.0 < beta  <= 1.0 || error("beta must be in (0, 1], got $beta")
    Oh_Bhat > 0.0      || error("Oh_Bhat must be positive, got $Oh_Bhat")
    De_Bhat > 0.0      || error("De_Bhat must be positive, got $De_Bhat")
    Lambda  > 0.0      || error("Lambda must be positive, got $Lambda")

    U_zimm  = 0.325   # non-draining theta-chain Kirkwood–Riseman constant
    zeta0   = 1.0     # nondraining: ζ₀/ζ_Z = 1

    Oh    = beta * Oh_Bhat
    # Oh_Bhat < 1: De_Bhat is already Rayleigh-time De — no conversion factor.
    # (Oh_Bhat > 1 viscous-time regime not implemented here.)
    De    = De_Bhat
    z_max = Lambda / 2.0

    # chi is the fundamental Oldroyd-B coupling.
    # Equivalent forms: 3·Oh_Bhat·(1−β)/De_Bhat = 3·Oh_code·(1−β)/(β·De_code)
    chi = 3.0 * Oh_Bhat * (1.0 - beta) / De_Bhat

    # phi is back-computed from chi — it is NOT a physical polymer concentration here.
    # U_zimm cancels in the stress: phi·U_zimm/De = chi/3 = Oh_Bhat·(1−β)/De_Bhat
    phi = chi * De / (3.0 * U_zimm * zeta0)

    return (Oh=Oh, De=De, phi=phi, z_max=z_max, chi=chi)
end

end # module Params
