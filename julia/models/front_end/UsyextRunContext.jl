"""
    UsyextRunContext.jl

Context assembly for the USYEXT (unsteady extension) integrator: reads a TOML
input file and builds everything `UsyextIntegrator.integrate_usyext` needs.
Every key, its meaning and its default are listed in docs/input_reference.md.

Builds:
  - UsyextCtx       : runtime polymer model context (kernel + derived scales)
  - Flow            : ConstantWiFlow | CaberFlow | CaberInertialFlow
  - IntegrationPlan : compiled sampling schedule
  - TolOpts         : adaptive controller tolerances

Reuses the same polymer/drag/spectrum wiring as StyextRunContext
(same kernel, same E₀ normalisation, same η_s formula).

TOML differences from STYEXT:
  - flow.kind = "caber" | "caber_inertial" | "constant"
  - flow.De (or flow.De_values[1]) instead of limits.Wi_max
  - [integration] section for the step-size controller
  - [output] section for the sampling schedule
  - [initial_condition] section (read by `build_initial_M`)

Public API
----------
  load_usyext_context   — parse TOML → (UsyextCtx, flow, IntegrationPlan, TolOpts)
  build_initial_M       — optional pre-stretched initial conformation from the TOML
"""
module UsyextRunContext

import TOML

using ..Fene:           build_fene_model, na_for_model
using ..Drag:           build_drag_assessor, drag_draining_ratio, drag_ratio_zimm,
                        parse_drag_interp, DEFAULT_P_PHI, default_p_phi, DEFAULT_Q_OES,
                     parse_s_form, DEFAULT_G0_1D, DEFAULT_S_FORM, cu_lambda0_ratio
using ..Spectrum:       build_spectrum_assessor
using ..CoeffsRelax:    build_coeffs_kernel
using ..UsyextFlows:    ConstantWiFlow, CaberFlow, CaberInertialFlow
using ..UsyextIntegrator: UsyextCtx, TolOpts, IntegrationPlan

export load_usyext_context, build_initial_M

# ── TOML helpers (shared with StyextRunContext) ───────────────────────────────

_sec(d, s)       = get(d, s, Dict{String,Any}())
_getf(d, k, def) = Float64(get(d, k, def))
_geti(d, k, def) = Int(round(Float64(get(d, k, def))))
_gets(d, k, def) = String(get(d, k, def))

# ── Spectrum resolution ───────────────────────────────────────────────────────

const _SPEC_MAP = Dict(
    "blob"             => :blob,
    "rouse-full"       => :rouse_full,
    "rouse_full"       => :rouse_full,
    "rouse-analytical" => :rouse_full,
    "rouse-scaling"    => :rouse_scaling,
    "rouse_scaling"    => :rouse_scaling,
    "zimm-scaling"     => :zimm_scaling,
    "zimm_scaling"     => :zimm_scaling,
)

function _resolve_spectrum(spec_str::String, drag_str::String)::Symbol
    s = lowercase(strip(spec_str))
    d = lowercase(strip(drag_str))
    if s == "auto"
        d in ("constant","c1d2") && return :rouse_full
        d == "c2d2"              && return :blob
        error("Cannot auto-resolve spectrum for drag_model=\"$drag_str\"")
    end
    sym = get(_SPEC_MAP, s, nothing)
    sym !== nothing || error("Unknown spectrum_model: \"$spec_str\"")
    return sym
end

# ── Rouse weights ─────────────────────────────────────────────────────────────

function _rouse_weights(Nm::Int)::Vector{Float64}
    Sp = zeros(Nm)
    for p in 1:Nm
        if isodd(p)
            th    = p * π / (2*(Nm+1))
            Sp[p] = (2.0/(Nm+1)) * cot(th)^2
        end
    end
    return Sp
end

# ── ηₛ from Zimm formula ──────────────────────────────────────────────────────

function _etas_from_lambdaZ(lambdaZ::Float64, kBT::Float64, E0::Float64,
                              hK_star::Float64, NK::Float64,
                              draining_model::Symbol)::Float64
    U_INF     = 0.325
    alpha_inf = drag_draining_ratio(hK_star, Inf)
    alpha_c   = draining_model === :nondraining ?
                    alpha_inf :
                    drag_draining_ratio(hK_star, NK)
    U = U_INF * alpha_c / alpha_inf
    return lambdaZ * kBT / (U * E0^3)
end

# ── U_1 for CaBER flow ────────────────────────────────────────────────────────

"""
    _compute_U1(Gam_eq, hK_star, NK, draining_model) -> Float64

Polymer prefactor of the capillary-thinning stress balance,
U₁ = λ₁,₀ k_BT/(η_s E₀³) (report Sec. 4.5.1). Equals Γ_eq·U_Z, where
Γ_eq = λ₁,₀/λ_Z and U_Z = U_∞·(α_c/α_∞) with U_∞ = 0.325 (the Zimm universal
ratio of a non-draining theta coil), which is identically λ₁,₀ k_BT/(η_s E₀³)
given the η_s formula used in this context. U₁ → 0.325 in the dilute
non-draining limit. The product φ·U₁ is formed inside `eval_flow`.
"""
function _compute_U1(Gam_eq::Float64,
                     hK_star::Float64, NK::Float64,
                     draining_model::Symbol)::Float64
    U_INF     = 0.325
    alpha_inf = drag_draining_ratio(hK_star, Inf)
    alpha_c   = draining_model === :nondraining ?
                    alpha_inf :
                    drag_draining_ratio(hK_star, NK)
    Uz = U_INF * (alpha_c / alpha_inf)
    return Gam_eq * Uz
end

# ── Sample-strain schedule ────────────────────────────────────────────────────

function _build_sample_strains(output_sec::Dict, strain_max::Float64)::Vector{Float64}
    sample_mode = lowercase(_gets(output_sec, "sample_mode", "strain_grid"))

    if sample_mode == "strain_grid"
        dout = _getf(output_sec, "dstrain_out", 1e-2)
        dout > 0 || error("output.dstrain_out must be > 0")
        s = collect(0.0 : dout : strain_max)
        if isempty(s) || abs(s[end] - strain_max) > 1e-12
            push!(s, strain_max)
        end
        return sort(unique(s))

    elseif sample_mode == "strain_samples"
        raw = get(output_sec, "strain_samples", Float64[])
        isempty(raw) && error("output.strain_samples is empty")
        s = sort(unique(Float64.(raw)))
        s[1] < 0 && error("output.strain_samples must be >= 0")
        s[end] > strain_max + 1e-12 && error("output.strain_samples must be <= strain_max")
        s[1] > 1e-12 && pushfirst!(s, 0.0)
        abs(s[end] - strain_max) > 1e-12 && push!(s, strain_max)
        return sort(unique(s))

    else
        error("output.sample_mode=\"$sample_mode\" not supported in strain-mode integrator")
    end
end

# ── Initial condition (pre-stretch) ───────────────────────────────────────────

"""
    build_initial_M(toml_path, ctx; eps0=nothing) -> Union{Matrix{Float64}, Nothing}

Build the modal conformation initial condition M(0) ∈ ℝ^{Nm×3} from the TOML
`[initial_condition]` section.  Returns `nothing` for the equilibrium IC (so the
integrator falls back to its default `(1/3) I`), or an Nm×3 matrix for a
pre-stretched IC.

Supported `kind`:
  - `"equilibrium"` (default): returns `nothing`.
  - `"uniaxial_affine"`: affine uniaxial stretch of the equilibrium coil at
    Hencky strain ε₀ = `micro_affine_strain`.  Each principal component scales as
    `M_ii = (1/3)·exp(2·αᵢ·ε₀)` with the flow's `alphas` ratios — i.e. for the
    canonical `alphas = (-½,-½,1)`, `M_zz = (1/3)e^{2ε₀}`, `M_xx = M_yy =
    (1/3)e^{-ε₀}`.  Applied to all modes (`apply_to_modes = "all"`).

This affine map `M_ii = (1/3)exp(2αᵢε₀)` is the standard incompressible uniaxial
affine deformation of an isotropic equilibrium coil (the chains are stretched
with the fluid before the flow starts).  `ctx` is the `UsyextCtx` returned by
`load_usyext_context`; `eps0` overrides the TOML strain.

Example: `M0 = build_initial_M("input.toml", ctx)` then
`integrate_usyext(ctx, flow, plan, tol; M0 = M0)`.
"""
function build_initial_M(toml_path::AbstractString, ctx;
                         eps0::Union{Float64,Nothing}=nothing)
    d      = TOML.parsefile(toml_path)
    ic_sec = _sec(d, "initial_condition")
    kind   = lowercase(_gets(ic_sec, "kind", "equilibrium"))

    if kind == "equilibrium"
        return nothing
    elseif kind == "uniaxial_affine"
        e0 = eps0 === nothing ? _getf(ic_sec, "micro_affine_strain", 0.0) : eps0
        abs(e0) < 1e-14 && return nothing   # zero strain ⇒ equilibrium
        Nm     = ctx.Nm
        alphas = ctx.alphas
        M0 = Matrix{Float64}(undef, Nm, 3)
        @inbounds for i in 1:3
            val = (1.0/3.0) * exp(2.0 * alphas[i] * e0)
            for p in 1:Nm
                M0[p, i] = val
            end
        end
        return M0
    else
        error("build_initial_M: initial_condition.kind=\"$kind\" not supported " *
              "(use \"equilibrium\" or \"uniaxial_affine\")")
    end
end

# ── Main loader ───────────────────────────────────────────────────────────────

"""
    load_usyext_context(toml_path; De=nothing, phi=nothing, phi_kw=nothing)
        -> (ctx::UsyextCtx, flow, plan::IntegrationPlan, tol::TolOpts)

Parse a USYEXT TOML file (transient extension or capillary thinning) and
assemble everything `integrate_usyext` needs.  The TOML keys, their meaning and
their defaults are listed in docs/input_reference.md; a commented example is
julia/examples/inputs/caber_example.toml.

Optional keyword overrides (useful for parameter sweeps from one file):
  De     : overrides `[flow] De` (for kind = "caber_inertial" this is De_R)
  phi    : overrides `[concentration] phi` (same spelling as `load_styext_context`)
  phi_kw : older spelling of `phi`, kept for existing scripts; giving both with
           different values is an error

Returns
-------
ctx  : polymer model context (kernel, Rouse weights, scales λ₁,₀, η_s, c k_BT)
flow : `ConstantWiFlow`, `CaberFlow` or `CaberInertialFlow` as selected by `[flow] kind`
plan : sampling schedule and initial step sizes (`[integration]`, `[output]`)
tol  : adaptive step-size controller tolerances (`[integration]`)

Capillary-thinning inputs: for kind = "caber" the Deborah number De = λ₁,₀/t_v
uses the viscocapillary time; for kind = "caber_inertial" De is the
Rayleigh-time Deborah number De_R and `[flow] Oh` (the Ohnesorge number,
default 0.005) is the second input, from which U_R = Oh·U₁/De_R is derived.
`[flow] U_R` is a legacy override that is converted to the equivalent Oh.

Example
-------
    ctx, flow, plan, tol = load_usyext_context("caber_example.toml"; De = 1.0)
    traj = integrate_usyext(ctx, flow, plan, tol)
"""
function load_usyext_context(toml_path :: AbstractString;
                              De        :: Union{Float64, Nothing} = nothing,
                              phi       :: Union{Float64, Nothing} = nothing,
                              phi_kw    :: Union{Float64, Nothing} = nothing)

    # `phi` and `phi_kw` are the same override; `phi_kw` is the older spelling.
    if phi !== nothing && phi_kw !== nothing && phi != phi_kw
        error("load_usyext_context: phi=$phi and phi_kw=$phi_kw disagree; give one of them")
    end
    phi_kw = phi !== nothing ? phi : phi_kw

    d = TOML.parsefile(toml_path)

    poly_sec  = _sec(d, "polymer")
    conc_sec  = _sec(d, "concentration")
    mod_sec   = _sec(d, "model")
    flow_sec  = _sec(d, "flow")
    lim_sec   = _sec(d, "limits")
    intg_sec  = _sec(d, "integration")
    out_sec   = _sec(d, "output")

    # ── Polymer ───────────────────────────────────────────────────────────────
    NK      = _getf(poly_sec, "NK",      5000.0)
    Nm      = _geti(poly_sec, "Nm",      1)
    hK_star = _getf(poly_sec, "hK_star", 0.25)
    Q0_sq   = _getf(poly_sec, "Q0_sq",   1.0)
    kBT     = _getf(poly_sec, "kBT",     1.0)
    lambda0 = _getf(poly_sec, "lambda0", 1.0)

    isfinite(NK) && NK > 0        || error("polymer.NK must be finite and positive")
    Nm >= 1                        || error("polymer.Nm must be ≥ 1")
    0.0 < hK_star < 0.5            || error("polymer.hK_star must be in (0, 0.5)")

    # ── Concentration ─────────────────────────────────────────────────────────
    if phi_kw !== nothing
        phi = phi_kw
    else
        phi_raw = get(conc_sec, "phi", nothing)
        if phi_raw !== nothing
            phi = Float64(phi_raw)
        else
            phi_vals = get(conc_sec, "phi_values", nothing)
            phi = (phi_vals !== nothing && !isempty(phi_vals)) ?
                  Float64(phi_vals[1]) : 1.0
        end
    end
    phi >= 0 || error("phi must be ≥ 0")

    # ── Model selectors ───────────────────────────────────────────────────────
    fe_str       = _gets(mod_sec, "fe_model",       "fenepme")
    drag_str     = _gets(mod_sec, "drag_model",     "c2d2")
    draining_str = _gets(mod_sec, "draining_model", "partialdraining")
    spectrum_str = _gets(mod_sec, "spectrum_model", "auto")

    fe_model       = Symbol(lowercase(fe_str))
    drag_model     = Symbol(lowercase(drag_str))
    draining_model = Symbol(lowercase(draining_str))
    spectrum_model = _resolve_spectrum(spectrum_str, drag_str)

    fe_model in (:fenepme, :fenepm, :fenep) ||
        error("fe_model must be fenepme/fenepm/fenep. Got: $fe_model")
    drag_model in (:constant, :c1d2, :c2d2) ||
        error("drag_model must be constant/c1d2/c2d2. Got: $drag_model")
    draining_model in (:partialdraining, :nondraining) ||
        error("draining_model must be partialdraining/nondraining. Got: $draining_model")

    # drag_interp: "RP"/"RPI" (default, the original interpolation), "OESCUI", "OESI",
    # "BARECUI", "BAREI" (and the archived "RPD"/"RPDW"/"RPDP"/"CU"); see Drag.jl.  p_phi is
    # read only for "CU"/"OESCUI"/"BARECUI"; otherwise at the scheme's default (1.86 CU, 2.0 CUI).
    drag_interp = parse_drag_interp(_gets(mod_sec, "drag_interp", "RP"))
    p_phi       = drag_interp in (:cu, :oescui, :barecui) ? _getf(mod_sec, "p_phi", default_p_phi(drag_interp)) :
                                                            default_p_phi(drag_interp)
    drag_interp in (:cu, :oescui, :barecui) && !(isfinite(p_phi) && p_phi > 0) &&
        error("model.p_phi must be finite and > 0 for drag_interp = \"CU\"/\"OESCUI\"/\"BARECUI\". Got: $p_phi")
    # g0_1D / s_form (archived CU scheme only): relaxation-time factor g(s). C_phi: overlap
    # constant in c = φ/(C_phi E₀³), read for all schemes. The defaults change nothing.
    g0_1D  = drag_interp === :cu ? _getf(mod_sec, "g0_1D", DEFAULT_G0_1D) : DEFAULT_G0_1D
    s_form = (drag_interp === :cu && g0_1D != 1.0) ?
                 parse_s_form(_gets(mod_sec, "s_form", "sum")) : DEFAULT_S_FORM
    drag_interp === :cu && !(isfinite(g0_1D) && g0_1D > 0) &&
        error("model.g0_1D must be finite and > 0 for drag_interp = \"CU\" (1.0 = off). Got: $g0_1D")
    # q_oes: stretch-delayed shielding exponent, ψ = min(φ/E^q, 1); OESI/OESCUI only, default 0
    # (shielding set by the overlap alone). Our capillary-thinning work sets 0.25 explicitly.
    is_oes = drag_interp in (:oesi, :oescui)
    q_oes  = is_oes ? _getf(mod_sec, "q_oes", DEFAULT_Q_OES) : DEFAULT_Q_OES
    !is_oes && haskey(mod_sec, "q_oes") &&
        @warn "model.q_oes is read only for drag_interp = \"OESI\"/\"OESCUI\"; ignored for $(drag_interp)"
    is_oes && !(isfinite(q_oes) && 0.0 <= q_oes <= 1.0) &&
        error("model.q_oes must be finite and in [0, 1] for drag_interp = \"OESI\"/\"OESCUI\". Got: $q_oes")
    C_phi = _getf(mod_sec, "C_phi", 1.0)
    (isfinite(C_phi) && C_phi > 0) || error("model.C_phi must be finite and > 0. Got: $C_phi")
    # De_mode: "input" (default) keeps λ₁,₀/λ_Z = ζ₀/ζ_Z from the drag model; "cu_fit" (legacy,
    # accepted with any drag_interp) replaces it by the Martin-type correlation
    # λ₁,₀/λ_Z = (1 + φ^p)^(1/p), p = p_phi, wherever λ₁,₀/λ_Z enters (De, U₁, η_s).
    De_mode_str = lowercase(_gets(mod_sec, "De_mode", "input"))
    De_mode = De_mode_str == "input" ? :input : De_mode_str == "cu_fit" ? :cu_fit :
              error("model.De_mode must be \"input\" or \"cu_fit\". Got: \"$De_mode_str\"")
    if De_mode === :cu_fit && !(drag_interp in (:cu, :oescui, :barecui))
        p_phi = _getf(mod_sec, "p_phi", DEFAULT_P_PHI)
        (isfinite(p_phi) && p_phi > 0) || error("model.p_phi must be finite and > 0 for De_mode = \"cu_fit\". Got: $p_phi")
    end

    # ── Flow ──────────────────────────────────────────────────────────────────
    flow_kind = lowercase(_gets(flow_sec, "kind", "caber"))
    flow_kind in ("caber","caber_inertial","constant","piece_wise_constant") ||
        error("flow.kind must be caber/caber_inertial/constant. Got: $flow_kind")

    av = get(flow_sec, "alphas", [-0.5, -0.5, 1.0])
    length(av) == 3 || error("flow.alphas must have 3 entries")
    alphas = (Float64(av[1]), Float64(av[2]), Float64(av[3]))

    # De: explicit flow.De, or flow.De_values[1] as fallback
    if De !== nothing
        De_val = De
    else
        De_raw = get(flow_sec, "De", nothing)
        if De_raw !== nothing
            De_val = Float64(De_raw)
        else
            De_vals = get(flow_sec, "De_values", nothing)
            De_val  = (De_vals !== nothing && !isempty(De_vals)) ?
                      Float64(De_vals[1]) : 1.0
        end
    end
    De_val > 0 && isfinite(De_val) || error("flow.De must be > 0 and finite")

    Wi_const = _getf(flow_sec, "Wi", 1.0)

    # ── Limits ────────────────────────────────────────────────────────────────
    max_frac_stretch_sq = _getf(lim_sec, "max_frac_stretch_sq", 0.999)

    # ── Derived scales (same formulas as StyextRunContext) ────────────────────
    Q0      = sqrt(Q0_sq)
    NKs     = NK / Nm
    bK      = Q0 / sqrt(NKs)
    E0      = bK * sqrt(NK)      # = Q0 * sqrt(Nm)
    E0_sq   = E0^2
    Einf    = bK * NK
    Einf_sq = Einf^2
    Ymax    = max_frac_stretch_sq * Einf_sq

    # FENE pole (report Sec. 2.3): Q∞² = Q₀²·N_K for FENE-PME (whole-chain argument),
    # Q₀²·N_K/N_m per spring for FENE-PM/P.  (The liquid-bridge solver's Params.jl uses
    # N_K for every model, so the two differ for N_m > 1.)
    Qinf_sq = fe_model === :fenepme ? Q0_sq * NK : Q0_sq * NK / Nm

    c_star = E0^(-3)
    ckBT   = phi * c_star * kBT / C_phi      # c = φ/(C_phi E₀³)
    # Overlap ratio seen by the CaBER stress balance (K̃ = φ_σ U₁ and the ψ switch); the
    # drag and spectrum keep the true φ. C_phi = 1 → φ_σ = φ (the historical value).
    phi_stress = phi / C_phi

    # ── Drag LUT + spectrum ───────────────────────────────────────────────────
    drag_lut = build_drag_assessor(phi, hK_star, NK, drag_model, draining_model;
                                   drag_interp=drag_interp, p_phi=p_phi,
                                   g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)
    spectrum  = build_spectrum_assessor(
        spectrum_model = spectrum_model,
        Nm             = Nm,
        NK             = NK,
        hK_star        = hK_star,
        phi            = phi,
        drag_model     = drag_model,
        draining_model = draining_model,
        drag_interp    = drag_interp,
        p_phi          = p_phi,
        g0_1D          = g0_1D,
        s_form         = s_form,
        q_oes          = q_oes,
    )

    # ── Equilibrium drag + η_s ───────────────────────────────────────────────
    # E = 1 is the equilibrium coil.  Γ_eq = ζ₀/ζ_Z = λ₁,₀/λ_Z: the equilibrium longest
    # relaxation time at this φ relative to its dilute (Zimm) value (report Sec. 2.2).
    Gam_eq, _ = drag_ratio_zimm(1.0, phi, hK_star, NK, drag_model, draining_model;
                                drag_interp=drag_interp, p_phi=p_phi,
                                g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)   # incl. g(s₀) for CU
    Gam_eq > 0 || error("Equilibrium drag ratio must be > 0")
    # Legacy cu_fit: λ₁,₀/λ_Z = (1+φ^p)^(1/p) instead of the drag-model value; this sets
    # η_s (through λ_Z = λ₁,₀/Γ_eq) and U₁ = Γ_eq·U_Z.
    De_mode === :cu_fit && (Gam_eq = cu_lambda0_ratio(phi, p_phi))
    # [flow] De_Z (optional): with cu_fit, De = De_Z · λ₁,₀/λ_Z overrides flow.De.
    De_Z = _getf(flow_sec, "De_Z", NaN)
    if De_mode === :cu_fit && isfinite(De_Z)
        De_Z > 0 || error("flow.De_Z must be > 0. Got: $De_Z")
        De_val = De_Z * Gam_eq
    end
    lambdaZ   = lambda0 / Gam_eq
    etas      = _etas_from_lambdaZ(lambdaZ, kBT, E0, hK_star, NK, draining_model)

    # ── Rouse weights + FENE model ─────────────────────────────────────────────
    Sp      = _rouse_weights(Nm)
    fene    = build_fene_model(fe_model, Nm, Q0_sq, Qinf_sq)
    kernel  = build_coeffs_kernel(fene, drag_lut, spectrum)

    # ── Axis indices ──────────────────────────────────────────────────────────
    alphas_vec = collect(alphas)
    i_ext      = argmax(alphas_vec)
    i_comp     = argmin(alphas_vec)

    # ── UsyextCtx ─────────────────────────────────────────────────────────────
    ctx = UsyextCtx(kernel, Sp, fe_model, alphas, i_ext, i_comp,
                    Nm, E0, Ymax, lambda0, etas, ckBT)

    # ── Flow context ──────────────────────────────────────────────────────────
    local flow_ctx
    if flow_kind == "constant"
        flow_ctx = ConstantWiFlow(De_val, Wi_const, alphas)
    elseif flow_kind == "caber"
        U_1      = _compute_U1(Gam_eq, hK_star, NK, draining_model)
        flow_ctx = CaberFlow(De_val, alphas, phi_stress, U_1)
    elseif flow_kind == "caber_inertial"
        # De_val is interpreted as De_R (the Rayleigh-time Deborah number).
        # X_R = inertiocapillary Newtonian end-plate factor (default 0.5115, from the
        # self-similar prefactor A ≈ 0.47 of Gaillard et al.: 2X_R−1 = 2A³/9).
        X_R = _getf(flow_sec, "X_R", 0.5115)
        X_e = _getf(flow_sec, "Xe",  1.5)
        U_1 = _compute_U1(Gam_eq, hK_star, NK, draining_model)

        # The Ohnesorge number Oh is the inertial input: U_R is not an independent
        # constant but U_R = Oh·U₁/De_R (the relaxation time is carried by the
        # solvent), formed per evaluation inside eval_flow.  Oh = 0 is the degenerate
        # limit without polymer arrest and is allowed.  flow.U_R remains accepted as
        # a legacy override; it is converted to the equivalent Oh = U_R·De_R/U₁ for
        # THIS De_R only (a constant-U_R sweep does not correspond to one fluid; give Oh).
        U_R_in = _getf(flow_sec, "U_R", NaN)
        if isfinite(U_R_in) && U_R_in >= 0
            Oh_val = U_R_in * De_val / U_1
            @info "caber_inertial: flow.U_R=$(U_R_in) override → equivalent Oh=$(Oh_val) at De_R=$(De_val)"
        else
            Oh_val = _getf(flow_sec, "Oh", NaN)
            if !(isfinite(Oh_val) && Oh_val >= 0)
                Oh_val = 0.005
                @info "caber_inertial: flow.Oh not specified — defaulting to Oh=0.005 (Gaillard aqueous)"
            end
        end
        flow_ctx = CaberInertialFlow(De_val, alphas, phi_stress, Oh_val, U_1; XR=X_R, Xe=X_e)
    else
        error("load_usyext_context: flow.kind=\"$flow_kind\" not implemented in Julia port yet")
    end

    # ── Integration plan ──────────────────────────────────────────────────────
    ind_str  = lowercase(_gets(intg_sec, "independent", "strain"))
    ind_sym  = Dict("strain"=>:strain, "time"=>:time, "auto"=>:auto)[ind_str]

    strain_max = _getf(intg_sec, "strain_max", 6.0)
    strain_max > 0 || error("integration.strain_max must be > 0")

    sample_strains = _build_sample_strains(out_sec, strain_max)

    plan = IntegrationPlan(
        ind_sym,
        sample_strains,
        _getf(intg_sec, "dstrain_init", 1e-3),
        _getf(intg_sec, "dt_init",      1e-3),
        _getf(intg_sec, "Wi_min_strain_independent", 0.0),
    )

    # ── TolOpts ───────────────────────────────────────────────────────────────
    tol = TolOpts(
        rtol         = _getf(intg_sec, "rtol",          1e-7),
        atol         = _getf(intg_sec, "atol",          1e-9),
        safety       = _getf(intg_sec, "safety",        0.9),
        grow_max     = _getf(intg_sec, "grow_max",      2.0),
        shrink_min   = _getf(intg_sec, "shrink_min",    0.2),
        dstrain_min  = _getf(intg_sec, "dstrain_min",   1e-8),
        dstrain_max  = _getf(intg_sec, "dstrain_max",   5e-2),
        dt_min       = _getf(intg_sec, "dt_min",        1e-10),
        dt_max       = _getf(intg_sec, "dt_max",        1.0),
        scale_floor  = _getf(intg_sec, "scale_floor",   1e-14),
        epsdot_floor = _getf(intg_sec, "epsdot_floor",  1e-16),
        hit_eps      = _getf(intg_sec, "hit_eps",       1e-12),
        overshoot_eps = _getf(intg_sec, "overshoot_eps", 1e-10),
        max_attempts = _geti(intg_sec, "max_attempts",  50_000_000),
        max_wall_s   = _getf(intg_sec, "max_wall_s",    Inf),
    )

    return ctx, flow_ctx, plan, tol
end

end # module UsyextRunContext
