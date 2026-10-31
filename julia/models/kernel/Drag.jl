"""
    Drag.jl

Chain friction kernel for the C2D2 model.

Implements:
  - Kirkwood-Riseman draining function α(h*_k, N)
  - Full C2D2 drag ratio ζ/ζ_Z (constant / C1D2 / C2D2 models)
  - A piecewise-PCHIP LUT assessor for fast evaluation during time integration

Public API
----------
  drag_draining_ratio(hK_star, NK_per_blob)     -> alpha
  drag_ratio_zimm(E, phi, hK_star, NK,
                  drag_model, draining_model;
                  drag_interp=:rp, p_phi=default_p_phi(drag_interp))
                                                -> (zeta_ratio, diagnostics)
  build_drag_assessor(params)                   -> DragAssessor
  parse_drag_interp(str)                        -> :rp | :rpd | :rpdw | :rpdp | :cu
                                                   | :oescui | :oesi | :barecui | :barei
                                                   ("RPI" is an alias of "RP")
  oesi_dilute_weak_E(phi, E_max)                -> v2-family dilute/weak boundary E_b(φ) (NaN if none)
  oesi_shape_D(A)                               -> monotone shape factor D(A) of the v2 family
  DEFAULT_P_PHI                                 -> 1.86 (CU);  DEFAULT_P_PHI_OESCUI -> 2.0 (CUI schemes)
  default_p_phi(drag_interp)                    -> the scheme's default p_phi
  DEFAULT_Q_OES                                 -> 0.0 (stretch-delayed OES dressing exponent; OESI/OESCUI only)
  oesi_dressing_E(phi, q_oes, E_max)            -> E_q = φ^(1/q): where ψ = min(φ/E^q, 1) leaves 1 (NaN if none)

Drag-interpolation schemes (`drag_interp`, only used by drag_model = :c2d2):
  :rp  — the report scheme (RP): tension-blob G1/G2 drag for E > φ, correlation-blob
         drag ζ_c N_c for E ≤ φ, with a kink at the switch.  Default; bit-identical
         to the pre-2026-10-15 code.
  :cu  — Churchill–Usagi blend ζ = (ζ_w^p + ζ_s^p)^(1/p), p = p_phi, of the weakly
         screened (Batchelor + Mackaplow–Shaqfeh) drag ζ_w and the strongly screened
         drag ζ_s, with a quadratic-at-small-stretch Batchelor J and a screening length
         that crosses over smoothly to the correlation length.  Smooth in φ and E;
         Λ = 1 at rest for every φ.  See docs/logs/aru-to-ko/2026-10-15_cu-drag-interp-scheme.md.
  :rpd, :rpdw, :rpdp — RP with an overlap-dressed core-blob shielding factor
         α → α + (α_∞ − α)ψ, ψ = φE | φ/E | φ (2026-10-20).
  The v2 family (Aru's OES note Secs. 3.1–3.7; 2026-10-21/22), one code path
  `_drag_c2d2_v2` with two switches, overlap-enhanced shielding (oes) and the capped
  Churchill–Usagi blend (cui):
  :oescui — oes on, cui on.  RP's adopted form.  Tension-blob drag with the overlap dressing
         ψ = min(φ/E^q, 1), q = q_oes (default 0, i.e. RPDP's ψ = min(φ,1); the Thin-n-Win
         runs set q_oes = 0.25, 2026-10-22 pm), the regularised Mackaplow–Shaqfeh screening length
         χ/ξ_t = sqrt((1 + ln(1/φ_t))/φ_t), A = min(E², χ/ξ_t) and the monotone shape
         factor D(A) = sqrt(A) (A ≤ 4), 2 + ln(A/4) (A > 4); blended with the non-draining
         correlation-blob drag (α_∞/α_Z)·min(φ, √N_K) by (ζ_w^p + ζ_s^p)^(1/p), p = p_phi
         (default 2.0), capped at the Rouse drag (α_∞/α_Z)√N_K.
  :oesi — oes on, cui off: weak drag for E > φ, strong for E ≤ φ (continuous; slope
         corners at φ = 1, at the dilute/weak boundary φE³ = 1 + ln(E/φ) and at E = φ).
  :barecui — oes off, cui on: bare α(h*_K, N_K/N_t) and α(h*_K, N_K/N_c), cap
         (α(h*_K, 1)/α_Z)√N_K.
  :barei — oes off, cui off: bare C2D2 with the 21-Oct boundary and D(A); the original blob
         model at rest.
  eval_drag(assessor, E)                        -> zeta_ratio
  eval_drag_deriv(assessor, E)                  -> d(zeta_ratio)/dE
"""
module Drag

# PCHIP interpolation is implemented manually below (no external dependency).

export drag_draining_ratio, drag_ratio_zimm, DragAssessor,
       build_drag_assessor, eval_drag, eval_drag_deriv,
       parse_drag_interp, DEFAULT_P_PHI, DEFAULT_P_PHI_OESCUI, default_p_phi,
       DEFAULT_Q_OES, oesi_dressing_E,
       g1d_factor, parse_s_form, DEFAULT_G0_1D, DEFAULT_S_FORM, cu_lambda0_ratio,
       oesi_dilute_weak_E, oesi_shape_D

# ── Drag-interpolation scheme selector ───────────────────────────────────────

"""Default Churchill–Usagi exponent p_φ (fitted at rest to Martin's correlation)."""
const DEFAULT_P_PHI = 1.86

"""
    parse_drag_interp(str) -> Symbol

Map the TOML `[model] drag_interp` string (case-insensitive) to `:rp` (also spelled
`"RPI"`), `:rpd`, `:rpdw`, `:rpdp`, `:cu`, `:oescui`, `:oesi`, `:barecui` or `:barei`.
Errors on anything else.
"""
function parse_drag_interp(str::AbstractString)::Symbol
    s = lowercase(strip(String(str)))
    s == "rp" && return :rp
    s == "rpi" && return :rp                       # alias (2026-10-22): the original scheme
    s == "cu" && return :cu
    s == "rpd" && return :rpd
    s == "rpdw" && return :rpdw
    s == "rpdp" && return :rpdp
    s == "oesi" && return :oesi
    s == "oescui" && return :oescui
    s == "barei" && return :barei
    s == "barecui" && return :barecui
    error("drag_interp must be \"OESCUI\", \"OESI\", \"BARECUI\", \"BAREI\" or \"RPI\" (= \"RP\"; also the archived " *
          "\"RPD\", \"RPDW\", \"RPDP\", \"CU\"), case-insensitive. Got: \"$(str)\"")
end

# ── U_1D relaxation factor g(s) (CU scheme only; 2026-10-17, Aru→Ko handover) ──
#
# The universal ratio U_1D = λ₁D/E₀² changes from its Zimm value to its Rouse
# value (U_1D,R/U_1D,Z ≈ 0.44, Kröger et al. 2000) as hydrodynamic interactions
# are screened, by overlap (correlation blobs) or by stretching (tension blobs).
# RP's formulation (overlap-crossover note §4):
#
#     N_t = clamp(E², 1, N_K)                                   (code units, E₀ = 1)
#     N_c = overlap (correlation-blob) count — two forms (2026-10-17 pm, RP's settled ansatz):
#           :sum      → N_c′ = min((1 + φ^p)^(2/p), N_K)   CU-smooth continuation (DEFAULT);
#                       the hard corner at φ = 1 gave a bump / second minimum in Wi(φ)
#           :sum_hard → N_c  = clamp(φ², 1, N_K)            the 4a9e141 form, kept for reproducibility
#     s   = N_c + N_t − 2  (both forms),  s₀ = N_c − 1 (at rest)
#     g(s) = g0 + (1 − g0) / (1 + s^p)^(1/p),   p = p_phi,   g(0) = 1, g → g0 for s ≫ 1
#     (g0 is an empirical O(1) knob in this ansatz — RP's working value 0.5 — not a claim
#      about U_1D,R/U_1D,Z.)  The DRAG keeps its own blob counts; only s/s₀ use N_c′.
#
# The CU branch returns ζ_eff = g(s)·ζ as `zeta_ratio`, so Λ = λ(E)/λ₀ =
# [g(s)/g(s₀)]·ζ(E)/ζ₀ and λ₀/λ_Z = g(s₀)·ζ₀/ζ_Z follow automatically
# (s₀ = s(N_c, 1) is the at-rest value). The blob diagnostics use the bare ζ.
# g0_1D = 1 switches the factor off exactly (ζ_eff = 1.0·ζ, bit-identical).

"""Default U_1D floor g0 (1.0 = factor off; physically ≈ 0.44)."""
const DEFAULT_G0_1D = 1.0

"""Default form of the screening variable s(N_c, N_t): smooth N_c′ (see `_Nc_of`)."""
const DEFAULT_S_FORM = :sum

"""
    parse_s_form(str) -> Symbol

Map the TOML `[model] s_form` string (case-insensitive) to a supported form:
`"sum"` (default; smooth N_c′ = min((1+φ^p)^(2/p), N_K), s = N_c′ + N_t − 2) or
`"sum_hard"` (N_c = clamp(φ², 1, N_K); the 2026-10-17 am / commit 4a9e141 form).
"""
function parse_s_form(str::AbstractString)::Symbol
    s = lowercase(strip(String(str)))
    s == "sum"      && return :sum
    s == "sum_hard" && return :sum_hard
    error("s_form must be \"sum\" or \"sum_hard\" (case-insensitive). Got: \"$(str)\"")
end

"""
    _Nc_of(::Val{form}, phi, p, NK) -> N_c

Overlap (correlation-blob) count entering s. `:sum` → CU-smooth N_c′ = min((1+φ^p)^(2/p), N_K)
(→ 1 at φ = 0, → φ² for φ ≫ 1); `:sum_hard` → clamp(φ², 1, N_K).
"""
@inline _Nc_of(::Val{:sum},      phi::Float64, p::Float64, NK::Float64) = min((1.0 + phi^p)^(2.0 / p), NK)
@inline _Nc_of(::Val{:sum_hard}, phi::Float64, p::Float64, NK::Float64) = clamp(phi^2, 1.0, NK)
_Nc_of(::Val{F}, phi, p, NK) where {F} = error("_Nc_of: unknown s_form :$(F)")

"""s(N_c, N_t) dispatch on the form symbol. Add new forms as new `Val` methods."""
@inline _s_of(::Val{:sum},      Nc::Float64, Nt::Float64) = Nc + Nt - 2.0
@inline _s_of(::Val{:sum_hard}, Nc::Float64, Nt::Float64) = Nc + Nt - 2.0
_s_of(::Val{F}, Nc, Nt) where {F} = error("_s_of: unknown s_form :$(F)")

"""
    cu_lambda0_ratio(phi, p) -> λ₀/λ_Z

Model-anchored equilibrium relaxation-time ratio of RP's ansatz (2026-10-17):
λ₀/λ_Z = (1 + φ^p)^(1/p), p = p_phi fitted per system to SAOS λ₀/λ_Z. Used by
`De_mode = "cu_fit"`: De₀ = De_Z · (1 + φ^p)^(1/p). Independent of h*_K and g0.
"""
@inline cu_lambda0_ratio(phi::Float64, p::Float64)::Float64 = (1.0 + phi^p)^(1.0 / p)

"""
    g1d_factor(s, g0, p) -> g

U_1D relaxation factor g(s) = g0 + (1 − g0)/(1 + s^p)^(1/p). Returns exactly 1.0
when g0 == 1 (factor off).
"""
@inline function g1d_factor(s::Float64, g0::Float64, p::Float64)::Float64
    g0 == 1.0 && return 1.0
    return g0 + (1.0 - g0) / (1.0 + s^p)^(1.0 / p)
end

const _V2_SCHEMES  = (:oescui, :oesi, :barecui, :barei)
const _CUI_SCHEMES = (:oescui, :barecui)
const _ALL_DRAG_INTERP = (:rp, :rpd, :rpdw, :rpdp, :cu, _V2_SCHEMES...)

const _OES_SCHEMES = (:oescui, :oesi)

"""Validate (drag_interp, p_phi, g0_1D, s_form, q_oes); p_phi matters for :cu, :oescui and
:barecui, g0_1D/s_form only for :cu, q_oes only for :oescui and :oesi (0 ≤ q ≤ 1; q > 1 would
break continuity at φ = E)."""
function _check_drag_interp(drag_interp::Symbol, p_phi::Float64,
                            g0_1D::Float64 = DEFAULT_G0_1D, s_form::Symbol = DEFAULT_S_FORM,
                            q_oes::Float64 = DEFAULT_Q_OES)
    drag_interp in _ALL_DRAG_INTERP ||
        error("drag_interp must be one of $(_ALL_DRAG_INTERP). Got: :$(drag_interp)")
    if drag_interp in _OES_SCHEMES
        (isfinite(q_oes) && 0.0 <= q_oes <= 1.0) ||
            error("q_oes must be finite and in [0, 1] for drag_interp = :$(drag_interp). Got: $(q_oes)")
    end
    if drag_interp in _CUI_SCHEMES
        (isfinite(p_phi) && p_phi > 0) ||
            error("p_phi must be finite and > 0 for drag_interp = :$(drag_interp). Got: $(p_phi)")
    end
    if drag_interp === :cu
        (isfinite(p_phi) && p_phi > 0) ||
            error("p_phi must be finite and > 0 for drag_interp = :cu. Got: $(p_phi)")
        (isfinite(g0_1D) && g0_1D > 0) ||
            error("g0_1D must be finite and > 0 for drag_interp = :cu (1.0 = off). Got: $(g0_1D)")
        s_form in (:sum, :sum_hard) ||
            error("s_form must be :sum or :sum_hard. Got: :$(s_form)")
    end
    return nothing
end

# ── Draining ratio α(h*_k, N) ─────────────────────────────────────────────────

"""
    drag_draining_ratio(hK_star, NK_per_blob) -> alpha

Kirkwood-Riseman draining factor for an isotropic random-walk blob.

  hK_star     : Kuhn HI parameter  a_K / (√π b_K)  > 0
  NK_per_blob : number of Kuhn segments in the blob  (≥ 1; Inf for non-draining limit)

Returns α (dimensionless), same size as NK_per_blob.

Asymptotes:
  NK_per_blob → ∞ : α → 3π/1.6366 ≈ 5.775  (universal non-draining limit)
  NK_per_blob = 1  : α = 6π^(3/2) h*_k
"""
function drag_draining_ratio(hK_star::Float64, NK_per_blob::Real)::Float64
    hK_star > 0 || error("hK_star must be > 0, got $(hK_star)")
    NK_per_blob >= 1 || error("NK_per_blob must be >= 1, got $(NK_per_blob)")

    aK = sqrt(π) * hK_star

    if isinf(NK_per_blob)
        # Non-draining limit: 1/x → 0, polynomial fits → constants
        # P1 → 0.91,  P2 → 1.6
        # α = 3π / P2(∞)  from the formula below with x→∞
        return 3π / 1.6366
    end

    x   = sqrt(NK_per_blob)
    pr  = 0.9119 - 0.0191 / x + 0.1009 / x^2
    psr = 1.6366 - 1.1434 / x - 0.4521 / x^2

    denom = pr / x + 2 * aK * psr
    denom = max(denom, eps(Float64))
    return 3π * 2 * aK / denom
end

# Vectorised overload
function drag_draining_ratio(hK_star::Float64, NK_arr::AbstractVector)::Vector{Float64}
    return [drag_draining_ratio(hK_star, n) for n in NK_arr]
end

# ── Internal helpers ──────────────────────────────────────────────────────────

"""α for a blob of NK/Np Kuhn segments."""
function _alpha_blob(hK_star, NK, Np, draining_model::Symbol)
    if draining_model === :nondraining
        return drag_draining_ratio(hK_star, Inf)
    elseif draining_model === :partialdraining
        return drag_draining_ratio(hK_star, NK / max(Np, 1.0))
    else
        error("Unknown draining_model :$(draining_model). Use :nondraining or :partialdraining.")
    end
end

"""α for the whole Zimm coil (single blob = whole chain)."""
function _alpha_zimm(hK_star, NK, draining_model::Symbol)
    if draining_model === :nondraining
        return drag_draining_ratio(hK_star, Inf)
    elseif draining_model === :partialdraining
        return drag_draining_ratio(hK_star, NK)
    else
        error("Unknown draining_model :$(draining_model)")
    end
end

# ── Full drag ratio: scalar E ─────────────────────────────────────────────────

"""
    drag_ratio_zimm(E, phi, hK_star, NK, drag_model, draining_model;
                    drag_interp=:rp, p_phi=default_p_phi(drag_interp),
                    g0_1D=DEFAULT_G0_1D, s_form=DEFAULT_S_FORM, q_oes=DEFAULT_Q_OES)
                   -> (zeta_ratio, diag)

Compute ζ/ζ_Z for a single stretch value E = |R|/R₀ ≥ 1.

drag_model      : :constant | :c1d2 | :c2d2
draining_model  : :nondraining | :partialdraining
drag_interp     : :rp (default, report scheme) | :rpd | :rpdw | :rpdp (overlap-dressed RP)
                  | :cu (Churchill–Usagi blend) | :oescui | :oesi | :barecui | :barei
                  (the v2 family: overlap-enhanced shielding and/or capped CU blend);
                  only consulted for drag_model = :c2d2
p_phi           : CU exponent; only consulted for drag_interp = :cu, :oescui and :barecui.
                  Defaults to the scheme's own value (1.86 for CU, 2.0 for the CUI schemes)
g0_1D, s_form   : U_1D relaxation factor g(s) (CU only; g0_1D = 1 → off). The CU branch
                  returns ζ_eff = g(s)·ζ; diagnostics carry `g1d`, `s1d` (1.0, NaN otherwise)
q_oes           : stretch-delayed OES dressing exponent, ψ = min(φ/E^q, 1) (OESI/OESCUI only;
                  default 0 = ψ = min(φ, 1), bit-identical to 2026-10-22 am)

diag is a NamedTuple with fields: G, blob_drag_ratio, n_blob, xi_blob, blob_type.
For :cu, :oescui and :barecui, `G` is the effective value G_eff = blob_drag_ratio·n_blob/zeta_ratio
so that the :blob spectrum stays consistent with the chain drag.
"""
function drag_ratio_zimm(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                         drag_model::Symbol, draining_model::Symbol;
                         drag_interp::Symbol = :rp, p_phi::Float64 = default_p_phi(drag_interp),
                         g0_1D::Float64 = DEFAULT_G0_1D, s_form::Symbol = DEFAULT_S_FORM,
                         q_oes::Float64 = DEFAULT_Q_OES)

    E_min = 1.0
    E_max = sqrt(NK)
    E = clamp(E, E_min, E_max)
    phi = max(phi, 0.0)

    alphaZ = _alpha_zimm(hK_star, NK, draining_model)

    if drag_model === :constant
        return 1.0, (G=1.0, blob_drag_ratio=NaN, n_blob=1.0, xi_blob=NaN, blob_type=:NA,
                     g1d=1.0, s1d=NaN)

    elseif drag_model === :c1d2
        xiT    = 1.0 / E
        nT     = E^2
        alphaT = _alpha_blob(hK_star, NK, nT, draining_model)
        zb_rel = (alphaT * xiT) / alphaZ

        A = E / xiT          # = E²
        Y = A - 1.0
        J = 1.0 + (0.5*Y + (2/ℯ)*Y^2) / max(A, eps(Float64))
        G = 1.0 + log(max(J, eps(Float64)))

        zeta_ratio = max(zb_rel * nT / max(G, eps(Float64)), eps(Float64))
        return zeta_ratio, (G=G, blob_drag_ratio=zb_rel, n_blob=nT, xi_blob=xiT, blob_type=:T,
                            g1d=1.0, s1d=NaN)

    elseif drag_model === :c2d2
        if drag_interp === :rp
            return _drag_c2d2(E, phi, hK_star, NK, alphaZ, draining_model)
        elseif drag_interp === :rpd
            return _drag_c2d2(E, phi, hK_star, NK, alphaZ, draining_model; dressed=true)
        elseif drag_interp === :rpdw
            return _drag_c2d2(E, phi, hK_star, NK, alphaZ, draining_model; dressed=true, psi_form=:phi_over_E)
        elseif drag_interp === :rpdp
            return _drag_c2d2(E, phi, hK_star, NK, alphaZ, draining_model; dressed=true, psi_form=:phi)
        elseif drag_interp === :cu
            return _drag_c2d2_cu(E, phi, hK_star, NK, alphaZ, draining_model, p_phi;
                                 g0_1D=g0_1D, s_form=s_form)
        elseif drag_interp in _V2_SCHEMES
            oes, cui = _v2_switches(drag_interp)
            return _drag_c2d2_v2(E, phi, hK_star, NK, alphaZ, draining_model;
                                 oes=oes, cui=cui, p_phi=p_phi, q_oes=q_oes)
        else
            error("Unknown drag_interp :$(drag_interp). Use one of $(_ALL_DRAG_INTERP).")
        end

    else
        error("Unknown drag_model :$(drag_model). Use :constant, :c1d2, or :c2d2.")
    end
end

"""
    _alpha_dressed(alpha_bare, psi) -> α

Overlap-dressed shielding factor (RP, 2026-10-20; drag_interp = :rpd). ψ = φE is the
overlap volume fraction of the (partially stretched) chain, 2016 JOR Eq. 29. As ψ → 1 the
interior of the chain is no longer distinguishable from its surroundings and the core blob
becomes non-draining: α = α_bare + (α_∞ − α_bare)·ψ for ψ < 1, α = α_∞ for ψ ≥ 1.
All correlation blobs (ψ ≥ 1 by construction) are therefore non-draining.
"""
@inline function _alpha_dressed(alpha_bare::Float64, psi::Float64)::Float64
    a_inf = 3π / 1.6366
    psi >= 1.0 && return a_inf
    return alpha_bare + (a_inf - alpha_bare) * max(psi, 0.0)
end

function _drag_c2d2(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                    alphaZ::Float64, draining_model::Symbol; dressed::Bool=false,
                    psi_form::Symbol=:phiE)
    E_max = sqrt(NK)
    # dressing variable (used only if dressed):
    #   :phiE       ψ = φE, overlap volume fraction of the stretched chain (2016 Eq. 29)      [RPD]
    #   :phi_over_E ψ = φ/E, reaches 1 on the weak/strong-overlap boundary E = φ (2016 Eq. 40) [RPDW]
    #   :phi        ψ = φ, coil overlap only; stretch neither advances nor delays it         [RPDP]
    #   all give ψ = φ for coils (E = 1).
    psi   = psi_form === :phi_over_E ? phi / E : psi_form === :phi ? phi : phi * E

    # ---- correlation-blob baseline at equilibrium ----
    if phi > E_max
        xiC    = 1.0 / E_max
        alphaC = dressed ? 3π / 1.6366 : _alpha_blob(hK_star, NK, NK, draining_model)
        zbC    = (alphaC * xiC) / alphaZ
        nC     = NK
    elseif phi > 1.0
        nC     = phi^2
        xiC    = 1.0 / phi
        alphaC = dressed ? 3π / 1.6366 : _alpha_blob(hK_star, NK, nC, draining_model)
        zbC    = (alphaC * xiC) / alphaZ
    else
        nC  = 1.0
        xiC = 1.0
        # dressed: at ψ = φ = 1 (the only φ ≤ 1 point reaching the correlation core) the
        # single c-blob is non-draining, matching the ψ → 1 limit of the dressed coil.
        zbC = (dressed && phi >= 1.0) ? (3π / 1.6366) / alphaZ : 1.0
    end

    xiT = 1.0 / E
    nT  = clamp(E^2, 1.0, NK)
    A   = E / xiT    # = E²

    if E > phi
        # ---- tension-core region ----
        alphaT = _alpha_blob(hK_star, NK, nT, draining_model)
        dressed && (alphaT = _alpha_dressed(alphaT, psi))
        zbT    = (alphaT * xiT) / alphaZ

        phiT = max(phi * E * xiT^2, eps(Float64))
        B    = 1.0 + sqrt(max(log(1.0 / phiT), 0.0) / phiT)

        Y    = A - 1.0
        J    = 1.0 + (0.5*Y + (2/ℯ)*Y^2) / max(A, eps(Float64))

        G1  = 1.0 + log(max((A + J*(B-1)) / max(A + B, eps(Float64)), eps(Float64)))
        G2  = 1.0 + log(max(B / max(B + 1.0/max(A, eps(Float64)), eps(Float64)), eps(Float64)))
        G   = max(G1 / max(G2, eps(Float64)), eps(Float64))

        zeta_ratio = max(zbT * nT / G, eps(Float64))
        return zeta_ratio, (G=G, blob_drag_ratio=zbT, n_blob=nT, xi_blob=xiT, blob_type=:T,
                            g1d=1.0, s1d=NaN)

    else
        # ---- correlation-core region ----
        zeta_ratio = max(zbC * nC / 1.0, eps(Float64))
        return zeta_ratio, (G=1.0, blob_drag_ratio=zbC, n_blob=nC, xi_blob=xiC, blob_type=:C,
                            g1d=1.0, s1d=NaN)
    end
end


"""
    _drag_c2d2_cu(E, phi, hK_star, NK, alphaZ, draining_model, p_phi)

Churchill–Usagi (CU) drag-interpolation scheme for drag_model = :c2d2
(Aru → Ko handover 2026-10-15; overlap-crossover note Secs. 3.4 and 4).

    ζ/ζ_Z = (ζ_w^p + ζ_s^p)^(1/p),   p = p_phi

  ζ_w  — weakly screened (tension-blob) drag, used for ALL φ (no switch at E = φ):
         ζ_w = (α_t ξ_t/α_Z) · N_t · G2/G1,  with
           φ_t      = φ/E
           χ/ξ_t    = sqrt( log(1 + 1/φ_t) / φ_t )          (smooth crossover to ξ_c)
           B        = 1 + χ/ξ_t
           J        = 1 + (2/e)(A−1)²/A                      (quadratic at small stretch)
           G1       = 1 + log((A + J(B−1))/(A + B)),  G2 = 1 + log(B/(B + 1/A))
  ζ_s  — strongly screened (correlation-blob) drag:
         ζ_s = φ · α(h*_K, N_K/max(φ,E)²)/α_Z   (blob count capped at N_K)

Differences from the RP branch: `log(1 + 1/φ_t)` instead of `max(log(1/φ_t), 0)`,
and J without the linear ½(A−1)/A term.  At E = 1: G1 = G2, ζ_w = 1, and the blend
reduces to the equilibrium crossover, so Λ = ζ/ζ₀ = 1 at rest for every φ.

Diagnostics keep the RP regime-based definitions (n_blob, xi_blob, blob_drag_ratio,
blob_type = :T if E > φ else :C); G is returned as G_eff = blob_drag_ratio·n_blob/ζ
with the BARE ζ (so the blob spectrum is unaffected by g).

U_1D relaxation factor (2026-10-17): with `g0_1D ≠ 1` the returned `zeta_ratio` is
ζ_eff = g(s)·ζ, s = N_c + N_t − 2, N_t = clamp(E²,1,N_K), N_c from `_Nc_of(s_form)`
(smooth N_c′ for `:sum`, clamp(φ²,1,N_K) for `:sum_hard`); `g1d` and `s1d` are added
to the diagnostics. See `g1d_factor`.
"""
function _drag_c2d2_cu(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                       alphaZ::Float64, draining_model::Symbol, p_phi::Float64;
                       g0_1D::Float64 = DEFAULT_G0_1D, s_form::Symbol = DEFAULT_S_FORM)
    E_max = sqrt(NK)
    tiny  = eps(Float64)

    xiT = 1.0 / E
    nT  = clamp(E^2, 1.0, NK)
    A   = E / xiT    # = E²

    # ---- weakly screened (tension-blob) drag, all φ ----
    alphaT = _alpha_blob(hK_star, NK, nT, draining_model)
    zbT    = (alphaT * xiT) / alphaZ

    phiT = max(phi * E * xiT^2, tiny)                    # = φ/E
    B    = 1.0 + sqrt(log1p(1.0 / phiT) / phiT)          # χ/ξ_t → ξ_c/ξ_t for φ_t ≫ 1

    Y    = A - 1.0
    J    = 1.0 + (2/ℯ) * Y^2 / max(A, tiny)

    G1   = 1.0 + log(max((A + J*(B-1)) / max(A + B, tiny), tiny))
    G2   = 1.0 + log(max(B / max(B + 1.0/max(A, tiny), tiny), tiny))

    zeta_w = zbT * nT * G2 / max(G1, tiny)

    # ---- strongly screened (correlation-blob) drag ----
    # φ·α_t below the boundary (max(φ,E) = E), ζ_c N_c above it; blob count ≤ N_K.
    nS     = clamp(max(phi, E)^2, 1.0, NK)
    alphaS = _alpha_blob(hK_star, NK, nS, draining_model)
    zeta_s = phi * alphaS / alphaZ

    # ---- Churchill–Usagi blend (bare drag) ----
    zeta_ratio = max((zeta_w^p_phi + zeta_s^p_phi)^(1.0 / p_phi), tiny)

    # ---- U_1D relaxation factor g(s): ζ_eff = g(s)·ζ (exactly ζ when g0_1D = 1) ----
    nC   = _Nc_of(Val(s_form), phi, p_phi, NK)            # overlap count: smooth N_c′ (:sum) or clamp(φ²,1,N_K) (:sum_hard)
    s1d  = _s_of(Val(s_form), nC, nT)                     # nT = clamp(E², 1, N_K) above
    g1d  = g1d_factor(s1d, g0_1D, p_phi)
    zeta_eff = g1d == 1.0 ? zeta_ratio : max(g1d * zeta_ratio, tiny)

    # ---- diagnostics: RP regime-based blob definitions, effective G ----
    if E > phi
        n_blob = nT; xi_blob = xiT; zb = zbT; btype = :T
    else
        if phi > E_max
            xi_blob = 1.0 / E_max
            n_blob  = NK
            zb      = (_alpha_blob(hK_star, NK, NK, draining_model) * xi_blob) / alphaZ
        elseif phi > 1.0
            n_blob  = phi^2
            xi_blob = 1.0 / phi
            zb      = (_alpha_blob(hK_star, NK, n_blob, draining_model) * xi_blob) / alphaZ
        else
            n_blob = 1.0; xi_blob = 1.0; zb = 1.0
        end
        btype = :C
    end
    G_eff = max(zb * n_blob / zeta_ratio, tiny)          # bare ζ: spectrum shape independent of g

    return zeta_eff, (G=G_eff, blob_drag_ratio=zb, n_blob=n_blob, xi_blob=xi_blob,
                      blob_type=btype, g1d=g1d, s1d=s1d)
end

# ── OESI family (v2): overlap-enhanced shielding interpolation, 2026-10-21/22 ─
#
# Four schemes on one code path, `_drag_c2d2_v2(…; oes, cui, p_phi)`:
#
#     drag_interp   oes   cui
#     :oescui       on    on     RP's adopted form (OES + capped Churchill–Usagi blend)
#     :oesi         on    off
#     :barecui      off   on
#     :barei        off   off    bare C2D2 with the 21-Oct dilute/weak boundary and D(A)
#
# Aru → Ko handovers 2026-10-21 (`oesi-drag-interp`) and 2026-10-22 (`oescui-D-pphi-
# switches`); OES note `c2d2_OES.pdf` Secs. 3.1–3.7; reference `oesi.py:zeta_v2`.
# Revision of 2026-10-22: the Batchelor factor 1 + ln J(A) is replaced by the monotone
# D(A) below, p_phi defaults to 2.0 for the CUI schemes, and OES / CUI are switchable.

"""Non-draining limit of the Kirkwood–Riseman shielding factor α (Drag.jl fit)."""
const _ALPHA_INF = 3π / 1.6366

"""Default Churchill–Usagi exponent for the CUI schemes (:oescui, :barecui; 2026-10-22 fit)."""
const DEFAULT_P_PHI_OESCUI = 2.0

"""Default stretch-delayed OES dressing exponent q_oes (0 = ψ = min(φ, 1); Thin-n-Win runs use 0.25)."""
const DEFAULT_Q_OES = 0.0

"""
    oesi_dressing_E(phi, q_oes, E_max) -> E_q

Stretch at which the stretch-delayed OES dressing ψ = min(φ/E^q, 1) leaves 1 at fixed φ > 1:
E_q = φ^(1/q) (> φ for 0 < q < 1).  A slope corner of the OES schemes in E (2026-10-22 pm).
NaN if q = 0 (ψ = min(φ, 1), no corner in E), φ ≤ 1 (ψ < 1 for every E > 1) or E_q ≥ E_max.
"""
function oesi_dressing_E(phi::Float64, q_oes::Float64, E_max::Float64)::Float64
    (q_oes > 0.0 && phi > 1.0) || return NaN
    E_q = phi^(1.0 / q_oes)
    return E_q < E_max ? E_q : NaN
end

"""
    default_p_phi(drag_interp) -> p

Scheme-dependent default of the Churchill–Usagi exponent `p_phi`: `DEFAULT_P_PHI_OESCUI`
(2.0) for `:oescui` / `:barecui`, `DEFAULT_P_PHI` (1.86, Martin's correlation at rest) for
`:cu` and everything else (including the `cu_fit` De₀ with a non-CUI drag).
"""
default_p_phi(drag_interp::Symbol)::Float64 =
    drag_interp in (:oescui, :barecui) ? DEFAULT_P_PHI_OESCUI : DEFAULT_P_PHI

"""
    oesi_shape_D(A) -> D

Monotone shape factor D(A) of the v2 family, replacing 1 + ln J(A) (OES note Sec. 3.7):

    D(A) = sqrt(A)       for A ≤ 4
         = 2 + ln(A/4)   for A > 4

D(1) = 1 and D is C¹ at A = 4 (value 2, slope 1/4).  In the dilute regime (A = E²) the
shape factor E/D(E²) is 1 up to E = 2 and x/(1 + ln x), x = E/2, beyond: non-decreasing,
and Batchelor's E/ln(kE²) with k = e²/4 at large E.  E/(1 + ln J(E²)) instead rose to
1.10 at E ≈ 1.2 and dipped to 1.01 at E ≈ 2.1 (non-monotone dilute drag).
"""
@inline oesi_shape_D(A::Float64)::Float64 = A <= 4.0 ? sqrt(A) : 2.0 + log(A / 4.0)

"""
    _v2_weak(E, phi, hK_star, NK, alphaZ, draining_model, oes, q_oes) -> (zeta_w, zbT, nT, xiT, D)

Tension-blob ("weak") drag of the v2 family (OES note Eqs. 3.1–3.9 with D in place of
1 + ln J), valid for φ < E and continued with A = 1 for φ ≥ E:

    N_t     = clamp(E², 1, N_K),   ξ_t = 1/E,   a = α(h*_K, N_K/N_t)
    α_T     = a + (α_∞ − a)·min(φ/E^q, 1), q = q_oes   (oes)   |   a   (bare)
    φ_t     = φ/E,   χ/ξ_t = sqrt((1 + ln(1/φ_t))/φ_t)   (regularised Mackaplow–Shaqfeh; ∞ at φ = 0)
    A       = min(E², χ/ξ_t)   for φ < E   |   1   for φ ≥ E
    ζ_w/ζ_Z = (α_T ξ_t/α_Z)·N_t/D(A)

With oes on and φ ≥ E (so φ ≥ 1 and, for q ≤ 1, φ/E^q ≥ E^(1−q) ≥ 1), ψ = 1 and
ζ_w = (α_∞/α_Z)·E, as in the 21-Oct OESCUI; the dressing therefore completes at φ = E^q
instead of φ = 1 (q = 0 reproduces ψ = min(φ, 1) exactly, since E^0 = 1).
"""
@inline function _v2_weak(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                          alphaZ::Float64, draining_model::Symbol, oes::Bool, q_oes::Float64)
    nT  = clamp(E^2, 1.0, NK)
    xiT = 1.0 / E
    a   = _alpha_blob(hK_star, NK, nT, draining_model)
    oes && (a = a + (_ALPHA_INF - a) * min(phi / E^q_oes, 1.0))
    if E > phi
        phiT   = phi / E
        chi_xi = phiT > 0.0 ? sqrt((1.0 + log(1.0 / phiT)) / phiT) : Inf
        A      = min(E^2, chi_xi)
    else
        A = 1.0
    end
    D   = oesi_shape_D(A)
    zbT = (a * xiT) / alphaZ
    return zbT * nT / D, zbT, nT, xiT, D
end

"""
    _v2_strong(phi, hK_star, NK, alphaZ, draining_model, oes) -> (zeta_s, zbC, nC, xiC)

Correlation-blob ("strong") drag of the v2 family, continued to all φ (OES note Eq. 3.10):

    N_c     = min(max(φ, 1)², N_K),   ξ_c = 1/min(max(φ, 1), √N_K)
    α_S     = α_∞   (oes)   |   α(h*_K, N_K/N_c)   (bare)
    ζ_s/ζ_Z = (α_S/α_Z)·min(φ, √N_K)

For φ ≥ 1 this is ζ_bC·N_c with the blob diagnostics returned; for φ < 1 it is the formal
continuation (α_S/α_Z)·φ used only inside the CUI blend.  With oes off the bare strong drag
at rest is the original C2D2 blob model, φ·α(h*_K, N_K/φ²)/α_Z.
"""
@inline function _v2_strong(phi::Float64, hK_star::Float64, NK::Float64,
                            alphaZ::Float64, draining_model::Symbol, oes::Bool)
    E_max = sqrt(NK)
    nC    = min(max(phi, 1.0)^2, NK)
    xiC   = 1.0 / min(max(phi, 1.0), E_max)
    aS    = oes ? _ALPHA_INF : _alpha_blob(hK_star, NK, nC, draining_model)
    zbC   = (aS * xiC) / alphaZ
    return (aS / alphaZ) * min(phi, E_max), zbC, nC, xiC
end

"""Rouse cap of the CUI blend: (α_∞/α_Z)·√N_K with oes, (α(h*_K, 1)/α_Z)·√N_K bare."""
@inline function _v2_cap(hK_star::Float64, NK::Float64, alphaZ::Float64,
                         draining_model::Symbol, oes::Bool)
    a1 = oes ? _ALPHA_INF : _alpha_blob(hK_star, NK, NK, draining_model)
    return (a1 / alphaZ) * sqrt(NK)
end

"""
    _drag_c2d2_v2(E, phi, hK_star, NK, alphaZ, draining_model; oes, cui, p_phi, q_oes)

The v2 drag-interpolation family for drag_model = :c2d2 (reference `oesi.py:zeta_v2`):

    ζ_w = `_v2_weak`  (tension blobs; A = 1 continuation for φ ≥ E)
    ζ_s = `_v2_strong` (correlation blobs; continued to φ < 1)
    cui off (:oesi, :barei):     ζ/ζ_Z = ζ_w for E > φ, ζ_s for E ≤ φ
    cui on  (:oescui, :barecui): ζ/ζ_Z = min{ (ζ_w^p + ζ_s^p)^(1/p), cap },  p = p_phi,
                                 cap = (α_∞ | α(h*_K,1))/α_Z · √N_K   (`_v2_cap`)

Continuous in φ and E for all four schemes (A is continuous across the dilute/weak boundary
φE³ = 1 + ln(E/φ); at E = φ, A = 1, D = 1 and N_t = N_c, so with oes off ζ_w = ζ_s there as
well).  The Rouse cap removes the double counting of the blend once both arms reach single
Kuhn segments.  At rest ζ₀ is the same function at E = 1, so Λ = 1 at rest automatically;
with oes off and cui off, ζ₀/ζ_Z = 1 for φ ≤ 1 and φ·α(h*_K, N_K/φ²)/α_Z above (the original
blob model).  With `draining_model = :nondraining` α ≡ α_∞ and oes is a no-op.

Diagnostics (regime by E > φ): `blob_type` :T / :C, `n_blob`, `xi_blob`, `blob_drag_ratio`;
`G` = D(A) (:T) or 1.0 (:C) for the switched schemes, and the effective value
G_eff = blob_drag_ratio·n_blob/ζ for the CUI schemes (so the :blob spectrum stays
consistent with the chain drag, as in the CU branch); `g1d = 1.0`, `s1d = NaN`.
"""
function _drag_c2d2_v2(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                       alphaZ::Float64, draining_model::Symbol;
                       oes::Bool, cui::Bool, p_phi::Float64, q_oes::Float64 = DEFAULT_Q_OES)
    tiny = eps(Float64)
    zeta_w, zbT, nT, xiT, D = _v2_weak(E, phi, hK_star, NK, alphaZ, draining_model, oes, q_oes)
    zeta_s, zbC, nC, xiC    = _v2_strong(phi, hK_star, NK, alphaZ, draining_model, oes)
    weak = E > phi

    if cui
        cap        = _v2_cap(hK_star, NK, alphaZ, draining_model, oes)
        zeta_ratio = min((zeta_w^p_phi + zeta_s^p_phi)^(1.0 / p_phi), cap)
    else
        zeta_ratio = weak ? zeta_w : zeta_s
    end

    if weak
        zb = zbT; n_blob = nT; xi_blob = xiT; btype = :T
    else
        zb = zbC; n_blob = nC; xi_blob = xiC; btype = :C
    end
    G = cui ? max(zb * n_blob / zeta_ratio, tiny) : (weak ? D : 1.0)
    return zeta_ratio, (G=G, blob_drag_ratio=zb, n_blob=n_blob, xi_blob=xi_blob,
                        blob_type=btype, g1d=1.0, s1d=NaN)
end

"""(oes, cui) switches of a v2 scheme symbol."""
@inline _v2_switches(drag_interp::Symbol) =
    (drag_interp in (:oesi, :oescui), drag_interp in (:oescui, :barecui))

"""
    oesi_dilute_weak_E(phi, E_max) -> E_b

Stretch at which a chain at concentration 0 < φ < 1 leaves the dilute regime of the v2
family: the root E > 1 of φE³ = 1 + ln(E/φ), i.e. where the regularised screening length χ
equals the chain length (OES note Eq. 3.4).  Bisection in ln E on [0, ln E_max];
f(ln E) = φE³ − 1 − ln E + ln φ is increasing for E > 1 when φE³ > 1/3, which holds
beyond the root region of interest.  Returns NaN if there is no root in (1, E_max)
(the chain is dilute up to full stretch) or if φ ∉ (0, 1).
"""
function oesi_dilute_weak_E(phi::Float64, E_max::Float64)::Float64
    (0.0 < phi < 1.0) || return NaN
    f(lE) = phi * exp(3lE) - 1.0 - lE + log(phi)
    lo, hi = 0.0, log(E_max)
    f(hi) <= 0.0 && return NaN          # dilute up to E_max
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) > 0.0 ? (hi = mid) : (lo = mid)
        hi - lo < 1e-15 && break
    end
    return exp(0.5 * (lo + hi))
end

"""
    _v2_cap_E(phi, hK_star, NK, draining_model, oes, p_phi, q_oes=0) -> E_cap

Stretch at which the Rouse cap of a CUI scheme engages for 1 < φ < √N_K: the smallest E in
(1, √N_K) where the uncapped blend (ζ_w^p + ζ_s^p)^(1/p) reaches the cap.  It can lie
below E = φ or above it (OESCUI, N_K = 5000, h*_K = 0.025: E_cap ≈ 34.5 for φ = 60,
≈ 53.2 for φ = 45).  The blend need not be monotone in E for the bare scheme (the draining
tension-blob drag can fall with E at small h*_K), so the first crossing is bracketed on a
dense log grid and then bisected.  NaN if the cap is not reached below E_max, or if the
chain is capped already at E = 1 (no interior corner).
"""
function _v2_cap_E(phi::Float64, hK_star::Float64, NK::Float64,
                   draining_model::Symbol, oes::Bool, p_phi::Float64,
                   q_oes::Float64 = DEFAULT_Q_OES)::Float64
    E_max  = sqrt(NK)
    alphaZ = _alpha_zimm(hK_star, NK, draining_model)
    cap    = _v2_cap(hK_star, NK, alphaZ, draining_model, oes)
    zeta_s = _v2_strong(phi, hK_star, NK, alphaZ, draining_model, oes)[1]
    uncapped(lE) = (_v2_weak(exp(lE), phi, hK_star, NK, alphaZ, draining_model, oes, q_oes)[1]^p_phi +
                    zeta_s^p_phi)^(1.0 / p_phi)
    grid = range(0.0, log(E_max); length = 2001)
    uncapped(grid[1]) >= cap && return NaN        # capped from E = 1 on: no interior corner
    k = findfirst(lE -> uncapped(lE) >= cap, grid)
    k === nothing && return NaN                   # cap not reached below E_max
    lo, hi = grid[k-1], grid[k]
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        uncapped(mid) >= cap ? (hi = mid) : (lo = mid)
        hi - lo < 1e-14 && break
    end
    return exp(0.5 * (lo + hi))
end

# ── LUT + assessor for fast evaluation ───────────────────────────────────────

"""
    DragAssessor

Precomputed piecewise-cubic (PCHIP-style) interpolant of ζ/ζ_Z(E) on a
log-spaced grid. Built once per run; queried at every spatial node and time step.

Handles the slope kinks at the regime boundaries by fitting separate cubics between them:
`kinks` (sorted, possibly empty) and one `(xs[i], ys[i])` piece per interval, piece i
covering E ∈ (kinks[i−1], kinks[i]] (`length(xs) == length(kinks) + 1`).  drag_model = :c2d2
only: RP/RPD/RPDW/RPDP kink at E = φ (φ > 1; RPD also at E = 1/φ for φ < 1); the v2 family
(OESCUI/OESI/BARECUI/BAREI) kinks at E = φ for φ > 1 and at the dilute/weak boundary E_b(φ)
for 0 < φ < 1 (`oesi_dilute_weak_E`), the OES schemes with q_oes > 0 also where the dressing
leaves ψ = 1, E_q = φ^(1/q) (`oesi_dressing_E`), and the CUI schemes at the Rouse-cap onset
when it is reached (if the cap engages below E = φ it is the only corner, the drag being
flat beyond it).  The CU scheme is smooth and uses a single piece.
"""
struct DragAssessor
    drag_model :: Symbol             # :constant | :c1d2 | :c2d2
    drag_interp :: Symbol            # :rp | :rpd | :rpdw | :rpdp | :cu | :oescui | :oesi | :barecui | :barei  (c2d2 only)
    p_phi    :: Float64              # CU / CUI exponent (unused otherwise)
    q_oes    :: Float64              # stretch-delayed OES dressing exponent (OES schemes only)
    g0_1D    :: Float64              # U_1D floor of g(s) (CU only; 1.0 = off)
    s_form   :: Symbol               # form of s(N_c, N_t) (CU only)
    E_min    :: Float64
    E_max    :: Float64
    kinks    :: Vector{Float64}          # sorted slope corners in E (empty if none)
    xs       :: Vector{Vector{Float64}}  # log10 grids, one piece per interval between corners
    ys       :: Vector{Vector{Float64}}
end

"""
    build_drag_assessor(phi, hK_star, NK, drag_model, draining_model;
                        n_points=400, drag_interp=:rp, p_phi=default_p_phi(drag_interp),
                        g0_1D=DEFAULT_G0_1D, s_form=DEFAULT_S_FORM, q_oes=DEFAULT_Q_OES) -> DragAssessor

Build a DragAssessor by tabulating drag_ratio_zimm on a log-spaced E grid
and fitting piecewise cubics around the slope kinks (drag_model = :c2d2: at E = φ for
RP/RPD/RPDW/RPDP and the v2 family with φ > 1, plus E_q = φ^(1/q_oes) for the OES schemes
and the Rouse-cap onset for the CUI schemes when reached; at E = 1/φ for RPD with φ < 1;
at the dilute/weak boundary E_b(φ) for the v2 family with φ < 1; the CU scheme is smooth
and is fitted as a single piece).
"""
function build_drag_assessor(phi::Float64, hK_star::Float64, NK::Float64,
                              drag_model::Symbol, draining_model::Symbol;
                              n_points::Int=400,
                              drag_interp::Symbol=:rp,
                              p_phi::Float64=default_p_phi(drag_interp),
                              g0_1D::Float64=DEFAULT_G0_1D,
                              s_form::Symbol=DEFAULT_S_FORM,
                              q_oes::Float64=DEFAULT_Q_OES)::DragAssessor
    _check_drag_interp(drag_interp, p_phi, g0_1D, s_form, q_oes)
    E_min = 1.0
    E_max = sqrt(NK)
    # log-spaced grid
    E_grid = exp.(range(log(E_min), log(E_max); length=n_points))
    y_grid = [drag_ratio_zimm(e, phi, hK_star, NK, drag_model, draining_model;
                              drag_interp=drag_interp, p_phi=p_phi,
                              g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)[1]
              for e in E_grid]

    # Locate kink: only c2d2 (RP scheme) with phi > 1 has a kink at E = phi
    has_kink = (drag_model === :c2d2) && (drag_interp in (:rp, :rpd)) &&
               (phi > 1.0) && (phi < E_max)
    kink_E   = has_kink ? phi : NaN
    # :rpd also has a slope kink where the dressing saturates, ψ = φE = 1 (φ < 1).
    # :rpdw saturates at E = φ, the existing kink, so nothing extra is needed.
    if drag_model === :c2d2 && drag_interp in (:rpdw, :rpdp) && (phi > 1.0) && (phi < E_max)
        has_kink = true
        kink_E   = phi
    end
    if drag_model === :c2d2 && drag_interp === :rpd && 0.0 < phi < 1.0 && 1.0 / phi < E_max
        has_kink = true
        kink_E   = 1.0 / phi
    end
    # v2 family (2026-10-21/22): one slope corner per φ — at E = φ for 1 < φ < E_max
    # (weak/strong switch; A = min(E², χ/ξ_t) has its corner there for every scheme), at
    # the dilute/weak boundary E_b(φ) for 0 < φ < 1.  φ = 0 and φ = 1 have no corner (with
    # oes off there is no ψ corner at φ = 1 either).
    if drag_model === :c2d2 && drag_interp in _V2_SCHEMES
        oes, cui = _v2_switches(drag_interp)
        corners = Float64[]
        if phi > 1.0 && phi < E_max
            # Since D(A) has slope 1/2 at A = 1 (the old 1 + ln J was flat there), E = φ is a
            # genuine slope corner for all four schemes, the CUI ones included.  With q_oes > 0
            # the OES dressing ψ = min(φ/E^q, 1) leaves 1 at E_q = φ^(1/q) > φ (second corner).
            # For the CUI schemes the Rouse cap min(·, cap) is a further corner once it engages
            # below E_max (OESCUI: φ ≳ 37 for N_K = 5000); the drag is flat beyond it, so any
            # corner above the cap onset disappears (and if the cap engages below E = φ it is
            # the only corner).
            E_cap = cui ? _v2_cap_E(phi, hK_star, NK, draining_model, oes, p_phi, q_oes) : NaN
            E_q   = oes ? oesi_dressing_E(phi, q_oes, E_max) : NaN
            for c in (phi, E_q, E_cap)
                isfinite(c) || continue
                isfinite(E_cap) && c > E_cap && continue
                push!(corners, c)
            end
        elseif 0.0 < phi < 1.0
            E_b = oesi_dilute_weak_E(phi, E_max)
            isfinite(E_b) && 1.0 < E_b < E_max && push!(corners, E_b)
        end
        sort!(corners)
        # drop (near-)coincident corners: a two-node piece would be a plain secant
        corners = [c for (i, c) in enumerate(corners) if i == 1 || c > corners[i-1] * (1 + 1e-9)]
        if !isempty(corners)
            # Split at the corners themselves (inserted as grid nodes), so no piece contains a
            # segment straddling one; the straddling-segment split used by the RP family is
            # left untouched for bit-identity.
            f(e) = drag_ratio_zimm(e, phi, hK_star, NK, drag_model, draining_model;
                                   drag_interp=drag_interp, p_phi=p_phi,
                                   g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)[1]
            xs = Vector{Float64}[]; ys = Vector{Float64}[]
            for i in 1:length(corners)+1
                lo = i == 1 ? -Inf : corners[i-1]
                hi = i == length(corners)+1 ? Inf : corners[i]
                nodes = vcat(isfinite(lo) ? [lo] : Float64[],
                             E_grid[lo .< E_grid .< hi],
                             isfinite(hi) ? [hi] : Float64[])
                push!(xs, log10.(nodes)); push!(ys, f.(nodes))
            end
            return DragAssessor(drag_model, drag_interp, p_phi, q_oes, g0_1D, s_form, E_min, E_max,
                                corners, xs, ys)
        end
    end

    if has_kink
        kink_idx = searchsortedfirst(E_grid, kink_E)
        kink_idx = clamp(kink_idx, 2, length(E_grid)-1)
        L = 1:kink_idx
        R = kink_idx:length(E_grid)
        return DragAssessor(drag_model, drag_interp, p_phi, q_oes, g0_1D, s_form, E_min, E_max,
                            [kink_E], [log10.(E_grid[L]), log10.(E_grid[R])], [y_grid[L], y_grid[R]])
    else
        return DragAssessor(drag_model, drag_interp, p_phi, q_oes, g0_1D, s_form, E_min, E_max,
                            Float64[], [log10.(E_grid)], [y_grid])
    end
end

"""
    eval_drag(assessor, E) -> zeta_ratio

Evaluate ζ/ζ_Z at stretch E using the prebuilt piecewise cubic interpolant.
E is clamped to [E_min, E_max] — no out-of-domain error.
"""
function eval_drag(a::DragAssessor, E::Float64)::Float64
    E = clamp(E, a.E_min, a.E_max)
    i = searchsortedfirst(a.kinks, E)          # piece i covers (kinks[i-1], kinks[i]]
    return _pchip_eval(a.xs[i], a.ys[i], log10(E))
end

"""
    eval_drag_deriv(assessor, E) -> d(zeta_ratio)/dE

First derivative of ζ/ζ_Z with respect to E (chain rule through log10).
"""
function eval_drag_deriv(a::DragAssessor, E::Float64)::Float64
    E = clamp(E, a.E_min, a.E_max)
    i = searchsortedfirst(a.kinks, E)
    dydx = _pchip_deriv(a.xs[i], a.ys[i], log10(E))
    # chain rule: dy/dE = (dy/d log10 E) * (d log10 E / dE) = dydx / (E * log(10))
    return dydx / (E * log(10))
end

# ── Minimal PCHIP implementation ──────────────────────────────────────────────
# Fritsch-Carlson monotone cubic interpolation.
# We implement this directly to avoid a heavy dependency for a simple 1D spline.

"""Evaluate a monotone cubic interpolant at xq (scalar). xs must be sorted."""
function _pchip_eval(xs::Vector{Float64}, ys::Vector{Float64}, xq::Float64)::Float64
    n = length(xs)
    n >= 2 || return ys[1]
    # clamp to domain
    xq = clamp(xq, xs[1], xs[end])
    # binary search for interval
    k = searchsortedfirst(xs, xq) - 1
    k = clamp(k, 1, n-1)
    h  = xs[k+1] - xs[k]
    h  == 0.0 && return ys[k]
    t  = (xq - xs[k]) / h
    d  = _pchip_slopes(xs, ys)
    # Hermite basis
    h00 = (1 + 2t) * (1-t)^2
    h10 = t * (1-t)^2
    h01 = t^2 * (3 - 2t)
    h11 = t^2 * (t - 1)
    return h00*ys[k] + h10*h*d[k] + h01*ys[k+1] + h11*h*d[k+1]
end

"""Evaluate the derivative of the monotone cubic interpolant at xq."""
function _pchip_deriv(xs::Vector{Float64}, ys::Vector{Float64}, xq::Float64)::Float64
    n = length(xs)
    n >= 2 || return 0.0
    xq = clamp(xq, xs[1], xs[end])
    k  = clamp(searchsortedfirst(xs, xq) - 1, 1, n-1)
    h  = xs[k+1] - xs[k]
    h == 0.0 && return 0.0
    t  = (xq - xs[k]) / h
    d  = _pchip_slopes(xs, ys)
    # derivatives of Hermite basis wrt t, then divide by h
    dh00 = 6t*(t-1)
    dh10 = (1-t)*(1-3t)
    dh01 = 6t*(1-t)
    dh11 = t*(3t-2)
    return (dh00*ys[k] + dh10*h*d[k] + dh01*ys[k+1] + dh11*h*d[k+1]) / h
end

"""Fritsch-Carlson monotone slopes."""
function _pchip_slopes(xs::Vector{Float64}, ys::Vector{Float64})::Vector{Float64}
    n = length(xs)
    d = zeros(n)
    # secant slopes
    δ = diff(ys) ./ diff(xs)
    # endpoint slopes (one-sided)
    d[1]   = δ[1]
    d[end] = δ[end]
    # interior slopes via harmonic mean, zeroed at sign changes
    for k in 2:n-1
        if δ[k-1] * δ[k] <= 0
            d[k] = 0.0
        else
            w1 = 2*( xs[k+1]-xs[k]) + (xs[k]-xs[k-1])
            w2 = 2*(  xs[k]-xs[k-1]) + (xs[k+1]-xs[k])
            d[k] = (w1 + w2) / (w1/δ[k-1] + w2/δ[k])
        end
    end
    return d
end

end # module Drag
