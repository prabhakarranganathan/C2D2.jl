"""
    StyextRunContext.jl

Context assembly for the STYEXT (steady extension) PAC solver.

Mirrors MATLAB's styext_run_context.m.

Reads a TOML file and builds:
  - StyextParams  : portable (serialisable) parameter struct
  - StyextMPCtx   : runtime model-pack context (kernel + derived scales)
  - SolverOpts    : Newton + Armijo controls
  - ManifoldOpts  : PAC step-size + termination controls

Key differences from the VELB RuntimeCtx (Params.jl):
  - Qinf_sq for FENE-PM / FENE-P = Q₀² · NK / Nm  (VELB uses NK always)
  - Z.E normalization in eval_coeffs! uses √Y/E₀ (equilibrium → E=1)
  - ηₛ from Zimm formula: ηₛ = λ_Z · k_B·T / (U · E₀³)
  - phi read from phi_values[1] if phi not explicit (TOML compat)
  - spectrum "auto" → :rouse_full (constant drag) or :blob (c2d2 drag)

Public API
----------
  StyextParams           — portable parameter struct
  load_styext_context    — parse TOML, assemble all contexts
"""
module StyextRunContext

import TOML

using ..Fene:        build_fene_model, na_for_model
using ..Drag:        build_drag_assessor, drag_draining_ratio, drag_ratio_zimm,
                     parse_drag_interp, DEFAULT_P_PHI, default_p_phi, DEFAULT_Q_OES,
                     parse_s_form, DEFAULT_G0_1D, DEFAULT_S_FORM, cu_lambda0_ratio
using ..Spectrum:    build_spectrum_assessor
using ..CoeffsRelax: build_coeffs_kernel
using ..StyextFlows: PrincipalExtension, PlanarMixedFlow
using ..StyextModelPack: StyextMPCtx, build_mp_ctx
using ..StyextNewton:    SolverOpts
using ..StyextManifold:  ManifoldOpts

export StyextParams, load_styext_context

# ── Portable parameter struct ─────────────────────────────────────────────────

"""
    StyextParams

All user-facing parameters for a STYEXT run.
Loaded from a TOML file; all fields are plain scalars / strings.
Safe to serialise.
"""
struct StyextParams
    run_label           :: String
    NK                  :: Float64
    Nm                  :: Int
    hK_star             :: Float64
    Q0_sq               :: Float64
    kBT                 :: Float64
    lambda0             :: Float64
    phi                 :: Float64
    fe_model            :: Symbol
    drag_model          :: Symbol
    draining_model      :: Symbol
    drag_interp         :: Symbol        # :rp | :rpd | :rpdw | :rpdp | :cu | :oescui | :oesi | :barecui | :barei  (c2d2 only)
    p_phi               :: Float64       # CU / CUI exponent (read only for those schemes)
    spectrum_model      :: Symbol
    alphas              :: NTuple{3, Float64}
    Wi_max              :: Float64
    max_frac_stretch_sq :: Float64
    E0_sq               :: Float64
    Einf_sq             :: Float64
    Q0                  :: Float64
    E0                  :: Float64
    Qinf_sq             :: Float64
    ckBT                :: Float64
    etas                :: Float64
    Gam_eq              :: Float64
    lambdaZ             :: Float64
    Sp                  :: Vector{Float64}
    # ── steady planar mixed flow (2026-10-16) ─────────────────────────────────
    flow_kind           :: Symbol            # :styext | :pmf
    r                   :: Float64           # PMF flow-type (vorticity/strain); NaN for :styext
    Wi_convention       :: Symbol            # :strain (report Wi) | :shear (λ₀γ̇ = 2·Wi; r = 1 only)
    Wi_sweep            :: Vector{Float64}   # [sweep] Wi grid, strain convention (empty if absent)
    drag_lut_n_points   :: Int               # [model] drag_lut_n_points (steady default 4000; kernel default 400)
    # ── U_1D relaxation factor + overlap constant (2026-10-17) ────────────────
    g0_1D               :: Float64           # CU only; 1.0 = off
    s_form              :: Symbol            # CU only; :sum (smooth N_c′) | :sum_hard
    C_phi               :: Float64           # c = φ/(C_phi E₀³); divides ckBT (all schemes)
    De_mode             :: Symbol            # :input | :cu_fit (Gam_eq = λ₀/λ_Z = (1+φ^p)^(1/p))
    q_oes               :: Float64           # stretch-delayed OES dressing exponent (OESI/OESCUI only; 0 = off)
end

# ── TOML section helpers ──────────────────────────────────────────────────────

_sec(d, s)          = get(d, s, Dict{String, Any}())
_getf(d, k, def)    = Float64(get(d, k, def))
_geti(d, k, def)    = Int(round(Float64(get(d, k, def))))
_gets(d, k, def)    = String(get(d, k, def))
_getb(d, k, def)    = Bool(get(d, k, def))

# ── spectrum_model "auto" resolution ─────────────────────────────────────────

const _SPEC_MAP = Dict(
    "blob"             => :blob,
    "rouse-full"       => :rouse_full,
    "rouse_full"       => :rouse_full,
    "rouse-analytical" => :rouse_full,
    "rouse-scaling"    => :rouse_scaling,
    "rouse_scaling"    => :rouse_scaling,
    "zimm-scaling"     => :zimm_scaling,
    "zimm_scaling"     => :zimm_scaling,
    "zimm-thurston"    => :zimm_thurston,
    "zimm_thurston"    => :zimm_thurston,
)

function _resolve_spectrum(spec_str::String, drag_str::String)::Symbol
    spec_low = lowercase(strip(spec_str))
    drag_low = lowercase(strip(drag_str))

    if spec_low == "auto"
        if drag_low in ("constant", "c1d2")
            return :rouse_full
        elseif drag_low == "c2d2"
            return :blob
        else
            error("Cannot auto-resolve spectrum_model for drag_model = \"$drag_str\"")
        end
    end

    sym = get(_SPEC_MAP, spec_low, nothing)
    sym !== nothing || error("Unknown spectrum_model: \"$spec_str\"")
    return sym
end

# ── Rouse end-to-end weights (duplicate of Params._rouse_weights) ─────────────

function _rouse_weights(Nm::Int)::Vector{Float64}
    Sp = zeros(Nm)
    for p in 1:Nm
        if isodd(p)
            th = p * π / (2*(Nm+1))
            Sp[p] = (2.0 / (Nm+1)) * cot(th)^2
        end
    end
    return Sp
end

# ── Solvent viscosity from Zimm formula ───────────────────────────────────────

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

# ── Main context loader ───────────────────────────────────────────────────────

"""
    load_styext_context(toml_path; phi=nothing)
        -> (StyextParams, StyextMPCtx, SolverOpts, ManifoldOpts)

Parse a STYEXT TOML file and assemble the full runtime context.

If `phi` is provided it overrides the TOML concentration value.
"""
function load_styext_context(toml_path :: AbstractString;
                              phi       :: Union{Float64, Nothing} = nothing)

    d = TOML.parsefile(toml_path)

    run_sec  = _sec(d, "run")
    poly_sec = _sec(d, "polymer")
    conc_sec = _sec(d, "concentration")
    mod_sec  = _sec(d, "model")
    flow_sec = _sec(d, "flow")
    lim_sec  = _sec(d, "limits")
    cont_sec = _sec(d, "continuation")
    newt_sec = _sec(d, "newton")
    fy_sec   = _sec(d, "newton_fixed_Y")

    # ── Identity ──────────────────────────────────────────────────────────────
    run_label = _gets(run_sec, "label", "styext")

    # ── Polymer ───────────────────────────────────────────────────────────────
    NK      = _getf(poly_sec, "NK",      5000.0)
    Nm      = _geti(poly_sec, "Nm",      1)
    hK_star = _getf(poly_sec, "hK_star", 0.25)
    Q0_sq   = _getf(poly_sec, "Q0_sq",   1.0)
    kBT     = _getf(poly_sec, "kBT",     1.0)
    lambda0 = _getf(poly_sec, "lambda0", 1.0)

    isfinite(NK) && NK > 0 || error("polymer.NK must be finite and positive")
    Nm >= 1 || error("polymer.Nm must be ≥ 1")
    0.0 < hK_star < 0.5 || error("polymer.hK_star must be in (0, 0.5)")
    Q0_sq > 0 || error("polymer.Q0_sq must be > 0")
    kBT   > 0 || error("polymer.kBT must be > 0")
    lambda0 > 0 || error("polymer.lambda0 must be > 0")

    # ── Concentration ─────────────────────────────────────────────────────────
    if phi === nothing
        phi_direct = get(conc_sec, "phi", nothing)
        if phi_direct !== nothing
            phi = Float64(phi_direct)
        else
            phi_values = get(conc_sec, "phi_values", nothing)
            if phi_values !== nothing && !isempty(phi_values)
                phi = Float64(phi_values[1])
                length(phi_values) > 1 &&
                    @warn "phi_values has $(length(phi_values)) entries; using first: phi=$phi"
            else
                phi = 1.0
            end
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

    fe_model in (:fenepme, :fenepm, :fenep, :hookean) ||
        error("fe_model must be fenepme/fenepm/fenep/hookean. Got: $fe_model")
    drag_model in (:constant, :c1d2, :c2d2) ||
        error("drag_model must be constant/c1d2/c2d2. Got: $drag_model")
    draining_model in (:partialdraining, :nondraining) ||
        error("draining_model must be partialdraining/nondraining. Got: $draining_model")

    # drag_interp: "RP"/"RPI" (default; bit-identical to earlier inputs), "OESCUI", "OESI",
    # "BARECUI", "BAREI" (and the archived "RPD"/"RPDW"/"RPDP"/"CU").  p_phi is read only for
    # "CU"/"OESCUI"/"BARECUI"; otherwise at the scheme's default (1.86 CU, 2.0 CUI).
    drag_interp = parse_drag_interp(_gets(mod_sec, "drag_interp", "RP"))
    p_phi       = drag_interp in (:cu, :oescui, :barecui) ? _getf(mod_sec, "p_phi", default_p_phi(drag_interp)) :
                                                            default_p_phi(drag_interp)
    drag_interp in (:cu, :oescui, :barecui) && !(isfinite(p_phi) && p_phi > 0) &&
        error("model.p_phi must be finite and > 0 for drag_interp = \"CU\"/\"OESCUI\"/\"BARECUI\". Got: $p_phi")
    # g0_1D / s_form (CU only; 2026-10-17): U_1D relaxation factor g(s). C_phi: overlap
    # constant in c = φ/(C_phi E₀³), read for all schemes. Defaults are bit-identical.
    g0_1D  = drag_interp === :cu ? _getf(mod_sec, "g0_1D", DEFAULT_G0_1D) : DEFAULT_G0_1D
    s_form = (drag_interp === :cu && g0_1D != 1.0) ?
                 parse_s_form(_gets(mod_sec, "s_form", "sum")) : DEFAULT_S_FORM
    drag_interp === :cu && !(isfinite(g0_1D) && g0_1D > 0) &&
        error("model.g0_1D must be finite and > 0 for drag_interp = \"CU\" (1.0 = off). Got: $g0_1D")
    # q_oes (2026-10-22 pm): stretch-delayed OES dressing exponent, ψ = min(φ/E^q, 1); OESI/OESCUI
    # only, default 0 (bit-identical). The Thin-n-Win runs set 0.25 explicitly.
    is_oes = drag_interp in (:oesi, :oescui)
    q_oes  = is_oes ? _getf(mod_sec, "q_oes", DEFAULT_Q_OES) : DEFAULT_Q_OES
    !is_oes && haskey(mod_sec, "q_oes") &&
        @warn "model.q_oes is read only for drag_interp = \"OESI\"/\"OESCUI\"; ignored for $(drag_interp)"
    is_oes && !(isfinite(q_oes) && 0.0 <= q_oes <= 1.0) &&
        error("model.q_oes must be finite and in [0, 1] for drag_interp = \"OESI\"/\"OESCUI\". Got: $q_oes")
    C_phi = _getf(mod_sec, "C_phi", 1.0)
    (isfinite(C_phi) && C_phi > 0) || error("model.C_phi must be finite and > 0. Got: $C_phi")
    # De_mode (2026-10-17 pm): "input" keeps λ₀/λ_Z = ζ₀/ζ_Z from the drag model; "cu_fit"
    # anchors it to RP's ansatz λ₀/λ_Z = (1 + φ^p)^(1/p), p = p_phi (fitted to SAOS).
    De_mode_str = lowercase(_gets(mod_sec, "De_mode", "input"))
    De_mode = De_mode_str == "input" ? :input : De_mode_str == "cu_fit" ? :cu_fit :
              error("model.De_mode must be \"input\" or \"cu_fit\". Got: \"$De_mode_str\"")
    if De_mode === :cu_fit && !(drag_interp in (:cu, :oescui, :barecui))
        p_phi = _getf(mod_sec, "p_phi", DEFAULT_P_PHI)
        (isfinite(p_phi) && p_phi > 0) || error("model.p_phi must be finite and > 0 for De_mode = \"cu_fit\". Got: $p_phi")
    end

    # ── Flow ──────────────────────────────────────────────────────────────────
    # kind = "styext" (default; principal extension, alphas) | "pmf" (planar
    # mixed flow, κ = Wi·κ̂(r); r = 1 simple shear). For "pmf" the alphas are
    # the strain-axes frame (1, −1, 0) and any TOML alphas are ignored.
    flow_kind_str = lowercase(_gets(flow_sec, "kind", "styext"))
    flow_kind = flow_kind_str in ("styext", "ext", "extension") ? :styext :
                flow_kind_str in ("pmf", "shear", "mixed")       ? :pmf    :
                error("flow.kind must be \"styext\" or \"pmf\". Got: \"$flow_kind_str\"")

    if flow_kind === :pmf
        r = _getf(flow_sec, "r", 1.0)
        isfinite(r) && r >= 0 || error("flow.r must be finite and ≥ 0 for kind = \"pmf\". Got: $r")
        alphas = (1.0, -1.0, 0.0)
        haskey(flow_sec, "alphas") &&
            @warn "flow.alphas ignored for kind = \"pmf\" (strain-axes frame (1,-1,0) is implied)"
        Wi_conv_str = lowercase(_gets(flow_sec, "Wi_convention", "strain"))
        Wi_convention = Wi_conv_str == "strain" ? :strain :
                        Wi_conv_str == "shear"  ? :shear  :
                        error("flow.Wi_convention must be \"strain\" or \"shear\". Got: \"$Wi_conv_str\"")
        Wi_convention === :shear && r != 1.0 &&
            error("flow.Wi_convention = \"shear\" (Wi = λ₀γ̇) is only defined for simple shear r = 1. Got r = $r")
        flow = PlanarMixedFlow(r)
    else
        r = NaN
        Wi_convention = :strain
        av = get(flow_sec, "alphas", [-0.5, -0.5, 1.0])
        length(av) == 3 || error("flow.alphas must have 3 entries")
        alphas = (Float64(av[1]), Float64(av[2]), Float64(av[3]))
        maximum(alphas) > 0 || error("flow.alphas must have at least one positive entry")
        flow = PrincipalExtension(alphas)
    end
    # Input Wi values (limits.Wi_max, [sweep]) are divided by 2 under the
    # "shear" convention so that the solver always works in report Wi.
    Wi_in_scale = Wi_convention === :shear ? 0.5 : 1.0

    # ── Limits ────────────────────────────────────────────────────────────────
    Wi_max_raw = get(lim_sec, "Wi_max", 3.0)
    Wi_max = Wi_max_raw == "Inf" ? Inf :
             Wi_max_raw isa String ? parse(Float64, Wi_max_raw) : Float64(Wi_max_raw)
    Wi_max > 0 || error("limits.Wi_max must be > 0")
    Wi_max *= Wi_in_scale

    # ── [sweep]: fixed-Wi grid (optional) ─────────────────────────────────────
    # Either an explicit list `Wi_values = [...]` or a log grid from
    # `Wi_min`, `Wi_max`, `n_per_decade` (default 10). Stored in report Wi.
    sweep_sec = _sec(d, "sweep")
    Wi_sweep = Float64[]
    if haskey(sweep_sec, "Wi_values")
        Wi_sweep = Float64.(collect(sweep_sec["Wi_values"]))
    elseif haskey(sweep_sec, "Wi_min") && haskey(sweep_sec, "Wi_max")
        w0 = _getf(sweep_sec, "Wi_min", 1e-2);  w1 = _getf(sweep_sec, "Wi_max", 1e2)
        npd = _getf(sweep_sec, "n_per_decade", 10.0)
        w0 > 0 && w1 > w0 && npd > 0 || error("[sweep] needs 0 < Wi_min < Wi_max and n_per_decade > 0")
        n = Int(round(npd * log10(w1 / w0))) + 1
        Wi_sweep = exp10.(range(log10(w0), log10(w1); length = n))
    end
    all(w -> isfinite(w) && w >= 0, Wi_sweep) || error("[sweep] Wi values must be finite and ≥ 0")
    Wi_sweep .*= Wi_in_scale

    # ── Drag LUT resolution ───────────────────────────────────────────────────
    # Default 4000 for the steady loader since 2026-10-16 (RP/Aru, KMR grid):
    # the 400-point table (still the kernel/VELB default in build_drag_assessor)
    # limits c2d2 accuracy to ~1e-5 near E ≈ 1. Set 400 explicitly to reproduce
    # pre-2026-10-16 steady outputs bit-for-bit.
    drag_lut_n_points = _geti(mod_sec, "drag_lut_n_points", 4000)
    drag_lut_n_points >= 4 || error("model.drag_lut_n_points must be ≥ 4")

    max_frac_stretch_sq = _getf(lim_sec, "max_frac_stretch_sq", 0.999)
    0.0 < max_frac_stretch_sq < 1.0 ||
        error("limits.max_frac_stretch_sq must be in (0, 1)")

    # ── Derived scales ────────────────────────────────────────────────────────
    Q0      = sqrt(Q0_sq)
    NKs     = NK / Nm           # segments per mode
    bK      = Q0 / sqrt(NKs)   # modal spring length

    E0      = bK * sqrt(NK)    # = Q0 * sqrt(Nm)  (for Q0_sq=1)
    E0_sq   = E0^2             # = Q0_sq * Nm

    Einf    = bK * NK          # = Q0 * sqrt(NK*Nm)
    Einf_sq = Einf^2           # = Q0_sq * NK * Nm

    # FENE pole: STYEXT convention (differs from VELB for Nm>1).
    #   FENE-PME : Q∞² = Q₀²·NK        (whole-chain argument Y/Nm)
    #   FENE-PM/P: Q∞² = Q₀²·NK/Nm     (per-spring argument)
    #   Hookean  : no pole; value stored for bookkeeping only (f ≡ 1).
    # Params.jl (VELB) uses Q₀²·NK for every model, so the two paths silently
    # differ for Nm > 1. Reference data (MATLAB STYEXT, Python PMF port) follow
    # THIS convention.
    Qinf_sq = fe_model in (:fenepme, :hookean) ? Q0_sq * NK : Q0_sq * NK / Nm

    c_star = E0^(-3)
    conc   = phi * c_star / C_phi      # c = φ/(C_phi E₀³)  (C_phi = 1: historical)
    ckBT   = conc * kBT

    # ── Build drag LUT and spectrum assessor ──────────────────────────────────
    drag_lut = build_drag_assessor(phi, hK_star, NK, drag_model, draining_model;
                                   n_points=drag_lut_n_points,
                                   drag_interp=drag_interp, p_phi=p_phi,
                                   g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)

    spectrum = build_spectrum_assessor(
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

    # ── Equilibrium drag ratio → lambdaZ → etas ───────────────────────────────
    # Equilibrium: E = 1 (coiled state; matches MATLAB's r=sqrt(Y)/E0=1)
    Gam_eq, _ = drag_ratio_zimm(1.0, phi, hK_star, NK, drag_model, draining_model;
                                drag_interp=drag_interp, p_phi=p_phi,
                                g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)   # λ₀/λ_Z incl. g(s₀) for CU
    Gam_eq > 0 || error("Equilibrium drag ratio must be > 0; got $Gam_eq")
    # λ₀/λ_Z: model-anchored (1+φ^p)^(1/p) under cu_fit; otherwise the drag-model value
    # (ζ₀/ζ_Z incl. g(s₀) for CU). Sets ηₛ via λ_Z and hence the Trouton ratio scale.
    De_mode === :cu_fit && (Gam_eq = cu_lambda0_ratio(phi, p_phi))
    lambdaZ = lambda0 / Gam_eq

    etas = _etas_from_lambdaZ(lambdaZ, kBT, E0, hK_star, NK, draining_model)

    # ── Rouse weights ─────────────────────────────────────────────────────────
    Sp = _rouse_weights(Nm)

    # ── FENE model with STYEXT-correct Qinf_sq ────────────────────────────────
    fene = build_fene_model(fe_model, Nm, Q0_sq, Qinf_sq)

    # ── CoeffsKernel ─────────────────────────────────────────────────────────
    kernel = build_coeffs_kernel(fene, drag_lut, spectrum)

    Na = na_for_model(fe_model, Nm)

    # ── Portable StyextParams ─────────────────────────────────────────────────
    params = StyextParams(
        run_label, NK, Nm, hK_star, Q0_sq, kBT, lambda0,
        phi, fe_model, drag_model, draining_model, drag_interp, p_phi, spectrum_model,
        alphas, Wi_max, max_frac_stretch_sq,
        E0_sq, Einf_sq, Q0, E0, Qinf_sq,
        ckBT, etas, Gam_eq, lambdaZ, Sp,
        flow_kind, r, Wi_convention, Wi_sweep, drag_lut_n_points,
        g0_1D, s_form, C_phi, De_mode, q_oes,
    )

    # ── Runtime StyextMPCtx ───────────────────────────────────────────────────
    # Ymax encodes the Y_stop = max_frac_stretch_sq * Einf_sq
    Ymax_ctx = max_frac_stretch_sq * Einf_sq

    mp_ctx = build_mp_ctx(
        kernel, Sp, alphas, Na, Nm,
        E0_sq, Ymax_ctx,
        Q0_sq, Qinf_sq,
        lambda0, etas, ckBT;
        flow = flow,
    )

    # ── SolverOpts ────────────────────────────────────────────────────────────
    use_armijo_raw = get(newt_sec, "use_armijo", true)
    use_armijo     = use_armijo_raw isa Bool ? use_armijo_raw : Bool(Int(Float64(use_armijo_raw)))

    solver_opts = SolverOpts(
        tol_res         = _getf(newt_sec, "tol_res",    1e-8),
        max_newton      = _geti(newt_sec, "max_newton", 15),
        max_newton_fy   = _geti(fy_sec,   "max_newton_fixed_Y",
                                 _geti(newt_sec, "max_newton", 15)),
        min_denom_tol   = _getf(newt_sec, "min_denom_tol",  0.0),
        tol_J_cond      = _getf(newt_sec, "tol_J_cond",  1e-12),
        use_armijo      = use_armijo,
        armijo_c1       = _getf(newt_sec, "armijo_c1",  1e-4),
        armijo_beta     = _getf(newt_sec, "armijo_beta", 0.5),
        max_bt          = _geti(newt_sec, "max_bt", 30),
        tol_FY_rel_hard = _getf(fy_sec,   "tol_FY_rel_hard", 100.0),
        tol_FY_rel_soft = _getf(fy_sec,   "tol_FY_rel_soft", 1e-2),
        target_newton   = _geti(newt_sec, "target_newton", 10),
        target_rho      = _getf(newt_sec, "target_offmanifold_ratio", 0.3),
    )

    # ── ManifoldOpts ──────────────────────────────────────────────────────────
    max_steps_val = begin
        ms_cont = get(cont_sec, "max_steps", nothing)
        ms_lim  = get(lim_sec,  "max_steps", nothing)
        if ms_cont !== nothing
            Int(round(Float64(ms_cont)))
        elseif ms_lim !== nothing
            Int(round(Float64(ms_lim)))
        else
            5000
        end
    end

    man_opts = ManifoldOpts(
        ds0             = _getf(cont_sec, "ds0",    1e-3),
        ds_min          = _getf(cont_sec, "ds_min", 1e-8),
        ds_max          = _getf(cont_sec, "ds_max", 1e3),
        max_steps       = max_steps_val,
        Wi_max          = Wi_max,
        tol_FY_rel_hard = _getf(fy_sec, "tol_FY_rel_hard", 100.0),
        target_newton   = _geti(newt_sec, "target_newton", 10),
        target_rho      = _getf(newt_sec, "target_offmanifold_ratio", 0.3),
        verbose         = false,
        # X-closure stop (FENE-PM/P only): same fraction as Y_stop, applied to Q∞².
        # For Nm = 1, X ≡ Y so this coincides with Y_stop; for Nm > 1 it is the
        # criterion that actually fires (Y saturates below E∞²). Hookean/PME: off.
        X_stop          = Na > 0 ? max_frac_stretch_sq * Qinf_sq : Inf,
    )

    return params, mp_ctx, solver_opts, man_opts
end

end # module StyextRunContext
