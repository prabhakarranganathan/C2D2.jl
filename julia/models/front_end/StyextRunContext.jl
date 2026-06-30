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
using ..Drag:        build_drag_assessor, drag_draining_ratio, drag_ratio_zimm
using ..Spectrum:    build_spectrum_assessor
using ..CoeffsRelax: build_coeffs_kernel
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

    fe_model in (:fenepme, :fenepm, :fenep) ||
        error("fe_model must be fenepme/fenepm/fenep. Got: $fe_model")
    drag_model in (:constant, :c1d2, :c2d2) ||
        error("drag_model must be constant/c1d2/c2d2. Got: $drag_model")
    draining_model in (:partialdraining, :nondraining) ||
        error("draining_model must be partialdraining/nondraining. Got: $draining_model")

    # ── Flow ──────────────────────────────────────────────────────────────────
    av = get(flow_sec, "alphas", [-0.5, -0.5, 1.0])
    length(av) == 3 || error("flow.alphas must have 3 entries")
    alphas = (Float64(av[1]), Float64(av[2]), Float64(av[3]))
    maximum(alphas) > 0 || error("flow.alphas must have at least one positive entry")

    # ── Limits ────────────────────────────────────────────────────────────────
    Wi_max_raw = get(lim_sec, "Wi_max", 3.0)
    Wi_max = Wi_max_raw == "Inf" ? Inf :
             Wi_max_raw isa String ? parse(Float64, Wi_max_raw) : Float64(Wi_max_raw)
    Wi_max > 0 || error("limits.Wi_max must be > 0")

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

    # FENE pole: STYEXT convention (differs from VELB for Nm>1)
    Qinf_sq = fe_model === :fenepme ? Q0_sq * NK : Q0_sq * NK / Nm

    c_star = E0^(-3)
    conc   = phi * c_star
    ckBT   = conc * kBT

    # ── Build drag LUT and spectrum assessor ──────────────────────────────────
    drag_lut = build_drag_assessor(phi, hK_star, NK, drag_model, draining_model)

    spectrum = build_spectrum_assessor(
        spectrum_model = spectrum_model,
        Nm             = Nm,
        NK             = NK,
        hK_star        = hK_star,
        phi            = phi,
        drag_model     = drag_model,
        draining_model = draining_model,
    )

    # ── Equilibrium drag ratio → lambdaZ → etas ───────────────────────────────
    # Equilibrium: E = 1 (coiled state; matches MATLAB's r=sqrt(Y)/E0=1)
    Gam_eq, _ = drag_ratio_zimm(1.0, phi, hK_star, NK, drag_model, draining_model)
    Gam_eq > 0 || error("Equilibrium drag ratio must be > 0; got $Gam_eq")
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
        phi, fe_model, drag_model, draining_model, spectrum_model,
        alphas, Wi_max, max_frac_stretch_sq,
        E0_sq, Einf_sq, Q0, E0, Qinf_sq,
        ckBT, etas, Gam_eq, lambdaZ, Sp,
    )

    # ── Runtime StyextMPCtx ───────────────────────────────────────────────────
    # Ymax encodes the Y_stop = max_frac_stretch_sq * Einf_sq
    Ymax_ctx = max_frac_stretch_sq * Einf_sq

    mp_ctx = build_mp_ctx(
        kernel, Sp, alphas, Na, Nm,
        E0_sq, Ymax_ctx,
        Q0_sq, Qinf_sq,
        lambda0, etas, ckBT,
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
    )

    return params, mp_ctx, solver_opts, man_opts
end

end # module StyextRunContext
