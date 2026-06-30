"""
    UsyextRunContext.jl

Context assembly for the USYEXT (unsteady extension) ETDRK2 integrator.

Mirrors MATLAB's usyext_run_context.m.

Reads a TOML file and builds:
  - UsyextCtx       : runtime polymer model context (kernel + derived scales)
  - Flow            : ConstantWiFlow | CaberFlow
  - IntegrationPlan : compiled sampling schedule
  - TolOpts         : adaptive controller tolerances

Reuses the same polymer/drag/spectrum wiring as StyextRunContext
(same kernel, same E₀ normalisation, same ηₛ formula).

TOML differences from STYEXT:
  - flow.kind = "caber" | "constant"
  - flow.De (or flow.De_values[1]) instead of limits.Wi_max
  - [integration] section for the ODE controller
  - [output] section for the sampling schedule

Public API
----------
  load_usyext_context   — parse TOML → (UsyextCtx, flow, IntegrationPlan, TolOpts)
"""
module UsyextRunContext

import TOML

using ..Fene:           build_fene_model, na_for_model
using ..Drag:           build_drag_assessor, drag_draining_ratio, drag_ratio_zimm
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

Viscocapillary polymer prefactor `U_1 = λ₁ kT /(η_s R_eo³)` (TnW paper v4, Eq.
for `U_1`). Equals `Γ_eq · U_z`, where `U_z = U_∞·(α_c/α_∞)` (`U_∞ = 0.325`),
which is identically `λ₁ kT/(η_s R_eo³)` given the ηₛ formula used in this
context. `U_1 → 0.325` as `N_K → ∞`. The old group `K̃ = φ·U_1` is now derived
on demand inside `eval_flow`.
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
affine deformation of an isotropic equilibrium coil. It reproduces MATLAB's
reference routine `usyext_initial_M_diag.m` to machine precision (verified by Ko,
2026-06-11, acceptance test C), so the `micro_affine_strain` convention matches
the MATLAB pipeline.
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
    load_usyext_context(toml_path; De=nothing, phi=nothing)
        -> (UsyextCtx, flow, IntegrationPlan, TolOpts)

Parse a USYEXT TOML file and assemble all runtime contexts.

Optional keyword overrides:
  De  : override flow.De (useful for sweeps)
  phi : override concentration.phi
"""
function load_usyext_context(toml_path :: AbstractString;
                              De        :: Union{Float64, Nothing} = nothing,
                              phi_kw    :: Union{Float64, Nothing} = nothing)

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

    # FENE pole: STYEXT/USYEXT convention (differs from VELB for Nm>1)
    Qinf_sq = fe_model === :fenepme ? Q0_sq * NK : Q0_sq * NK / Nm

    c_star = E0^(-3)
    ckBT   = phi * c_star * kBT

    # ── Drag LUT + spectrum ───────────────────────────────────────────────────
    drag_lut = build_drag_assessor(phi, hK_star, NK, drag_model, draining_model)
    spectrum  = build_spectrum_assessor(
        spectrum_model = spectrum_model,
        Nm             = Nm,
        NK             = NK,
        hK_star        = hK_star,
        phi            = phi,
        drag_model     = drag_model,
        draining_model = draining_model,
    )

    # ── Equilibrium drag + ηₛ ────────────────────────────────────────────────
    # E=1 → equilibrium coil (matches MATLAB: r = √Y/E₀ = 1 at equilibrium)
    Gam_eq, _ = drag_ratio_zimm(1.0, phi, hK_star, NK, drag_model, draining_model)
    Gam_eq > 0 || error("Equilibrium drag ratio must be > 0")
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
        flow_ctx = CaberFlow(De_val, alphas, phi, U_1)
    elseif flow_kind == "caber_inertial"
        # De_val is interpreted as De_R (the Rayleigh-time Deborah number).
        # X_R = inertio-capillary Newtonian end-plate factor (default 0.5115 ⇔
        # Gaillard A≈0.47, 2X_R-1=2A³/9).
        X_R = _getf(flow_sec, "X_R", 0.5115)
        X_e = _getf(flow_sec, "Xe",  1.5)
        U_1 = _compute_U1(Gam_eq, hK_star, NK, draining_model)

        # Oh is the PRIMARY inertial input (2026-06-12 pivot): U_R is not an
        # independent constant — U_R = Oh·U_1/De_R (the relaxation time is
        # carried by the solvent), formed per-evaluation inside eval_flow.
        # Oh = 0 is the degenerate no-arrest (Keller–Miksis) limit and is
        # allowed.  flow.U_R remains accepted as an explicit back-compat
        # override, converted to the equivalent Oh = U_R·De_R/U_1 for THIS
        # De_R only (a constant-U_R sweep is an unphysical fluid — prefer Oh).
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
        flow_ctx = CaberInertialFlow(De_val, alphas, phi, Oh_val, U_1; XR=X_R, Xe=X_e)
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
