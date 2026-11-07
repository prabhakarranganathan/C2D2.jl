"""
    Spectrum.jl

Relaxation spectrum kernel for the multimode C2D2 model (technical report
Sec. 2.2 "The relaxation spectrum").

Computes the relative spectrum λ̂_p = λ_p/λ₁ for the modes p = 1…Nm, and the
stretch-dependent dilation Λ = λ₁/λ₁,₀ = ζ/ζ₀ of every relaxation time, where
λ₁ is the longest relaxation time of the stretched chain and λ₁,₀ its
equilibrium value at the given overlap ratio.

Spectrum models (TOML `[model] spectrum_model`):
  :rouse_scaling   λ̂_p = p^(−2)                        (Rouse scaling)
  :zimm_scaling    λ̂_p = p^(−3/2)                      (Zimm scaling)
  :rouse_full      exact Rouse eigenvalues of an Nm-spring chain
  :zimm_thurston   Rouse eigenvalues with Thurston's partial-draining correction
  :blob            blob spectrum of the C2D2 model: Rouse-like for the slow modes
                   (p ≤ number of blobs), Zimm-like inside a blob (report Sec. 2.2.1);
                   needs the blob count and the factor G from the drag kernel
  "auto" in the TOML selects :rouse_full for constant drag and :blob for C2D2 drag.

Public API
----------
  relative_spectrum(model, Nm, NK, hK_star; n_blob, G)  -> lambda_hat (Nm-vec)
  drag_dilation(zeta_ratio, zeta0_ratio)                 -> Lambda
  build_spectrum_assessor(; ...)                         -> SpectrumAssessor
  eval_spectrum(assessor, E, zeta_ratio)                 -> (lambda_hat, Lambda)
"""
module Spectrum

using ..Drag: drag_draining_ratio, drag_ratio_zimm, DEFAULT_P_PHI, default_p_phi, DEFAULT_G0_1D, DEFAULT_S_FORM, DEFAULT_Q_OES

export relative_spectrum, drag_dilation
export SpectrumAssessor, build_spectrum_assessor, eval_spectrum

# ── Relative spectrum ─────────────────────────────────────────────────────────

"""
    relative_spectrum(spectrum_model, Nm, NK, hK_star; n_blob=1.0, G=1.0)
                     -> Vector{Float64}

Return the relative spectrum λ̂_p = λ_p/λ₁ as an Nm-vector (λ̂₁ = 1 by construction);
report Sec. 2.2.1.

Arguments
---------
spectrum_model : Symbol — one of the models listed in the module docstring
Nm             : number of modes (springs) in the chain
NK             : number of Kuhn segments in the chain
hK_star        : hydrodynamic-interaction parameter of a Kuhn segment
n_blob         : number of blobs the chain is divided into (only used for :blob)
G              : Batchelor-type factor from the drag kernel (only used for :blob)

Example: `relative_spectrum(:zimm_scaling, 3, 3000.0, 0.05)` returns [1, 2^(−3/2), 3^(−3/2)].
"""
function relative_spectrum(spectrum_model::Symbol, Nm::Int, NK::Float64, hK_star::Float64;
                            n_blob::Float64=1.0, G::Float64=1.0)::Vector{Float64}
    lhat = if spectrum_model === :rouse_scaling
        _rouse_scaling(Nm)
    elseif spectrum_model === :zimm_scaling
        _zimm_scaling(Nm)
    elseif spectrum_model === :rouse_full
        _rouse_full(Nm)
    elseif spectrum_model === :zimm_thurston
        _zimm_thurston(Nm, hK_star)
    elseif spectrum_model === :blob
        _blob_spectrum(Nm, NK, hK_star, n_blob, G)
    else
        error("Unknown spectrum_model :$(spectrum_model). " *
              "Choose :rouse_scaling, :zimm_scaling, :rouse_full, :zimm_thurston, or :blob.")
    end

    # Normalise so that λ̂_1 = 1
    lhat ./= lhat[1]
    return lhat
end

# ── Relaxation-time dilation Λ = λ₁/λ₁,₀ = ζ/ζ₀ ─────────────────────────────

"""
    drag_dilation(zeta_ratio, zeta0_ratio) -> Lambda

Λ(E, φ) = ζ(E,φ)/ζ₀(φ) = λ₁/λ₁,₀: every relaxation time of the stretched chain
is the equilibrium one times Λ (report Sec. 2.2).

Both inputs are already normalised by ζ_Z (the dilute Zimm drag), so
  Λ = zeta_ratio / zeta0_ratio.
"""
function drag_dilation(zeta_ratio::Float64, zeta0_ratio::Float64)::Float64
    zeta0_ratio > 0 || error("zeta0_ratio must be > 0")
    return zeta_ratio / zeta0_ratio
end

# ── SpectrumAssessor: LUT over stretch E ─────────────────────────────────────

"""
    SpectrumAssessor

Stores the equilibrium drag ratio ζ₀/ζ_Z and provides a fast path to
evaluate (λ̂_p, Λ) given the current stretch E.

For the :blob spectrum, λ̂_p changes with E (because the blob count and G depend
on E).  For all other spectrum models λ̂_p is fixed (computed once at construction).
"""
struct SpectrumAssessor
    spectrum_model :: Symbol
    Nm             :: Int
    NK             :: Float64
    hK_star        :: Float64
    zeta0_ratio    :: Float64       # ζ₀/ζ_Z  (equilibrium; used for Λ)
    # Fixed spectra (non-blob): precomputed
    lambda_hat_eq  :: Vector{Float64}   # empty if :blob
    # For blob spectrum: drag params needed at runtime
    phi            :: Float64
    drag_model     :: Symbol
    draining_model :: Symbol
    drag_interp    :: Symbol        # drag-interpolation scheme (c2d2 only; see Drag.jl)
    p_phi          :: Float64       # CU / CUI exponent (unused otherwise)
    g0_1D          :: Float64       # U_1D factor floor (CU only; 1.0 = off)
    s_form         :: Symbol        # s(N_c, N_t) form (CU only)
    q_oes          :: Float64       # stretch-delayed OES dressing exponent (OESI/OESCUI only)
end

"""
    build_spectrum_assessor(; spectrum_model, Nm, NK, hK_star,
                              phi, drag_model, draining_model,
                              drag_interp=:rp, p_phi=default_p_phi(drag_interp),
                              g0_1D=DEFAULT_G0_1D, s_form=DEFAULT_S_FORM, q_oes=DEFAULT_Q_OES) -> SpectrumAssessor

Construct a SpectrumAssessor. Computes the equilibrium drag ratio ζ₀/ζ_Z
(at E = 1, the coiled state) once at construction time, with the same
drag-interpolation scheme as the chain drag so that Λ = 1 at rest.  The keyword
arguments are those of `Drag.drag_ratio_zimm`.  (With the archived CU scheme's
factor g on, `zeta0_ratio` = g(s₀)·ζ₀/ζ_Z = λ₁,₀/λ_Z.)
"""
function build_spectrum_assessor(; spectrum_model::Symbol,
                                   Nm::Int,
                                   NK::Float64,
                                   hK_star::Float64,
                                   phi::Float64,
                                   drag_model::Symbol,
                                   draining_model::Symbol,
                                   drag_interp::Symbol = :rp,
                                   p_phi::Float64 = default_p_phi(drag_interp),
                                   g0_1D::Float64 = DEFAULT_G0_1D,
                                   s_form::Symbol = DEFAULT_S_FORM,
                                   q_oes::Float64 = DEFAULT_Q_OES)::SpectrumAssessor
    # Equilibrium drag (E = 1 = coiled state)
    zeta0_ratio, _ = drag_ratio_zimm(1.0, phi, hK_star, NK, drag_model, draining_model;
                                     drag_interp=drag_interp, p_phi=p_phi,
                                     g0_1D=g0_1D, s_form=s_form, q_oes=q_oes)

    if spectrum_model !== :blob
        lhat_eq = relative_spectrum(spectrum_model, Nm, NK, hK_star)
    else
        lhat_eq = Float64[]
    end

    return SpectrumAssessor(spectrum_model, Nm, NK, hK_star,
                            zeta0_ratio, lhat_eq,
                            phi, drag_model, draining_model,
                            drag_interp, p_phi, g0_1D, s_form, q_oes)
end

"""
    eval_spectrum(assessor, E, zeta_ratio) -> (lambda_hat, Lambda)

Given the current stretch E and the current drag ratio ζ/ζ_Z (from a DragAssessor),
return:
  lambda_hat : Nm-vector of relative relaxation times λ̂_p = λ_p/λ₁
  Lambda     : scalar dilation Λ = ζ/ζ₀ = λ₁/λ₁,₀
"""
function eval_spectrum(a::SpectrumAssessor, E::Float64,
                       zeta_ratio::Float64)::Tuple{Vector{Float64}, Float64}
    Lambda = drag_dilation(zeta_ratio, a.zeta0_ratio)

    if a.spectrum_model !== :blob
        return copy(a.lambda_hat_eq), Lambda
    else
        # For blob spectrum, n_blob and G depend on E — recompute each call
        _, diag = drag_ratio_zimm(E, a.phi, a.hK_star, a.NK,
                                   a.drag_model, a.draining_model;
                                   drag_interp=a.drag_interp, p_phi=a.p_phi,
                                   g0_1D=a.g0_1D, s_form=a.s_form, q_oes=a.q_oes)
        n_blob = diag.n_blob
        G      = diag.G
        lhat   = relative_spectrum(:blob, a.Nm, a.NK, a.hK_star; n_blob=n_blob, G=G)
        return lhat, Lambda
    end
end

# ── Spectrum implementations ──────────────────────────────────────────────────

function _rouse_scaling(Nm::Int)::Vector{Float64}
    p = 1:Nm
    return p .^ (-2.0)
end

function _zimm_scaling(Nm::Int)::Vector{Float64}
    p = 1:Nm
    return p .^ (-1.5)
end

function _rouse_full(Nm::Int)::Vector{Float64}
    # Analytical Rouse eigenvalues: μ_p = 4 sin²(pπ / 2(Nm+1))
    p     = 1:Nm
    theta = (p .* π) ./ (2 * (Nm + 1))
    ev    = 4 .* sin.(theta).^2
    ev    = sort(ev)
    return ev[1] ./ ev   # λ̂_p = μ_1 / μ_p
end

function _zimm_thurston(Nm::Int, hK_star::Float64)::Vector{Float64}
    # Thurston approximation: λ̂_p^Zimm ≈ λ̂_p^Rouse * correction(h*, p)
    a = 1.66; b = 0.78; c = 1.4
    p    = 1:Nm
    evR  = _rouse_full(Nm)
    # evR is already λ̂ = μ_1/μ_p; invert to get eigenvalues
    ev_rouse = evR[1] ./ evR
    ev_zimm  = ev_rouse .* (1 - a*hK_star^b) .* ((p ./ Nm) .^ (-c * hK_star^b))
    ev_zimm  = max.(ev_zimm, eps(Float64))
    ev_zimm  = sort(ev_zimm)
    return ev_zimm[1] ./ ev_zimm
end

# Blob spectrum (report Sec. 2.2.1): for p ≤ N_b (number of blobs) the modes are
# Rouse-like, λ̂_p = p^(−2 + ln G/ln N_b); for p > N_b they are Zimm-like inside a
# blob, λ̂_p = p^(−3/2)·[α(h*_K, N_K/p)/α(h*_K, N_K/N_b)]·G/√N_b.  The two branches
# meet at p = N_b.  The caller normalises so that λ̂₁ = 1.
function _blob_spectrum(Nm::Int, NK::Float64, hK_star::Float64,
                         n_blob::Float64, G::Float64)::Vector{Float64}
    p      = Float64.(1:Nm)
    Nb_r   = max(n_blob, 1.0)
    Nb_cut = max(1, floor(Int, Nb_r))
    lnNb   = log(Nb_r)
    lnG    = log(max(G, eps(Float64)))
    tiny   = 1e-12

    # Slow branch exponent (Rouse-like, possibly tilted by G)
    expo_R = if lnNb < tiny || Nb_r <= 1.0 + tiny
        -2.0
    else
        -2.0 + lnG / lnNb
    end

    lhat = zeros(Nm)

    # Slow branch: p ≤ Nb_cut
    for ip in 1:Nm
        if p[ip] <= Nb_cut
            lhat[ip] = p[ip]^expo_R
        else
            # Fast branch: Zimm-like within a blob
            alpha_den = drag_draining_ratio(hK_star, NK / Nb_r)
            alpha_num = drag_draining_ratio(hK_star, NK / p[ip])
            alpha_ratio = alpha_num / max(alpha_den, tiny)
            lhat[ip] = p[ip]^(-1.5) * alpha_ratio * (G / sqrt(Nb_r))
        end
    end

    return lhat
end

end # module Spectrum
