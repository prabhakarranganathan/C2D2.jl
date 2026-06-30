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
                  drag_model, draining_model)   -> (zeta_ratio, diagnostics)
  build_drag_assessor(params)                   -> DragAssessor
  eval_drag(assessor, E)                        -> zeta_ratio
  eval_drag_deriv(assessor, E)                  -> d(zeta_ratio)/dE
"""
module Drag

# PCHIP interpolation is implemented manually below (no external dependency).

export drag_draining_ratio, drag_ratio_zimm, DragAssessor,
       build_drag_assessor, eval_drag, eval_drag_deriv

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
    drag_ratio_zimm(E, phi, hK_star, NK, drag_model, draining_model)
                   -> (zeta_ratio, diag)

Compute ζ/ζ_Z for a single stretch value E = |R|/R₀ ≥ 1.

drag_model      : :constant | :c1d2 | :c2d2
draining_model  : :nondraining | :partialdraining

diag is a NamedTuple with fields: G, blob_drag_ratio, n_blob, xi_blob, blob_type.
"""
function drag_ratio_zimm(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                         drag_model::Symbol, draining_model::Symbol)

    E_min = 1.0
    E_max = sqrt(NK)
    E = clamp(E, E_min, E_max)
    phi = max(phi, 0.0)

    alphaZ = _alpha_zimm(hK_star, NK, draining_model)

    if drag_model === :constant
        return 1.0, (G=1.0, blob_drag_ratio=NaN, n_blob=1.0, xi_blob=NaN, blob_type=:NA)

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
        return zeta_ratio, (G=G, blob_drag_ratio=zb_rel, n_blob=nT, xi_blob=xiT, blob_type=:T)

    elseif drag_model === :c2d2
        return _drag_c2d2(E, phi, hK_star, NK, alphaZ, draining_model)

    else
        error("Unknown drag_model :$(drag_model). Use :constant, :c1d2, or :c2d2.")
    end
end

function _drag_c2d2(E::Float64, phi::Float64, hK_star::Float64, NK::Float64,
                    alphaZ::Float64, draining_model::Symbol)
    E_max = sqrt(NK)

    # ---- correlation-blob baseline at equilibrium ----
    if phi > E_max
        xiC    = 1.0 / E_max
        alphaC = _alpha_blob(hK_star, NK, NK, draining_model)
        zbC    = (alphaC * xiC) / alphaZ
        nC     = NK
    elseif phi > 1.0
        nC     = phi^2
        xiC    = 1.0 / phi
        alphaC = _alpha_blob(hK_star, NK, nC, draining_model)
        zbC    = (alphaC * xiC) / alphaZ
    else
        nC  = 1.0
        xiC = 1.0
        zbC = 1.0
    end

    xiT = 1.0 / E
    nT  = clamp(E^2, 1.0, NK)
    A   = E / xiT    # = E²

    if E > phi
        # ---- tension-core region ----
        alphaT = _alpha_blob(hK_star, NK, nT, draining_model)
        zbT    = (alphaT * xiT) / alphaZ

        phiT = max(phi * E * xiT^2, eps(Float64))
        B    = 1.0 + sqrt(max(log(1.0 / phiT), 0.0) / phiT)

        Y    = A - 1.0
        J    = 1.0 + (0.5*Y + (2/ℯ)*Y^2) / max(A, eps(Float64))

        G1  = 1.0 + log(max((A + J*(B-1)) / max(A + B, eps(Float64)), eps(Float64)))
        G2  = 1.0 + log(max(B / max(B + 1.0/max(A, eps(Float64)), eps(Float64)), eps(Float64)))
        G   = max(G1 / max(G2, eps(Float64)), eps(Float64))

        zeta_ratio = max(zbT * nT / G, eps(Float64))
        return zeta_ratio, (G=G, blob_drag_ratio=zbT, n_blob=nT, xi_blob=xiT, blob_type=:T)

    else
        # ---- correlation-core region ----
        zeta_ratio = max(zbC * nC / 1.0, eps(Float64))
        return zeta_ratio, (G=1.0, blob_drag_ratio=zbC, n_blob=nC, xi_blob=xiC, blob_type=:C)
    end
end

# ── LUT + assessor for fast evaluation ───────────────────────────────────────

"""
    DragAssessor

Precomputed piecewise-cubic (PCHIP-style) interpolant of ζ/ζ_Z(E) on a
log-spaced grid. Built once per run; queried at every spatial node and time step.

Handles the kink at E = φ (regime boundary) by fitting separate cubics on
either side.
"""
struct DragAssessor
    drag_model :: Symbol             # :constant | :c1d2 | :c2d2
    E_min    :: Float64
    E_max    :: Float64
    kink_E   :: Float64          # NaN if no kink
    # Left piece: E ∈ [E_min, kink_E]  (or full range if no kink)
    xs_L     :: Vector{Float64}  # log10 grid
    ys_L     :: Vector{Float64}
    # Right piece: E ∈ [kink_E, E_max] (empty if no kink)
    xs_R     :: Vector{Float64}
    ys_R     :: Vector{Float64}
end

"""
    build_drag_assessor(phi, hK_star, NK, drag_model, draining_model;
                        n_points=400) -> DragAssessor

Build a DragAssessor by tabulating drag_ratio_zimm on a log-spaced E grid
and fitting piecewise cubics around the kink at E = φ (if drag_model = :c2d2).
"""
function build_drag_assessor(phi::Float64, hK_star::Float64, NK::Float64,
                              drag_model::Symbol, draining_model::Symbol;
                              n_points::Int=400)::DragAssessor
    E_min = 1.0
    E_max = sqrt(NK)
    # log-spaced grid
    E_grid = exp.(range(log(E_min), log(E_max); length=n_points))
    y_grid = [drag_ratio_zimm(e, phi, hK_star, NK, drag_model, draining_model)[1]
              for e in E_grid]

    # Locate kink: only c2d2 with phi > 1 has a kink at E = phi
    has_kink = (drag_model === :c2d2) && (phi > 1.0) && (phi < E_max)
    kink_E   = has_kink ? phi : NaN

    if has_kink
        kink_idx = searchsortedfirst(E_grid, kink_E)
        kink_idx = clamp(kink_idx, 2, length(E_grid)-1)
        L = 1:kink_idx
        R = kink_idx:length(E_grid)
        return DragAssessor(drag_model, E_min, E_max, kink_E,
                            log10.(E_grid[L]), y_grid[L],
                            log10.(E_grid[R]), y_grid[R])
    else
        return DragAssessor(drag_model, E_min, E_max, NaN,
                            log10.(E_grid), y_grid,
                            Float64[], Float64[])
    end
end

"""
    eval_drag(assessor, E) -> zeta_ratio

Evaluate ζ/ζ_Z at stretch E using the prebuilt piecewise cubic interpolant.
E is clamped to [E_min, E_max] — no out-of-domain error.
"""
function eval_drag(a::DragAssessor, E::Float64)::Float64
    E = clamp(E, a.E_min, a.E_max)
    xq = log10(E)
    if isnan(a.kink_E) || E <= a.kink_E
        return _pchip_eval(a.xs_L, a.ys_L, xq)
    else
        return _pchip_eval(a.xs_R, a.ys_R, xq)
    end
end

"""
    eval_drag_deriv(assessor, E) -> d(zeta_ratio)/dE

First derivative of ζ/ζ_Z with respect to E (chain rule through log10).
"""
function eval_drag_deriv(a::DragAssessor, E::Float64)::Float64
    E = clamp(E, a.E_min, a.E_max)
    xq = log10(E)
    if isnan(a.kink_E) || E <= a.kink_E
        dydx = _pchip_deriv(a.xs_L, a.ys_L, xq)
    else
        dydx = _pchip_deriv(a.xs_R, a.ys_R, xq)
    end
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
