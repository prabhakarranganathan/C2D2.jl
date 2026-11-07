"""
    Fene.jl

Finite-extensibility (FENE, finitely extensible nonlinear elastic) spring-law
kernel for the multimode C2D2 model: the mode-effective Peterlin factors f_p that
multiply the spring force of each mode (technical report Sec. 2.3 and Appendix A).

Four spring laws are supported, selected by the TOML key `[model] fe_model`:
  - `"hookean"`  Hookean: f = 1 for every mode (no finite extensibility; Na = 0;
                 `"oldroyd_b"` is a legacy alias)
  - `"fenepme"`  FENE-PME: one Peterlin factor for the whole chain, from the
                 chain's mean-square end-to-end distance E² (Na = 0)
  - `"fenepm"`   FENE-PM: one Peterlin factor from the mean modal trace (Na = 1)
  - `"fenep"`    FENE-P: a factor per mode, through the Rouse weight matrix
                 (Na = Nm; the most faithful and the most expensive, O(Nm²))
Na is the number of auxiliary closure variables X the steady solver carries.

The spring law and the drag model are independent choices.  A Hookean spring
law combined with conformation-dependent (C2D2) drag is a valid model, but it is
not the classical Oldroyd-B model, which assumes constant drag.

Public API
----------
  fene_peterlin(chi, Q0_sq, Qinf_sq)       -> (fval, df_dchi)
  eval_fene(model, Xa, Y; want_derivs)     -> FeneResult
  na_for_model(fe_model, Nm)               -> Int
  build_fene_model(fe_model, Nm, Q0_sq, Qinf_sq) -> model struct

All functions are pure (no mutation, no globals).
"""
module Fene

using LinearAlgebra: mul!

export AbstractFeneModel, FenePME, FenePM, FeneP, Hookean
export FeneResult, eval_fene, na_for_model, fene_peterlin
export build_fene_model, rouse_weight_matrix, orthogonal_matrix_rouse

# ── Model type hierarchy ───────────────────────────────────────────────────────

"""
Abstract supertype for all FENE model variants.
Dispatch on this to select PME / PM / P behaviour without runtime if-chains.
"""
abstract type AbstractFeneModel end

"""
FENE-PME: single scalar Peterlin factor based on chain end-to-end distance E².
Na = 0 (no X auxiliary variables). O(Nm) cost.
"""
struct FenePME <: AbstractFeneModel
    Nm     :: Int
    Q0_sq  :: Float64   # dimensionless equilibrium spring length squared
    Qinf_sq:: Float64   # dimensionless contour length squared (pole)
end

"""
FENE-PM: single scalar Peterlin factor based on mean modal trace X̄ = (1/Nm) Σ tr Mₚ.
Na = 1. O(Nm) cost.
"""
struct FenePM <: AbstractFeneModel
    Nm     :: Int
    Q0_sq  :: Float64
    Qinf_sq:: Float64
end

"""
Hookean spring law: `f = 1` identically.
No finite extensibility, no FE auxiliary variables, no FE pole.
Na = 0. O(Nm) cost, zero-allocation hot path.

A finite `Nk` is permitted for diagnostic use (e.g. as a stretch warning threshold)
but does not make the spring law finitely extensible when `f = 1`.

Use `:hookean` in TOML inputs; `:oldroyd_b` is a legacy alias for the same branch.
Note: combining this spring law with configuration-dependent drag (C2D2 drag) gives a
valid model, but the result is not classical Oldroyd-B (which assumes constant drag).
"""
struct Hookean <: AbstractFeneModel
    Nm :: Int
end

"""
FENE-P: mode-effective factors f_p built from the spring-wise Peterlin factors through
the Rouse weight matrix W = Π⊙Π (report Appendix A, which motivates this closure from
the TFNA, "two-fluid normal-mode analysis", approach).
Na = Nm. O(Nm²) cost (dense Jacobian when requested).
The weight matrix W is precomputed at construction time.
"""
struct FeneP <: AbstractFeneModel
    Nm     :: Int
    Q0_sq  :: Float64
    Qinf_sq:: Float64
    W      :: Matrix{Float64}   # Nm×Nm,  W[i,p] = Π[i,p]²
end

# ── Result type ────────────────────────────────────────────────────────────────

"""
Result of a FENE evaluation at a single spatial point / closure state.

Fields
------
fp       : Nm-vector of mode-effective Peterlin factors fₚ
df_dX    : Nm×Na Jacobian ∂fₚ/∂Xⱼ  (empty matrix if not requested)
df_dY    : Nm-vector ∂fₚ/∂Y          (zeros vector if not requested)
chi      : spring-wise stretch arguments (diagnostic)
"""
struct FeneResult
    fp    :: Vector{Float64}
    df_dX :: Matrix{Float64}
    df_dY :: Vector{Float64}
end

# ── Helper: number of auxiliary X variables ───────────────────────────────────

"""
    na_for_model(fe_model::Symbol, Nm::Int) -> Int

Return the number of X-type auxiliary closure variables for a given model.
  :fenepme → 0
  :fenepm  → 1
  :fenep   → Nm
"""
function na_for_model(fe_model::Symbol, Nm::Int)::Int
    fe_model === :hookean   && return 0   # primary name
    fe_model === :oldroyd_b && return 0   # legacy alias → Hookean
    fe_model === :fenepme   && return 0
    fe_model === :fenepm    && return 1
    fe_model === :fenep     && return Nm
    error("Unknown fe_model :$(fe_model). Choose :hookean, :fenepme, :fenepm, or :fenep.  (:oldroyd_b is a legacy alias for :hookean)")
end

# ── Constructor helper ─────────────────────────────────────────────────────────

"""
    build_fene_model(fe_model, Nm, Q0_sq, Qinf_sq) -> AbstractFeneModel

Construct the appropriate concrete FENE model struct.
For FeneP, this precomputes the Rouse weight matrix W.
"""
function build_fene_model(fe_model::Symbol, Nm::Int,
                          Q0_sq::Float64, Qinf_sq::Float64)::AbstractFeneModel
    if fe_model === :hookean || fe_model === :oldroyd_b  # :oldroyd_b is a legacy alias
        return Hookean(Nm)
    elseif fe_model === :fenepme
        return FenePME(Nm, Q0_sq, Qinf_sq)
    elseif fe_model === :fenepm
        return FenePM(Nm, Q0_sq, Qinf_sq)
    elseif fe_model === :fenep
        W = _rouse_weight_matrix(Nm)
        return FeneP(Nm, Q0_sq, Qinf_sq, W)
    else
        error("Unknown fe_model :$(fe_model). Choose :hookean, :fenepme, :fenepm, or :fenep.  (:oldroyd_b is a legacy alias for :hookean)")
    end
end

# ── Scalar Peterlin primitive ──────────────────────────────────────────────────

"""
    fene_peterlin(chi, Q0_sq, Qinf_sq) -> (fval, df_dchi)

Scalar Peterlin spring-force factor and its derivative (report Sec. 2.3):

  f(χ) = (Q∞² − Q₀²) / (Q∞² − χ)
  df/dχ = (Q∞² − Q₀²) / (Q∞² − χ)²

χ is the mean-square spring (or chain) extension, Q₀² its equilibrium value and
Q∞² the fully extended value (the pole), all in the same units.  Throws if any
chi ≥ Qinf_sq (the spring would be beyond full extension).  Works element-wise
on arrays.
"""
function fene_peterlin(chi::Real, Q0_sq::Float64, Qinf_sq::Float64)
    chi >= Qinf_sq && throw(ArgumentError(
        "FENE pole hit: chi=$(chi) >= Qinf_sq=$(Qinf_sq). Tighten solver or reduce step."))
    fnum = Qinf_sq - Q0_sq
    fden = Qinf_sq - chi
    fval    = fnum / fden
    df_dchi = fnum / fden^2
    return fval, df_dchi
end

# Array-valued overload (used by FeneP)
function fene_peterlin(chi::AbstractVector, Q0_sq::Float64, Qinf_sq::Float64)
    any(>=(Qinf_sq), chi) && throw(ArgumentError(
        "FENE pole hit: max(chi)=$(maximum(chi)) >= Qinf_sq=$(Qinf_sq)."))
    fnum  = Qinf_sq - Q0_sq
    fden  = Qinf_sq .- chi
    fval    = fnum ./ fden
    df_dchi = fnum ./ fden.^2
    return fval, df_dchi
end

# ── eval_fene: main dispatch ───────────────────────────────────────────────────

"""
    eval_fene(model, Xa, Y; want_derivs=false) -> FeneResult

Evaluate mode-effective FENE factors and (optionally) their derivatives.

Arguments
---------
model        : one of FenePME, FenePM, FeneP
Xa           : Na-vector of X auxiliary closure variables
                 (length 0 for PME, 1 for PM, Nm for P)
Y            : scalar E² (mean-squared end-to-end distance)
want_derivs  : if true, populate df_dX and df_dY in the result

Returns
-------
FeneResult with fp (Nm-vector), df_dX (Nm×Na), df_dY (Nm-vector).
Empty arrays if want_derivs=false.
"""
# Hookean: f_p = 1 for all modes. Na = 0, zero allocation, no arguments consumed.
function eval_fene(m::Hookean, Xa::AbstractVector, Y::Float64;
                   want_derivs::Bool=false)::FeneResult
    fp    = ones(m.Nm)
    df_dX = zeros(m.Nm, 0)
    df_dY = zeros(m.Nm)
    return FeneResult(fp, df_dX, df_dY)
end

function eval_fene(m::FenePME, Xa::AbstractVector, Y::Float64;
                   want_derivs::Bool=false)::FeneResult
    Nm = m.Nm
    chi = Y / Nm
    fscalar, df_dchi = fene_peterlin(chi, m.Q0_sq, m.Qinf_sq)
    fp = fill(fscalar, Nm)

    if want_derivs
        df_dX = zeros(Nm, 0)                           # Na=0
        df_dY = fill(df_dchi / Nm, Nm)
    else
        df_dX = zeros(Nm, 0)
        df_dY = zeros(Nm)
    end
    return FeneResult(fp, df_dX, df_dY)
end

function eval_fene(m::FenePM, Xa::AbstractVector, Y::Float64;
                   want_derivs::Bool=false)::FeneResult
    Nm = m.Nm
    length(Xa) == 1 || error("FenePM expects Na=1, got $(length(Xa))")
    chi = Xa[1]
    fscalar, df_dchi = fene_peterlin(chi, m.Q0_sq, m.Qinf_sq)
    fp = fill(fscalar, Nm)

    if want_derivs
        df_dX = fill(df_dchi, Nm, 1)                  # Nm×1
        df_dY = zeros(Nm)
    else
        df_dX = zeros(Nm, 1)
        df_dY = zeros(Nm)
    end
    return FeneResult(fp, df_dX, df_dY)
end

function eval_fene(m::FeneP, Xa::AbstractVector, Y::Float64;
                   want_derivs::Bool=false)::FeneResult
    Nm = m.Nm
    length(Xa) == Nm || error("FeneP expects Na=Nm=$(Nm), got $(length(Xa))")

    # chi_i = Σ_q W[i,q] * X_q   (spring-wise stretch)
    chi     = m.W * Xa                                 # Nm-vector
    f_spring, df_dchi = fene_peterlin(chi, m.Q0_sq, m.Qinf_sq)

    # fp = W' * f_spring
    fp = m.W' * f_spring                               # Nm-vector

    if want_derivs
        # Dense Jacobian: df_eff/dX = W' * diag(df/dchi) * W
        dW = m.W .* df_dchi                            # row-wise scale: Nm×Nm
        df_dX = m.W' * dW                              # Nm×Nm
        df_dY = zeros(Nm)
    else
        df_dX = zeros(Nm, Nm)
        df_dY = zeros(Nm)
    end
    return FeneResult(fp, df_dX, df_dY)
end

# ── Internal: Rouse orthogonal matrix and weight matrix ───────────────────────

"""
    orthogonal_matrix_rouse(Nm) -> Matrix{Float64}

Analytic Rouse normal-mode matrix Π (Nm×Nm).
Π[i,p] = sqrt(2/(Nm+1)) * sin(i*p*π/(Nm+1))
Columns are orthonormal; rows index springs, columns index modes.
"""
function orthogonal_matrix_rouse(Nm::Int)::Matrix{Float64}
    Pi = Matrix{Float64}(undef, Nm, Nm)
    c  = sqrt(2.0 / (Nm + 1))
    for p in 1:Nm, i in 1:Nm
        Pi[i, p] = c * sin(i * p * π / (Nm + 1))
    end
    return Pi
end
# Keep private alias for internal use
const _orthogonal_matrix_rouse = orthogonal_matrix_rouse

"""
    rouse_weight_matrix(Nm) -> Matrix{Float64}

W[i,p] = Π[i,p]²   (element-wise square of the Rouse matrix)
"""
function rouse_weight_matrix(Nm::Int)::Matrix{Float64}
    Pi = orthogonal_matrix_rouse(Nm)
    return Pi .^ 2
end
const _rouse_weight_matrix = rouse_weight_matrix

end # module Fene
