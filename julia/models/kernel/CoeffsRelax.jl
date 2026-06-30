"""
    CoeffsRelax.jl

Compute θ_p and σ_p — the modal evolution coefficients for the C2D2 model.

    θ_p(Z) = f_p(Z) / (Λ(Z) λ̂_p(Z))
    σ_p(Z) = 1      / (3 Λ(Z) λ̂_p(Z))

Performance design
------------------
eval_coeffs! is called N_cells times per RHS evaluation, which is called
O(stages * time_steps) times — potentially millions of calls per run.

All intermediate vectors (fp, lambda_hat, trM, inv_Lam_lhat) are pre-allocated
inside CoeffsKernel as scratch buffers and reused every call.
The public-facing CoeffsResult is a lightweight struct of views into those buffers.
Zero heap allocations occur on the hot path.

Public API
----------
  ClosureZ                             — auxiliary closure variables at one node
  CoeffsKernel                         — assembled kernel with scratch buffers
  build_coeffs_kernel(fene, drag, spec) -> CoeffsKernel
  eval_coeffs!(kernel, Z)              — in-place, writes into kernel.scratch.*
  closure_from_M_diag!(Z, M_diag, ...) — in-place ClosureZ construction
"""
module CoeffsRelax

using LinearAlgebra: dot, mul!
using ..Fene:     AbstractFeneModel, FenePME, FenePM, FeneP, Hookean,
                  fene_peterlin, rouse_weight_matrix
using ..Drag:     DragAssessor, eval_drag
using ..Spectrum: SpectrumAssessor, eval_spectrum

export ClosureZ, CoeffsKernel, CoeffsScratch,
       build_coeffs_kernel, eval_coeffs!,
       closure_from_M_diag!, closure_from_M_diag,
       # legacy non-mutating interface (used in tests)
       CoeffsResult, eval_coeffs

# ── Auxiliary closure variables ───────────────────────────────────────────────

"""
    ClosureZ

Closure variables at one spatial node.  Constructed by closure_from_M_diag!.

Fields
------
Xa  : Na-vector  (length 0/1/Nm for PME/PM/P)
Y   : scalar     E² = Σ_p S_p tr M_p
E   : √Y
"""
mutable struct ClosureZ
    Xa :: Vector{Float64}
    Y  :: Float64
    E  :: Float64
end

ClosureZ(Na::Int) = ClosureZ(zeros(Na), 0.0, 0.0)
# Convenience constructor matching old API — used in tests and diagnostics
ClosureZ(Xa::Vector{Float64}, Y::Float64) = ClosureZ(Xa, Y, sqrt(max(Y, 0.0)))

# ── Scratch buffers (pre-allocated, reused every eval_coeffs! call) ───────────

"""
    CoeffsScratch

Pre-allocated workspace for eval_coeffs!.
One instance lives inside CoeffsKernel; never allocate another.
"""
mutable struct CoeffsScratch
    fp         :: Vector{Float64}   # FENE factors           [Nm]
    lambda_hat :: Vector{Float64}   # relative spectrum       [Nm]
    theta_p    :: Vector{Float64}   # evolution coeffs θ_p   [Nm]
    sigma_p    :: Vector{Float64}   # evolution coeffs σ_p   [Nm]
    trM        :: Vector{Float64}   # tr M_p scratch          [Nm]
    chi        :: Vector{Float64}   # spring-wise χ_i (FeneP) [Nm]
    Lambda     :: Float64           # relaxation-time dilation factor
    zeta_ratio :: Float64           # ζ/ζ_Z (diagnostic)
end

CoeffsScratch(Nm::Int) = CoeffsScratch(
    zeros(Nm), zeros(Nm), zeros(Nm), zeros(Nm), zeros(Nm), zeros(Nm),
    1.0, 1.0)

# ── CoeffsKernel ─────────────────────────────────────────────────────────────

"""
    CoeffsKernel

Assembled kernel: FENE model + drag LUT + spectrum assessor + scratch buffers.
Build once with build_coeffs_kernel; call eval_coeffs! at every node every step.
"""
struct CoeffsKernel
    fene     :: AbstractFeneModel
    drag_lut :: DragAssessor
    spectrum :: SpectrumAssessor
    Nm       :: Int
    Na       :: Int
    scratch  :: CoeffsScratch
    # Cached reusable ClosureZ (avoids constructing a new one each node)
    Z_buf    :: ClosureZ
end

"""
    build_coeffs_kernel(fene, drag_lut, spectrum) -> CoeffsKernel
"""
function build_coeffs_kernel(fene::AbstractFeneModel,
                              drag_lut::DragAssessor,
                              spectrum::SpectrumAssessor)::CoeffsKernel
    Nm = fene.Nm
    Na = _na(fene)
    return CoeffsKernel(fene, drag_lut, spectrum, Nm, Na,
                        CoeffsScratch(Nm), ClosureZ(Na))
end

_na(::FenePME)   = 0
_na(::FenePM)    = 1
_na(m::FeneP)    = m.Nm
_na(::Hookean)   = 0   # Hookean: no FENE auxiliary variables

# ── In-place closure variable construction ───────────────────────────────────

"""
    closure_from_M_diag!(Z, M_diag, fe_model, Nm, Sp)

Fill ClosureZ Z in-place from the diagonal conformation tensor at one node.

M_diag  : Nm×2  view  columns = [M_zz, M_rr]
fe_model: Symbol
Nm      : number of modes
Sp      : Nm-vector of end-to-end weights
"""
function closure_from_M_diag!(Z::ClosureZ,
                               M_diag::AbstractMatrix{Float64},
                               fe_model::Symbol, Nm::Int,
                               Sp::Vector{Float64},
                               trM_buf::Vector{Float64})
    # tr M_p = M_zz_p + 2 M_rr_p  (uniaxial)
    @inbounds for p in 1:Nm
        trM_buf[p] = M_diag[p,1] + 2.0*M_diag[p,2]
    end

    # Y = E² = Σ_p S_p tr M_p
    Y = dot(Sp, trM_buf)
    Z.Y = Y
    Z.E = sqrt(max(Y, 0.0))

    if fe_model === :fenepme || fe_model === :hookean || fe_model === :oldroyd_b
        # Xa is empty — nothing to write
    elseif fe_model === :fenepm
        Z.Xa[1] = sum(trM_buf) / Nm
    elseif fe_model === :fenep
        @inbounds for p in 1:Nm
            Z.Xa[p] = trM_buf[p]
        end
    else
        error("Unknown fe_model: $fe_model")
    end
    nothing
end

# Allocating convenience version (used in tests and non-hot paths)
function closure_from_M_diag(M_diag::AbstractMatrix{Float64},
                              fe_model::Symbol, Nm::Int,
                              Sp::Vector{Float64})::ClosureZ
    Na    = (fe_model === :fenepme || fe_model === :hookean || fe_model === :oldroyd_b) ? 0 :
            fe_model === :fenepm  ? 1 : Nm
    Z     = ClosureZ(Na)
    trM   = zeros(Nm)
    closure_from_M_diag!(Z, M_diag, fe_model, Nm, Sp, trM)
    return Z
end

# ── In-place FENE evaluation (writes into scratch.fp) ────────────────────────

function _eval_fene_inplace!(fp::Vector{Float64}, chi::Vector{Float64},
                              fene::FenePME, Z::ClosureZ)
    c, df = fene_peterlin(Z.Y / fene.Nm, fene.Q0_sq, fene.Qinf_sq)
    fill!(fp, c)
end

function _eval_fene_inplace!(fp::Vector{Float64}, chi::Vector{Float64},
                              fene::FenePM, Z::ClosureZ)
    c, df = fene_peterlin(Z.Xa[1], fene.Q0_sq, fene.Qinf_sq)
    fill!(fp, c)
end

function _eval_fene_inplace!(fp::Vector{Float64}, chi::Vector{Float64},
                              fene::FeneP, Z::ClosureZ)
    Nm = fene.Nm
    # chi = W * Xa  (spring-wise stretch)
    mul!(chi, fene.W, Z.Xa)
    # f_spring applied element-wise
    for i in 1:Nm
        fs, _ = fene_peterlin(chi[i], fene.Q0_sq, fene.Qinf_sq)
        chi[i] = fs    # reuse chi buffer for f_spring
    end
    # fp = W' * f_spring
    mul!(fp, fene.W', chi)
end

# Hookean: f_p = 1 for all modes (linear springs, no pole).
function _eval_fene_inplace!(fp::Vector{Float64}, chi::Vector{Float64},
                              ::Hookean, ::ClosureZ)
    fill!(fp, 1.0)
end


# ── In-place spectrum evaluation (writes into scratch.lambda_hat) ─────────────

function _eval_spectrum_inplace!(lambda_hat::Vector{Float64},
                                  spec::SpectrumAssessor, E::Float64,
                                  zeta_ratio::Float64)::Float64
    if spec.spectrum_model !== :blob
        # Fixed spectrum — copy from the precomputed vector
        copyto!(lambda_hat, spec.lambda_hat_eq)
        return zeta_ratio / spec.zeta0_ratio   # Lambda
    else
        # Blob spectrum: recompute n_blob and G from drag diagnostics
        # eval_spectrum allocates here for :blob — acceptable since blob
        # is always used with c2d2 drag which already goes through the LUT
        lhat_new, Lambda = eval_spectrum(spec, E, zeta_ratio)
        copyto!(lambda_hat, lhat_new)
        return Lambda
    end
end

# ── Main in-place evaluation ─────────────────────────────────────────────────

"""
    eval_coeffs!(kernel, Z)

Compute θ_p and σ_p in-place. Results are in kernel.scratch.theta_p and .sigma_p.
Zero heap allocations for non-blob spectra with FenePME or FenePM.

After calling this, read results from kernel.scratch directly:
    theta_p = kernel.scratch.theta_p
    sigma_p = kernel.scratch.sigma_p
    fp      = kernel.scratch.fp
"""
function eval_coeffs!(kernel::CoeffsKernel, Z::ClosureZ)
    sc  = kernel.scratch
    Nm  = kernel.Nm

    # 1. FENE factors → sc.fp
    _eval_fene_inplace!(sc.fp, sc.chi, kernel.fene, Z)

    # 2. Drag ratio
    zeta_ratio = eval_drag(kernel.drag_lut, Z.E)

    # 3. Spectrum + dilation → sc.lambda_hat, Lambda
    Lambda = _eval_spectrum_inplace!(sc.lambda_hat, kernel.spectrum,
                                     Z.E, zeta_ratio)

    # 4. Assemble θ_p = f_p / (Λ λ̂_p),  σ_p = 1/(3 Λ λ̂_p)
    sc.Lambda     = Lambda
    sc.zeta_ratio = zeta_ratio
    @inbounds for p in 1:Nm
        inv_ll        = 1.0 / (Lambda * sc.lambda_hat[p])
        sc.theta_p[p] = sc.fp[p] * inv_ll
        sc.sigma_p[p] = inv_ll / 3.0
    end
    nothing
end

# ── Legacy allocating interface (used in tests, not in production RHS) ────────

"""
    CoeffsResult  — output struct for the allocating eval_coeffs interface.
"""
struct CoeffsResult
    theta_p    :: Vector{Float64}
    sigma_p    :: Vector{Float64}
    fp         :: Vector{Float64}
    lambda_hat :: Vector{Float64}
    Lambda     :: Float64
    zeta_ratio :: Float64
end

"""
    eval_coeffs(kernel, Z) -> CoeffsResult

Allocating version of eval_coeffs!. For tests and diagnostics only —
use eval_coeffs! in production.
"""
function eval_coeffs(kernel::CoeffsKernel, Z::ClosureZ)::CoeffsResult
    eval_coeffs!(kernel, Z)
    sc = kernel.scratch
    # Lambda and zeta_ratio are stored in scratch by eval_coeffs!
    return CoeffsResult(
        copy(sc.theta_p), copy(sc.sigma_p), copy(sc.fp),
        copy(sc.lambda_hat), sc.Lambda, sc.zeta_ratio)
end

end # module CoeffsRelax
