"""
    MTensor.jl

Modal conformation-tensor operations for the 3-D (Nm×3) M_diag layout used by
STYEXT and USYEXT.  The VELB/liquid-bridge solvers use a 2-D (Nm×2) layout
([M_zz, M_rr]) handled by CoeffsRelax.closure_from_M_diag!.

M_diag layout (Nm×3)
--------------------
Column 1 : M_xx
Column 2 : M_yy
Column 3 : M_zz
(order matches the MATLAB convention; extensional axis chosen by alphas)

tr M_p = M_xx + M_yy + M_zz

Public API
----------
  modal_trace!(trM, M_diag)               — in-place trace per mode
  modal_E_sq(M_diag, Sp)                  — E² = Σ_p Sp[p] tr M_p
  closure_from_M_diag_3d!(Z, ...)         — fill ClosureZ in-place (3-D version)
  closure_from_M_diag_3d(M_diag, ...)     — allocating wrapper
  initial_M_diag(ic_kind, eps_i, Nm, alphas) -> Nm×3 Matrix
"""
module MTensor

using LinearAlgebra: dot
using ..CoeffsRelax: ClosureZ

export modal_trace!, modal_E_sq,
       closure_from_M_diag_3d!, closure_from_M_diag_3d,
       initial_M_diag

# ── Modal trace ───────────────────────────────────────────────────────────────

"""
    modal_trace!(trM, M_diag)

Compute tr M_p = M_p[xx] + M_p[yy] + M_p[zz] for each mode p, writing
results in-place into trM (length Nm).
"""
function modal_trace!(trM::AbstractVector{Float64},
                      M_diag::AbstractMatrix{Float64})
    Nm = size(M_diag, 1)
    @inbounds for p in 1:Nm
        trM[p] = M_diag[p,1] + M_diag[p,2] + M_diag[p,3]
    end
    nothing
end

# ── E² ────────────────────────────────────────────────────────────────────────

"""
    modal_E_sq(M_diag, Sp) -> Float64

E² = Σ_p Sp[p] * tr M_p.  Hot-path inline; avoids allocating a trM buffer.
"""
function modal_E_sq(M_diag::AbstractMatrix{Float64},
                    Sp::AbstractVector{Float64})::Float64
    Nm = size(M_diag, 1)
    s  = 0.0
    @inbounds for p in 1:Nm
        s += Sp[p] * (M_diag[p,1] + M_diag[p,2] + M_diag[p,3])
    end
    return s
end

# ── 3-D closure construction ──────────────────────────────────────────────────

"""
    closure_from_M_diag_3d!(Z, M_diag, fe_model, Nm, Sp, trM_buf)

Fill ClosureZ Z in-place from an Nm×3 diagonal conformation tensor.

This is the 3-D analogue of CoeffsRelax.closure_from_M_diag! (which uses the
Nm×2 uniaxial layout [M_zz, M_rr] with tr = M_zz + 2 M_rr).
Here tr M_p = M_xx + M_yy + M_zz (columns 1–3).

Arguments
---------
Z        : ClosureZ to fill
M_diag   : Nm×3 matrix
fe_model : :fenepme | :fenepm | :fenep | :hookean | :oldroyd_b
Nm       : number of modes
Sp       : Nm-vector of end-to-end spectral weights (used to compute E²)
trM_buf  : Nm-vector scratch buffer (modified in-place)
"""
function closure_from_M_diag_3d!(Z::ClosureZ,
                                  M_diag::AbstractMatrix{Float64},
                                  fe_model::Symbol, Nm::Int,
                                  Sp::Vector{Float64},
                                  trM_buf::Vector{Float64})
    # 1. Trace per mode
    @inbounds for p in 1:Nm
        trM_buf[p] = M_diag[p,1] + M_diag[p,2] + M_diag[p,3]
    end

    # 2. E²
    Y   = dot(Sp, trM_buf)
    Z.Y = Y
    Z.E = sqrt(max(Y, 0.0))

    # 3. Auxiliary variables
    if fe_model === :fenepme || fe_model === :hookean || fe_model === :oldroyd_b
        # Xa is empty — nothing to write
    elseif fe_model === :fenepm
        Z.Xa[1] = sum(trM_buf) / Nm
    elseif fe_model === :fenep
        @inbounds for p in 1:Nm
            Z.Xa[p] = trM_buf[p]
        end
    else
        error("closure_from_M_diag_3d!: unknown fe_model = $fe_model")
    end
    nothing
end

# Allocating wrapper (for tests and non-hot paths)
function closure_from_M_diag_3d(M_diag::AbstractMatrix{Float64},
                                  fe_model::Symbol, Nm::Int,
                                  Sp::Vector{Float64})::ClosureZ
    Na   = (fe_model === :fenepme || fe_model === :hookean ||
             fe_model === :oldroyd_b) ? 0 :
             fe_model === :fenepm     ? 1 : Nm
    Z    = ClosureZ(Na)
    trM  = zeros(Nm)
    closure_from_M_diag_3d!(Z, M_diag, fe_model, Nm, Sp, trM)
    return Z
end

# ── Initial conformation tensor ───────────────────────────────────────────────

"""
    initial_M_diag(ic_kind, eps_i, Nm, alphas) -> Matrix{Float64}  (Nm×3)

Build the initial diagonal conformation tensor for USYEXT integrators.

Arguments
---------
ic_kind : :equilibrium or :uniaxial_affine
eps_i   : pre-strain (scalar, used only for :uniaxial_affine)
Nm      : number of modes
alphas  : 3-element vector of flow-rate ratios; extensional axis = argmax

Convention
----------
At equilibrium M = (1/3)*ones(Nm,3), giving tr M_p = 1 for every mode.

For kind = :uniaxial_affine with micro_affine_strain = eps_i:
    M_ext  = (1/3) * exp( 2 eps_i)   (extensional axis, argmax of alphas)
    M_comp = (1/3) * exp( -  eps_i)  (the two compressive axes)

The extensional axis must be unique (argmax unambiguous); an error is thrown
if two or more alpha values are equal to the maximum.
All Nm modes receive the same initial M row (apply_to_modes = "all").

Matches MATLAB usyext_initial_M_diag.m exactly.
"""
function initial_M_diag(ic_kind::Symbol, eps_i::Float64,
                         Nm::Int,
                         alphas::AbstractVector{Float64})::Matrix{Float64}

    if ic_kind === :equilibrium
        return fill(1.0/3.0, Nm, 3)

    elseif ic_kind === :uniaxial_affine
        # Extensional axis: column with the strictly largest alpha value.
        max_a = maximum(alphas)
        if count(==(max_a), alphas) > 1
            error("initial_M_diag: extensional axis is ambiguous: " *
                  "alphas = $alphas has more than one maximum. " *
                  "uniaxial_affine requires a unique extensional direction " *
                  "(e.g. alphas = [-0.5, -0.5, 1.0]).")
        end
        i_ext  = argmax(alphas)
        i_comp = setdiff(1:3, i_ext)

        M_row         = zeros(3)
        M_row[i_ext]  = (1.0/3.0) * exp( 2.0 * eps_i)
        for ic in i_comp
            M_row[ic] = (1.0/3.0) * exp(-1.0 * eps_i)
        end

        # All modes get the same IC row
        return repeat(reshape(M_row, 1, 3), Nm, 1)

    else
        error("initial_M_diag: unknown ic_kind = $ic_kind. " *
              "Expected :equilibrium or :uniaxial_affine.")
    end
end

end # module MTensor
