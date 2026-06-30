"""
    Stress.jl

Polymer stress from modal conformation tensors.

Implements the formula used by both STYEXT (steady extension) and USYEXT
(unsteady extension / CaBER), mirroring MATLAB's
  styext_polymer_stress.m / usyext_polymer_stress.m
(the two MATLAB files are identical in physics).

Convention
----------
The polymer contribution to the stress tensor, divided by c kBT, is:

    τ_α = -Σ_p ( 3 f_p M_{p,α} - 1 )

where
  M_diag : Nm×3  (columns = xx, yy, zz in the principal flow frame)
  f_p    : FENE Peterlin factor for mode p  (scalar broadcast or Nm-vector)

At equilibrium (M_{p,α} = 1/3, f_p = 1): τ_α = -Σ_p(1-1) = 0. ✓
In extension (zz stretched): τ_zz < 0;  τ_xx > 0. ✓

Public API
----------
  polymer_stress(M_diag, fp)  -> (xx, yy, zz)
"""
module Stress

export polymer_stress

# ── Polymer stress ────────────────────────────────────────────────────────────

"""
    polymer_stress(M_diag, fp) -> NamedTuple{(:xx, :yy, :zz)}

Compute τ/(c kBT) from the diagonal modal conformation tensor and FENE factors.

Arguments
---------
M_diag : Nm×3  (columns = xx, yy, zz)
fp     : Nm-vector of FENE Peterlin factors; pass `fill(f, Nm)` for uniform f.

Formula per component α:
    τ_α = -Σ_p ( 3 f_p M_{p,α} - 1 )
"""
function polymer_stress(M_diag::AbstractMatrix{Float64},
                        fp::AbstractVector{Float64})
    Nm = size(M_diag, 1)
    @assert length(fp) == Nm  "polymer_stress: length(fp) = $(length(fp)) ≠ Nm = $Nm"

    xx = 0.0;  yy = 0.0;  zz = 0.0
    @inbounds for p in 1:Nm
        fp_p  = fp[p]
        xx   -= 3.0*fp_p*M_diag[p,1] - 1.0
        yy   -= 3.0*fp_p*M_diag[p,2] - 1.0
        zz   -= 3.0*fp_p*M_diag[p,3] - 1.0
    end
    return (xx=xx, yy=yy, zz=zz)
end

# Convenience overload: uniform FENE factor (scalar broadcast)
function polymer_stress(M_diag::AbstractMatrix{Float64}, fp_scalar::Float64)
    return polymer_stress(M_diag, fill(fp_scalar, size(M_diag, 1)))
end

end # module Stress
