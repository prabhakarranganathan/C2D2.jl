"""
    StyextFlows.jl

Continuation coordinate mapping for the steady-extension PAC solver.

Mirrors MATLAB's
  styext_Wi_from_mu.m / styext_mu_from_Wi.m

The continuation coordinate μ maps the semi-infinite interval Wi ∈ (0, Wi_pole)
onto μ ∈ (0, ∞) via a logistic-type transform that avoids the FENE singularity
at Wi → Wi_pole.

    Wi = Wi_pole · (1 − exp(−μ))
    μ  = −log(1 − Wi/Wi_pole)   [guarded against Wi → Wi_pole]

Public API
----------
  Wi_from_mu(mu, Wi_pole)   -> Wi
  mu_from_Wi(Wi, Wi_pole)   -> mu
"""
module StyextFlows

export Wi_from_mu, mu_from_Wi

"""
    Wi_from_mu(mu, Wi_pole) -> Wi

Map continuation coordinate μ → Wi.

    Wi = Wi_pole · (1 − exp(−μ))
"""
@inline function Wi_from_mu(mu::Float64, Wi_pole::Float64)::Float64
    return Wi_pole * (1.0 - exp(-mu))
end

"""
    mu_from_Wi(Wi, Wi_pole) -> mu

Map Wi → continuation coordinate μ, guarding against Wi → Wi_pole.

    μ = −log(1 − Wi_safe / Wi_pole)

Wi is clamped to (1 − 1e-12) · Wi_pole to avoid log(0).
"""
function mu_from_Wi(Wi::Float64, Wi_pole::Float64)::Float64
    Wi_pole > 0.0 || error("mu_from_Wi: Wi_pole must be positive, got $Wi_pole")
    Wi_safe = min(Wi, (1.0 - 1e-12) * Wi_pole)
    ratio   = max(0.0, Wi_safe / Wi_pole)
    return -log(1.0 - ratio)
end

end # module StyextFlows
