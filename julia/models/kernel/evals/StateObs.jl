"""
    StateObs.jl

Canonical observables from one C2D2 model evaluation.

Mirrors the observable computation in MATLAB's
  styext_state_obs.m / usyext_state_obs.m
(both compute the same extensional-viscosity and Trouton-ratio formulas).

Formulas
--------
Given τ/(c kBT) components (from Stress.polymer_stress), flow parameters Wi,
alphas, and material scales lambda0, etas, ckBT:

    strain_rate  = Wi / lambda0
    etaE         = -(τ_ext - τ_comp) / (strain_rate × lambda0)
                 = -(τ_ext - τ_comp) / Wi             [simplifies]
    Tr           = etaE × ckBT × lambda0 / (3 etas)

The extensional axis is argmax(alphas); the two compressive axes are the
remaining indices.

When Wi ≈ 0 the extensional viscosity and Trouton ratio are undefined:
etaE and Tr are set to NaN and status is forced to :nan_wi.

Public API
----------
  C2D2Obs          — observables struct
  make_obs(...)    — constructor
"""
module StateObs

export C2D2Obs, make_obs

# ── Observables struct ────────────────────────────────────────────────────────

"""
    C2D2Obs

Canonical observables from one C2D2 model evaluation.

Fields
------
Wi      : Weissenberg number
E2      : E² = Σ_p Sp[p] tr M_p  (microstructure norm squared)
etaE1   : extensional viscosity (ext axis − compressive axis 1), dimensionless
etaE2   : extensional viscosity (ext axis − compressive axis 2), dimensionless
Tr1     : Trouton ratio from etaE1
Tr2     : Trouton ratio from etaE2
tau_xx  : τ_xx / (c kBT)
tau_yy  : τ_yy / (c kBT)
tau_zz  : τ_zz / (c kBT)
status  : :ok | :nan_wi (NaN etaE when Wi ≈ 0)
"""
struct C2D2Obs
    Wi      :: Float64
    E2      :: Float64
    etaE1   :: Float64
    etaE2   :: Float64
    Tr1     :: Float64
    Tr2     :: Float64
    tau_xx  :: Float64
    tau_yy  :: Float64
    tau_zz  :: Float64
    status  :: Symbol
end

# ── Constructor ───────────────────────────────────────────────────────────────

"""
    make_obs(Wi, E2, tau, alphas, lambda0, etas, ckBT; status=:ok) -> C2D2Obs

Compute extensional viscosities and Trouton ratios from stress and parameters.

Arguments
---------
Wi      : Weissenberg number
E2      : E² value (microstructure norm squared)
tau     : NamedTuple (xx, yy, zz) of τ/(c kBT) components  [from polymer_stress]
alphas  : 3-element vector; extensional axis = argmax(alphas)
lambda0 : longest relaxation time [time units]
etas    : solvent viscosity
ckBT    : c kB T  (concentration × thermal energy)
status  : optional tag; forced to :nan_wi when Wi ≈ 0 (default :ok)

Formulas
--------
    etaE = -(τ_ext - τ_comp) / Wi
    Tr   = etaE × ckBT × lambda0 / (3 etas)

etaE1/Tr1 use the first compressive axis; etaE2/Tr2 use the second.
For uniaxial flow (two equal compressive alphas) etaE1 = etaE2.
"""
function make_obs(Wi::Float64, E2::Float64,
                  tau,
                  alphas::AbstractVector{Float64},
                  lambda0::Float64, etas::Float64, ckBT::Float64;
                  status::Symbol = :ok)::C2D2Obs

    # Identify extensional and compressive axes
    i_ext  = argmax(alphas)
    i_comp = setdiff(1:3, i_ext)   # 2-element vector, sorted ascending

    taus   = (tau.xx, tau.yy, tau.zz)
    tau_e  = taus[i_ext]
    tau_c1 = taus[i_comp[1]]
    tau_c2 = taus[i_comp[2]]

    if abs(Wi) < 1e-15
        etaE1  = NaN
        etaE2  = NaN
        Tr1    = NaN
        Tr2    = NaN
        status = :nan_wi
    else
        etaE1 = -(tau_e - tau_c1) / Wi
        etaE2 = -(tau_e - tau_c2) / Wi
        Tr1   = etaE1 * ckBT * lambda0 / (3.0 * etas)
        Tr2   = etaE2 * ckBT * lambda0 / (3.0 * etas)
    end

    return C2D2Obs(Wi, E2, etaE1, etaE2, Tr1, Tr2,
                   tau.xx, tau.yy, tau.zz, status)
end

end # module StateObs
