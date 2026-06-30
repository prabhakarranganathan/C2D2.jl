# Validation

Validation records for the shipped C2D2 homogeneous-flow solver (STYEXT steady
uniaxial extension; USYEXT transient uniaxial / CaBER-type thinning).

## In-repo regression references

The primary validation is the reference-case regression suite, which compares
solver output against frozen expected CSVs to a relative tolerance of 0.5%:

- Inputs: `julia/experiments/c2d2ref/inputs/`
- Expected output: `julia/experiments/test/reference/`
- Run: `julia julia/experiments/test/test_styext.jl` and `…/test_usyext.jl`

## Sweep-level checks

| Folder | Claim |
|---|---|
| [`2026-06-11_two_limit_generic_checks/`](2026-06-11_two_limit_generic_checks) | Two-limit CaBER (USYEXT): the elastic `Wi_e` plateau is invariant between the viscous and inertio-capillary (`CaberInertialFlow`) regimes; checked across `De` sweeps at φ ∈ {0.01, 0.1, 0.5, 1}. |

> The liquid-bridge / slender-filament (VELB) free-surface solver and its
> capillary-thinning validation (Bhat et al. Fig. 3a, ALE mesh-refinement
> studies, Deborah-number sweeps) are part of a separate, not-yet-released
> package and are not included here.
