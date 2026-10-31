# Julia C2D2 validation reference data

This folder holds ground-truth CSVs used by the Julia port regression tests
(`test_styext.jl`, `test_usyext.jl`, `test_pmf.jl`). The `styext_*`/`usyext_*`
files are MATLAB-generated; the `pmf_shear_*` files come from an independent
Python full-tensor implementation (`c2d2_tensor.py`, 2026-10-16).

## Files

| File | Source case | Contains |
|---|---|---|
| `styext_ref_01_summary.csv` | FENE-PM, constant, Nm=1, phi=1.0 | PAC continuation path: E2, Wi, etaE1, etaE2, tau_* |
| `styext_ref_02_summary.csv` | FENE-PM, c2d2, Nm=3, phi=0.1 | same columns |
| `usyext_ref_01_summary.csv` | FENE-PM, constant, Nm=1, phi=0.03, De=1 | trajectory: estrain, t, Wi, E2, etaE1, etaE2, tau_* |
| `usyext_ref_01_caber_diag.csv` | same | CaBER diagnostics: estrain, t, Wi, R, X, N1_p |
| `usyext_ref_02_summary.csv` | FENE-PM, c2d2, Nm=3, phi=0.1, De=0.1 | same |
| `usyext_ref_02_caber_diag.csv` | same | same |
| `usyext_ref_03_summary.csv` | FENE-PM, constant, Nm=1, phi=0.09, De=0.02 | near-Bhat regime |
| `usyext_ref_03_caber_diag.csv` | same | same |
| `pmf_shear_N1_fenepm_c2d2.csv` | steady simple shear, FENE-PM, c2d2 (RP), Nm=1, phi=0, NK=500, h*K=0.25 | Wi (=λ₀γ̇), tau_xy, N1, N2, eta_p, Psi1, Y, residual |
| `pmf_shear_N10_rouse_fenepm_const.csv` | as above, constant drag, Nm=10, fixed Rouse p⁻² spectrum | same |
| `pmf_shear_N10_zimm_fenepm_c2d2.csv` | as above, c2d2 (RP), Nm=10, fixed Zimm p⁻³ᐟ² spectrum | same |
| `oesi_reference_values.csv` | drag kernel only: NK=5000, partialdraining, h*K ∈ {0.25, 0.17, 0.025}, 8 φ × 6 E | ζ/ζ_Z for the 21-Oct OESI/OESCUI (old 1 + ln J, p_phi = 1.86; **superseded**, columns no longer tested) and RPI (= RP, still tested), full double precision; from `oesi.py` (2026-10-21, copy in `docs/validation/2026-10-21_oesi/`) |
| `oesi_v2_reference_values.csv` | drag kernel only: partialdraining, (h*K, NK) ∈ {(0.25, 5000), (0.17, 5000), (0.025, 5000), (0.04, 2426), (0.017, 2426), (0.04, 11186), (0.017, 11186)}, 13 φ × 10 E, 340 rows | ζ/ζ_Z for BAREI, BARECUI, OESI, OESCUI (D(A) shape factor, p_phi = 2.0), full double precision; from `oesi.py:zeta_v2` (2026-10-22, copy in `docs/validation/2026-10-22_oesi_v2/`) |
| `oesi_v2q_reference_values.csv` | same 340 rows | the four columns above plus `zeta_oes_q0.25`, `zeta_oes_cui_q0.25`: OESI/OESCUI with the stretch-delayed dressing ψ = min(φ/E^q, 1), q_oes = 0.25, p_phi = 2.0; from `oesi.py:zeta_v2(..., q=0.25)` (2026-10-22 pm) |

PMF reference conventions (see file headers): `Wi = λ₀γ̇` is the **lab shear
rate** (= 2 × the report-appendix Wi the Julia solver uses internally); stress
is tensile-positive `+Σ(3f_pM_p − I)` (Julia `Stress.jl` is the opposite sign,
so `tau_xy` flips sign); FENE-PM `Q∞² = NK/Nm` (STYEXT convention). The Julia
TOML twins are in `julia/experiments/pmf/inputs/`.

## How to regenerate

From the `matlab/` directory in MATLAB:

```matlab
run('experiments/c2d2ref/run_all_refs.m')
```

This runs all 5 MATLAB cases and copies the CSVs here with canonical names.

The `pmf_shear_*` CSVs were generated with an independent Python full-tensor
implementation of the steady-shear C2D2 equations (`c2d2_tensor.py`, not in this
repo). The `oesi_*` CSVs come from `docs/validation/2026-10-22_oesi_v2/oesi.py`.

## Rules

- **Never modify these files by hand.** They are the ground truth.
- To change a reference, re-run the MATLAB case with a deliberate parameter change
  and commit with a message explaining why the ground truth changed.
- The TOML inputs are in `matlab/experiments/c2d2ref/inputs/`.
