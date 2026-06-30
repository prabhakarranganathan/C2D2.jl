# Julia C2D2 validation reference data

This folder holds MATLAB-generated ground-truth CSVs used by the Julia port
regression tests (`test_styext.jl`, `test_usyext.jl`).

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

## How to regenerate

From the `matlab/` directory in MATLAB:

```matlab
run('experiments/c2d2ref/run_all_refs.m')
```

This runs all 5 MATLAB cases and copies the CSVs here with canonical names.

## Rules

- **Never modify these files by hand.** They are the ground truth.
- To change a reference, re-run the MATLAB case with a deliberate parameter change
  and commit with a message explaining why the ground truth changed.
- The TOML inputs are in `matlab/experiments/c2d2ref/inputs/`.
