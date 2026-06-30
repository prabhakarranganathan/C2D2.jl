# Thin-n-Win — canonical paper data (single source of truth)

_Last updated 2026-06-24 (Aru). This file is the authoritative record of which
experimental systems, parameters, and model runs the TnW paper uses. If the
spreadsheet, the M3 harvested data, or the figures disagree with this file,
this file wins — reconcile the others to it._

## Concentration-basis convention (differs by dataset — this is the thing that kept getting lost)

| Dataset | Samples | h\*K | c\* basis → φ | KM factor |
|---|---|---|---|---|
| Anna (annamckinley) | A1, A2, A3 | 0.01 | c/c\* number-density | — |
| Clasen-Boger (clasenetal) | A, B, C, D, E | 0.03 | **c/c\*_KM** (Martin) | KM = 5 |
| Clasen-DEP (clasenetal) | X, Y | 0.015 | **c/c\*_KM** (Martin) | KM = 0.35 |
| Calabrese (Calabrese2025) | VC1, VC2 (7M); VC3, VC4 (16M) | 0.015 | c/c\* number-density | — |
| Gaillard-visc (Gaillard2024/2025) | G1 | 0.005 | c/c\* Graessley (good solvent) | — |

c/c\*_KM = c[g/cm³]·[η]·KM·e   (sheet formula `=c(g/cu.m)/1e6·[η]·KM·EXP(1)`).
c/c\* number-density = (c/M)·N_A·(b_K·√N_K)³.

**Dropped / out of scope:** Gaillard-aqueous (inertiocapillary), Aisling (end-plate
motion), Clasen Sample Z (c/c\* > 3).

## Canonical model runs (`Plan_Exptl` enabled rows ↔ M3 sim folders)

| Dataset | folder tag | h\*K | M3 output |
|---|---|---|---|
| Anna | `anna` | 0.01 | `outputs/exptl_comparison/anna/hsK_0p01` |
| Clasen-Boger A–D | `clasen_boger_km` | 0.03 | `…/clasen_boger_km/…` ; render `~/tnw_render/data/experimental_clasen_boger_km` |
| Clasen-Boger E | `clasen_e_km` | 0.03 | `…/clasen_e_km/hsK_0p03` (sim jobid 57620577, 2026-06-24) |
| Clasen-DEP | `clasen_dep_km` | 0.015 | `…/clasen_dep_km/…` |
| Calabrese | `calabrese` | 0.015 | `…/calabrese/hsK_0p015` |
| Gaillard-visc | `gaillard` | 0.005 | `…/gaillard/hsK_0p005` |

All other `Plan_Exptl` folder tags are **disabled** (see the `notes` column for why):
`clasen` (0.01/0.02/0.04 — old Sample-E), `clasen_boger` (c/c\* geometric),
`clasen_boger_m15`, `clasen_dep` (geometric), `clasen_dep_min`, `clasen_m15`
(all M⁻¹·⁵ / min-c\* — abandoned 2026-06-22), `gaillard_aq` (out of scope).

## Sample E rebase (2026-06-24)

Sample E was stale at h\*K=0.04 / c/c\* in the canonical KM data. Rebased to
h\*K=0.03 / c/c\*_KM for consistency with A–D via a fresh C2D2 styext+caber sim
(jobid 57620577). E's KM φ = c(g/g)·2091.7 = {0.0165, 0.0209, 0.0661, 0.209,
0.661, 2.092}. Exact steady-state W_ics/W_isc harvested; renorm via the MATLAB
closed-form (U=0.31163, φ̄=0.00918, all φ>φ̄ ⇒ moderately-dilute branch).
**E6 (φ=2.092, semidilute) shows a W_ics/W_isc up-jump → W̄iₑ=0.284; likely a
high-φ feature-detection glitch (cf. Sample-X despike) — despike before final.**

## Per-sample parameter table

See `Thin-n-Win Paper/aru-notes/si_param_table_REVIEW.md` (and `.csv`) — 44 rows,
one per (sample, concentration), with ρ, η_s, γ, M_w, NK, h\*K, φ, R₀, Oh, R\*
(=R_ve), D̄e₀ (Debar), W̄iₑ, φ̄. Still to add there: c\*, λ₁, b_K (computed via
the dataset-note formulae) for the SI table.

## Per-dataset master shift S (Panel G, D̄e\_\* = D̄e₀·S)

Anna 0.80 · Clasen-Boger(A–D) 0.45 · Clasen-DEP 1.40 · Calabrese 1.41 ·
Gaillard 11.  (Clasen-Boger to be re-fit over A–E once rebased E is in the
master_points; the old E-only `clasen` group fit was 1.07.)

## Provenance pointers

- Parameter sheet: `julia/experiments/tnwpaper/parameter-sets.xlsx`
  (`Plan_Exptl` = run plan; `*_expdata` tabs = raw per-sample inputs).
- Renorm closed-form: `matlab/experiments/tnwpaper/tnwpaper_export_pipkin_renorm_params.m`
  (calc_debar / calc_wiebar / calc_phibar; θ=0.22, t̃η_p0=3, H_v=0.1418).
- Harvest/master builders: M3 `~/tnw_render/recovered/` (build_master.py, harvest_*.py, dump_U.jl).
- Figure plotters: `Thin-n-Win Paper/figure-src/plot_exptl.jl` (panels B,E,F,G), `plot_clasen_si.jl`.
