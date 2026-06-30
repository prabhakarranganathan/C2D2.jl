# Two-limit CaBER — generic sweep-level checks (2026-06-11)

**Code under test:** commit `4d4d361` — `CaberInertialFlow` (Oh ≪ 1 inertio-capillary
branch) + pre-stretch IC, implemented by Aru, reviewed/wired/committed by Ko.
**Machine:** RP's Mac, Julia 1.12, base USYEXT chain (stdlib only).
**Sweeps:** NK=3000, hsK=0.05, fenepm, constant drag, Nm=1, φ ∈ {0.01, 0.1, 0.5, 1}.

| Sweep | Regime | De grid |
|---|---|---|
| S03 (`generic_validation`) | viscous | De_v = 0.01 … 100 |
| S08 | viscous | De_v = 300 … 3000 |
| S05 | inertial (Oh=0.005, X_R=0.5115) | De_R = 0.01 … 100 |
| S07 | inertial, matched grid | De_R = Oh·De_v = 5e-5 … 15 |

Analysis script: `julia/experiments/tnwpaper/run/analyze_two_limit_checks.py`
(re-run from repo root with the framework python3.10; full console output in
`analysis_output.txt`; merged tables + per-sweep `Wi_e_vs_De_*.csv` here).

## Headline result — regime-INVARIANT elastic plateau (Check A)

At matched De_v = De_R/Oh, where the inertial grid reaches the plateau:

- **φ=0.5: viscous Wi_e = 0.7192, inertial Wi_e = 0.7246 → agreement to 0.75%.** ✓
- φ=1: inertial lowest converged cell (De_R=0.01) gives 0.91 vs viscous 0.70 —
  approaching from above, not yet converged (the De_R ≤ 5e-4 cells are
  strain-mode stiff, see Open issues).
- φ=0.1: inertial still falling at De_R=5e-5 (1.12 vs viscous 0.88) — needs
  De_R ~ 1e-5.
- φ=0.01: no inertial arrest at any De_R on this grid (trajectory is a pure
  pure-solvent exponential to FENE saturation; only fallback features).

**Structural finding:** the inertial Pipkin curve leaves the shared plateau
~2 decades earlier in De_v than the viscous one (corner at De_v ~ 0.03 vs ~3).
Matched-grid comparisons MUST run De_R down to ~1e-5 (Oh=0.005).

## Check B — regime-DEPENDENT post-corner rise (qualitative ✓, asymptote needs NK=30000)

Fits over the rising min-feature branch (above 1.5× plateau):

| regime | φ=0.1 | φ=0.5 | φ=1 | theory 4β/3 |
|---|---|---|---|---|
| viscous | 0.82 | 0.79 | 0.74 | 4/3 |
| inertial | 0.28 | 0.31 | 0.45 | 8/9 |

The inertial rise is much shallower than the viscous one at matched windows —
the regime-dependent signature is clearly present — but neither reaches its
asymptote at NK=3000: the FENE-saturated zone (no Wi minimum; Wi_e_fallback
∝ De exactly: 0.426·De_v viscous, 2.23·De_R inertial, φ-independent) cuts the
branch off early. Aru's single-trajectory affine checks at NK=30000 already
nailed the β exponents (E²~Wi^{2.56} vs theory 8/3; viscous 3.7 vs 4).
**Asymptotic sweep-level slopes need an NK=30000 sweep family.**

## φ^{-4/3} dilute plateau — not testable on this φ family

Measured plateau-vs-φ exponent ≈ -0.3 (viscous) — but the family
φ ∈ {0.01..1} is mostly semidilute (plateau saturates at ~0.70 for φ ≥ 0.5).
A dilute family (φ ≤ 1e-2, and likely NK=30000) is required.

## Open issues

1. **Strain-mode stiffness at deep plateau** (φ=1, De_R ≤ 5e-4): Wi ~ 1e-4 ⇒
   relaxation/Wi blows up the strain-ODE; the stepper crawls at dstrain_min
   (>40 min/cell, killed). `integration.Wi_min_strain_independent` (auto →
   time-mode) is the designed remedy, but the first time-mode test also
   exceeded 4 min — needs proper investigation before deep-plateau production.
   Dropped cells: φ=1 × De_R ∈ {1.5e-4, 5e-4} (and the trimmed S07 φ=1 re-run
   was abandoned with it; φ=1 inertial coverage starts at De_R=0.01 from S05).
2. φ=0.01 inertial: no plateau exists at NK=3000 (see above) — expected
   physics or grid artifact, RP to advise.
3. S06 (c2d2-drag inertial) φ=0.5/φ=1 sweeps incomplete (same slow cells);
   c2d2 vs constant differences at the completed cells are negligible at
   high De and small at low De.
