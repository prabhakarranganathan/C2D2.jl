# Output file reference

Every run writes plain CSV files with a header row, plus a copy of the input
TOML. This page lists the columns of each file, their meaning and their
units. The notation is that of `docs/input_reference.md`: lengths in units of
the equilibrium coil size E₀, times in units of λ₁,₀ (the equilibrium longest
relaxation time), stresses in units of c k_BT. Section numbers refer to
version 3 of the technical report
([10.5281/zenodo.24035542](https://doi.org/10.5281/zenodo.24035542)).

Common conventions:

- `idx` is the row number, starting at 1 (the equilibrium state).
- `E2` is the chain stretch squared, E² = Σ_p S_p tr M_p / E₀², so E² = 1 for
  chains at rest and E² → N_K at full extension.
- `tau_xx`, `tau_yy`, `tau_zz` are the diagonal polymer stresses divided by
  c k_BT with the sign convention τ = −Σ_p (3 f_p M_p − I), so the stress
  along the stretching axis is negative (compressive-positive sign). The
  material functions below carry the compensating minus sign and are positive.
- `etaE1`, `etaE2` are the polymer extensional viscosities,
  −(τ_ext − τ_comp)/Wi, in units of c k_BT λ₁,₀, for the first and second
  compressive axis (equal in uniaxial extension). `Tr1`, `Tr2` are the
  corresponding Trouton ratios, the polymer extensional viscosity over 3 η_s.
  Both are NaN where Wi = 0.

## Steady extension: `summary.csv`

Written by `StyextManifold.write_summary_csv`, one row per accepted steady
state, in order of increasing stretch along the continuation curve.

| column | meaning |
|---|---|
| `idx` | row number |
| `E2` | chain stretch squared E² |
| `Wi` | Weissenberg number λ₁,₀ ε̇ at this steady state |
| `etaE1`, `etaE2` | polymer extensional viscosities (units c k_BT λ₁,₀) |
| `Tr1`, `Tr2` | Trouton ratios |
| `tau_xx`, `tau_yy`, `tau_zz` | polymer stress / (c k_BT) |
| `Wi_pole` | the Wi at which the steady modal tensors at this stretch would diverge (Sec. 3.2 of the report); the curve approaches it as the chains extend |
| `newton_iters` | Newton iterations the point needed |
| `ds` | arclength step that produced the point |
| `FY_rel` | relative residual of the fallback fixed-stretch solve (0 when the arclength corrector converged) |
| `status` | `eq` equilibrium; `PAC-cnv` arclength corrector converged; `FY-cnv` fixed-stretch fallback converged; `FY-dnc` fallback did not converge |

The curve traced is stress versus Wi: for extension it bends back on itself
near the coil–stretch transition, which is why the points are parametrised by
stretch and not by Wi.

## Transient extension and capillary thinning: `summary.csv`

Written by `UsyextIntegrator.write_summary_csv`, one row per sample strain of
the `[output]` schedule.

| column | meaning |
|---|---|
| `idx` | row number |
| `estrain` | Hencky strain ε = ln(L/L₀); for capillary thinning R/R₀ = exp(−ε/2) |
| `t` | time in units of λ₁,₀ |
| `tWi_ref` | not used (always NaN; kept for compatibility with the MATLAB reference files) |
| `Wi` | Weissenberg number λ₁,₀ ε̇ at this instant (set by the flow protocol; for capillary thinning it follows from the stress balance) |
| `De` | the Deborah number of the run, as input |
| `E2` | chain stretch squared E² |
| `etaE1`, `etaE2` | polymer extensional viscosities (units c k_BT λ₁,₀) |
| `Tr1`, `Tr2` | Trouton ratios |
| `tau_xx`, `tau_yy`, `tau_zz` | polymer stress / (c k_BT) |
| `status` | `ok`, or `terminated` for the last row if the run stopped early (chains at the stretch cap, non-finite state, or step budget exhausted) |

## Capillary thinning: `caber_diag.csv`

Written by `UsyextIntegrator.write_caber_diag_csv`, one row per sample; the
terms of the one-dimensional stress balance of the thinning filament (report
Sec. 4.5.1). For a constant-Wi run the columns from `R` on are NaN.

| column | meaning |
|---|---|
| `idx`, `estrain`, `t`, `Wi`, `De` | as in `summary.csv` |
| `R` | filament radius R/R₀ = exp(−ε/2) |
| `X` | end-plate (geometry) factor of the balance: its Newtonian value (0.7127 viscous, 0.5115 inertial) while the capillary pressure is carried by the solvent, 1.5 once the polymer stress dominates, linear in between |
| `N1_p` | polymer first normal stress difference (τ_zz − τ_rr)/(c k_BT); negative in extension |
| `K_til` | polymer-stress prefactor of the balance: φ U₁ in the viscous limit, φ Oh U₁/De_R in the inertial limit, with U₁ = λ₁,₀ k_BT/(η_s E₀³) |
| `capstress` | capillary term (2X − 1) De / R |
| `polystress` | polymer term K̃ N1_p (≤ 0: the polymer resists thinning) |
| `viscstress` | their sum, the stress the solvent carries; equal to 3 Wi in the viscous limit, where the balance reads 3 Wi = capstress + polystress. In the inertial limit the balance is quadratic in Wi and this column is only the sum of the two terms |

## Steady shear and planar mixed flow: `summary_sweep.csv`, `summary_pac.csv`

Written by `SteadyWiSweep.write_pmf_summary_csv`, one row per Wi of the
`[sweep]` grid (`summary_sweep.csv`) or per continuation point
(`summary_pac.csv`, with `--pac`). Two frames are reported: the strain-axes
frame in which the flow is κ = Wi κ̂(r), and the Fuller–Leal flow frame, which
for simple shear (r = 1) is the laboratory frame with v_x = γ̇ y.

| column | meaning |
|---|---|
| `idx` | row number |
| `E2` | chain stretch squared E² |
| `Wi` | λ₁,₀ times the principal strain rate (the report's convention) |
| `Wi_shear` | 2 Wi = λ₁,₀ γ̇, the usual shear Weissenberg number (γ̇ = √(2 D:D) for every r) |
| `eta_p` | polymer shear viscosity −τ_xy/(λ₁,₀ γ̇), units c k_BT λ₁,₀ (flow frame) |
| `Psi1`, `Psi2` | first and second normal-stress coefficients −(τ_xx − τ_yy)/(λ₁,₀ γ̇)² and −(τ_yy − τ_zz)/(λ₁,₀ γ̇)², units c k_BT λ₁,₀² |
| `tau_xx`, `tau_yy`, `tau_zz`, `tau_xy` | polymer stress / (c k_BT) in the flow frame, compressive-positive sign |
| `tau_11`, `tau_22`, `tau_12` | the same in the strain-axes frame (τ_33 = `tau_zz`) |
| `etaE1`, `etaE2`, `Tr1`, `Tr2` | planar extensional viscosities and Trouton ratios in the strain-axes frame (so r = 0 reproduces steady planar extension) |
| `Wi_pole`, `newton_iters`, `ds`, `FY_rel` | solver diagnostics as for steady extension (`Wi_pole` is Inf for r ≥ 1, where there is no pole; `ds` is the Wi increment in a sweep) |
| `status` | `eq` equilibrium; `FWi-cnv` converged at this Wi; `FWi-fail` Newton failed, row holds the last converged state; `Y_stop` / `X_stop` the stretch cap or the spring pole was reached and the sweep stopped |

## `input.toml`

A copy of the input file, so that a result folder is self-describing.

## CaBER toolkit run folders

The browser toolkit under `tools/c2d2_caber_toolkit/` writes the same
`summary.csv` and `caber_diag.csv` per run, plus a `status.json` with the
outcome (`ok`, `failed` with the error message, or `running`), the wall time
and the number of output rows. See its `README.md`.
