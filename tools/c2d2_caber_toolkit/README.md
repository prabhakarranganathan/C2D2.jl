# C2D2 CaBER Toolkit

This page sets up capillary-thinning (CaBER) simulations with the C2D2 model, writes a self-contained run folder with job scripts, and plots the results. With it you can sweep one of N<sub>K</sub>, φ, h\*<sub>K</sub>, De or the prestretch ε<sub>0</sub> while the other parameters stay fixed. For each distinct fluid (N<sub>K</sub>, φ, h\*<sub>K</sub>), the matching steady uniaxial-extension curve is computed as well, so every transient trajectory can be read against its steady-state attractor.

## 1. One-time installation (about an hour)

1. **Install Julia 1.9 or newer.** The simplest route is juliaup:
    - macOS / Linux: `curl -fsSL https://install.julialang.org | sh`
    - Windows: `winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore`

    Open a new terminal and check that `julia --version` works.
2. **Get the code:**
    `git clone https://github.com/prabhakarranganathan/C2D2.jl.git`
    Note the full path of the folder you get. The page asks for it.
3. **Check the code runs:** `cd C2D2.jl` and then `julia julia/examples/run_usyext_example.jl`. It takes about a minute the first time, because Julia compiles the code. The solver uses only Julia's standard library, so there are no packages to install.
4. **Open `c2d2_caber_toolkit.html`** in a browser. Chrome or Edge can write the run folder directly; Safari and Firefox download it as a zip instead. The page loads two small libraries from a CDN (Plotly for the plots, JSZip for the zip), so it needs an internet connection; “Create run folder…” works without them.

## 2. Setting up a sweep (tab ①)

- Give the sweep a **run tag**. The run folder is named `<tag>_<date>_<time>`, so nothing is ever overwritten.
- Pick a **preset** (the PS-in-DEP solutions of Clasen et al. 2006, with the parameter values we use) or type your own values. The De helper turns your λ<sub>1</sub>, η<sub>s</sub>, σ and R<sub>0</sub> into De.
- Under *More settings*, **Drag interpolation** (`drag_interp`) picks the C2D2 drag scheme: `RP` (the technical report's original blob model, the default), the **bare-blob** model (`BAREI`, `BARECUI`) or the **shielded-blob** model (`OESI`, `OESCUI`). The `…CUI` variants blend the weakly and strongly screened limits smoothly with exponent `p_phi` (default 2); `OESI`/`OESCUI` also take `q_oes` (0 = shielding set by overlap alone, 0.25 = stretch-delayed shielding). `CU` is the earlier experimental blend. See the C2D2.jl README for details.
- Choose the **sweep parameter**, then give its values as a list or as a range. The swept field is greyed out in the form.
- Fill in **where C2D2.jl lives** and, for a cluster, the SLURM settings.
- The right-hand panel lists every run that will be written and previews each file. Then either **Create run folder…** (you choose the folder it goes in) or **Download as .zip**.

## 3. Running

Open a terminal **inside the run folder**:

| What | Command |
|---|---|
| everything, one after another, in one Julia session | `bash run_local.sh` |
| four inputs at a time | `bash run_local.sh 4` |
| on a SLURM cluster (one array task per input) | `sbatch job.slurm` |
| redo runs that already finished | `bash run_local.sh 1 --force` |
| one input by hand | `julia c2d2_run.jl /path/to/C2D2.jl inputs/caber_03_phi6.toml` |

Finished runs are skipped, so you can stop and restart at any time. The steady runs are quick and go first. A single-mode CaBER run takes seconds and a 100-mode run takes a few minutes, plus about 30 s at the start of each Julia session for compilation. On Windows without `bash`, run `julia c2d2_run.jl C:\path\to\C2D2.jl inputs` from the run folder.

## 4. Looking at the results (tab ②)

**Open run folder…** shows what has finished so far. **↻ Reload**, or *reload every 30 s*, picks up runs as they complete, so you can watch a sweep while it runs. (Safari and Firefox can open a folder once but cannot reload it.) The plots are:

- R/R<sub>0</sub>, Wi, the share of the capillary stress carried by the polymer and by the solvent, and M and η̃<sub>p</sub>, all against t/λ<sub>1</sub>, t/t<sub>v</sub>, Hencky strain, or t in seconds once λ<sub>1</sub> is given;
- the trajectories in the η̃<sub>p</sub>–Wi and M–Wi planes, with the steady-extension curve dotted;
- a feature table (below), downloadable as CSV, and any feature plotted against the sweep parameter.

The folder `example_PSDEP_phi_sweep` that comes with the toolkit is a finished single-mode φ sweep for the 2.8 M PS-DEP chains; open it in tab ② to see what the viewer shows.

You can overlay an experiment on the R/R<sub>0</sub> plot. Load a two-column CSV of time [s] and filament diameter [mm], and give D<sub>0</sub>, λ<sub>1</sub> [s] and, if needed, a time shift.

**Features of each CaBER run** (times in units of λ<sub>1</sub>):

| feature | meaning |
|---|---|
| Wi(0) | Wi at the start; Wi(0) = (2X<sub>v</sub>−1)De/3 for an equilibrium start |
| t\*, ε\* | when the polymer stress first equals the solvent stress |
| onset t, onset Wi | first local maximum of Wi after the start: the end of the solvent-dominated stage |
| Wi<sub>e</sub>, t<sub>e</sub> | the minimum of Wi after the onset (the elastic point), searched up to 60% of the final strain; with no overshoot, simply the minimum of Wi |
| t<sub>break</sub> | when R/R<sub>0</sub> first falls below the breakup threshold (default 2×10⁻⁵, editable), or when the chains reach the stretch cap and the run stops (finite extensibility), whichever comes first |

## 5. The run folder

```
<tag>_<date>_<time>/
  sweep.json        fixed parameters, swept parameter and values, list of runs (read by the viewer)
  runs.csv          one line per run: parameters, input file, output folder
  README.txt        short description of this sweep
  c2d2_run.jl       the driver: runs inputs, writes outputs/<id>/
  run_local.sh      run on this machine
  job.slurm         SLURM array job
  inputs/           styext_*.toml (steady extension) and caber_*.toml (CaBER), one per run
  outputs/<id>/     summary.csv, caber_diag.csv (CaBER only), input.toml, status.json
  logs/
```

`status.json` records `ok`, `failed` (with the error message) or `running`, the wall time and the number of output points.

## 6. Units and definitions

The code is written in dimensionless form. λ<sub>1</sub> (= `lambda0` = 1) is the longest relaxation time of the solution, and t<sub>v</sub> = η<sub>s</sub>R<sub>0</sub>/σ is the solvent viscocapillary time.

- **De** = λ<sub>1</sub>/t<sub>v</sub>. Time `t` in the outputs is in units of t<sub>v</sub>, so **t/λ<sub>1</sub> = t/De**.
- **Wi** = ε̇λ<sub>1</sub>, with ε̇ = −2 d ln R/dt the strain rate at the neck.
- **Hencky strain** ε = 2 ln(R<sub>0</sub>/R).
- **M** is the mean-square stretch of the chain relative to equilibrium. The column `E2` = Σ<sub>p</sub> S<sub>p</sub> tr **M**<sub>p</sub> equals N<sub>m</sub> at equilibrium, so M = `E2`/N<sub>m</sub>; the viewer plots M.
- **η̃<sub>p</sub>** (column `etaE1`) is the polymer contribution to the extensional viscosity, scaled with n k<sub>B</sub>T λ<sub>1</sub>, where n is the number density of chains. Stresses `tau_*` are the polymer stress scaled with n k<sub>B</sub>T, in the solver's sign convention (negative in tension).
- The mid-filament stress balance is 3Wi = (2X−1)De/R + K̃ N<sub>1,p</sub>. In terms of the output columns this reads `viscstress = capstress + polystress`, where `viscstress` = 3Wi is the solvent stress, `capstress` = (2X−1)De/R is the capillary stress and `polystress` = K̃N<sub>1,p</sub> is the polymer term. In the solver's sign convention `polystress` is negative in tension, so −`polystress` is the polymer's tensile stress. K̃ (column `K_til`) = U<sub>1</sub>φ weighs polymer stress against solvent stress, with U<sub>1</sub> = U<sub>Z</sub>λ<sub>1</sub>/λ<sub>Z</sub>.

### Output columns

**`outputs/caber_*/summary.csv`**

| column | meaning |
|---|---|
| idx | row number |
| estrain | Hencky strain ε = 2 ln(R<sub>0</sub>/R) |
| t | time / t<sub>v</sub> |
| tWi_ref | unused (NaN) |
| Wi | ε̇λ<sub>1</sub> |
| De | λ<sub>1</sub>/t<sub>v</sub> (constant) |
| E2 | Σ<sub>p</sub> S<sub>p</sub> tr **M**<sub>p</sub>; equals N<sub>m</sub> at equilibrium, and M = E2/N<sub>m</sub> |
| etaE1, etaE2 | η̃<sub>p</sub> from τ<sub>zz</sub>−τ<sub>xx</sub> and τ<sub>zz</sub>−τ<sub>yy</sub> (equal in uniaxial flow) |
| Tr1, Tr2 | polymer Trouton ratios, η̃<sub>p</sub> n k<sub>B</sub>T λ<sub>1</sub>/(3η<sub>s</sub>) |
| tau_xx, tau_yy, tau_zz | polymer stress / n k<sub>B</sub>T, negative in tension (z is the filament axis) |
| status | ok, or terminated where the run stopped early (stretch cap or time limit) |

**`outputs/caber_*/caber_diag.csv`**

| column | meaning |
|---|---|
| idx, estrain, t, Wi, De | as above |
| R | filament radius R/R<sub>0</sub> |
| X | the slender-filament geometry factor in (2X−1)/R, which moves from X<sub>v</sub> ≈ 0.713 (viscous) towards X<sub>e</sub> = 3/2 (elastic) |
| N1_p | polymer normal-stress difference in the solver's sign convention |
| K_til | K̃ = U<sub>1</sub>φ |
| capstress | capillary stress (2X−1)De/R |
| polystress | polymer term K̃ N1_p (negative in tension) |
| viscstress | solvent stress, 3Wi |

**`outputs/styext_*/summary.csv`** (steady uniaxial extension)

| column | meaning |
|---|---|
| idx | point along the continuation |
| E2 | as above, at steady state |
| Wi | ε̇λ<sub>1</sub> |
| etaE1, etaE2, Tr1, Tr2, tau_* | as for CaBER, at steady state |
| Wi_pole, newton_iters, ds, FY_rel | continuation diagnostics |
| status | `eq` (equilibrium point), `FY-cnv` or `PAC-cnv` (the solver branch that converged) |

The viewer uses the rows with Wi > 0. Where the steady curve folds back (coil–stretch hysteresis), several M values share the same Wi.

## 7. Notes

- With **N<sub>m</sub> > 1** the default (`blob`) relaxation spectrum lets each mode cross over from Zimm to Rouse as the chain stretches and the solution crowds. This multimode spectrum has not yet been through peer review. The pure Rouse and Zimm scalings are offered as checks.
- A **prestretch** ε<sub>0</sub> is applied affinely to every mode. With more than one mode the fast modes then start far from equilibrium, which is not physical, so we use prestretch with the single-mode model only.
- Every key the page writes is read by the C2D2.jl v1.0.0 input readers (`StyextRunContext.jl`, `UsyextRunContext.jl`). The inputs are plain TOML and can be edited by hand.
- Problems, questions: Prabhakar Ranganathan, prabhakar.ranganathan@monash.edu.
