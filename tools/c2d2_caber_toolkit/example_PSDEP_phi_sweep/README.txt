C2D2 CaBER sweep "phitest"   (folder phitest_2026-10-12_100315)
Written by the C2D2 CaBER toolkit on Mon Oct 12 2026 10:03:15 GMT+1100 (Australian Eastern Daylight Time).

What is here
  sweep.json      the sweep: fixed parameters, swept parameter and values, list of runs
  runs.csv        one line per run: parameters, input file, output folder
  inputs/         one TOML input per run (styext_* = steady extension, caber_* = CaBER)
  c2d2_run.jl     runs the inputs (needs Julia >= 1.9 and a clone of C2D2.jl)
  run_local.sh    run on this machine:  bash run_local.sh   (or  bash run_local.sh 4  for 4 at a time)
  job.slurm       run on a SLURM cluster:  sbatch job.slurm
  outputs/<id>/   results of each run, created as the runs finish
  logs/           run logs

Fixed: NK = 3774, hK_star = 0.015, Nm = 1, spectrum_model = auto, De = 14.59857, eps0 = 0, fe_model = fenepm, drag_model = c2d2, strain_max = 25
Sweep: phi = 1.5, 3, 6
Runs : 3 steady + 3 CaBER

To look at the results, open c2d2_caber_toolkit.html, go to "View results" and open this folder.
The meaning of every output column is in the toolkit's "How to run / output files" tab and in its README.
