#!/bin/bash
#SBATCH --job-name=tnw_geninert
#SBATCH --account=YOUR_PROJECT
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

# TnW generic inertial sweeps: U_R in {0.01, 1}, constant + c2d2 drag, NK=3000,
# hsK=0.05, De_R floored at 0.1. Exports plans from Plan_Generic then runs all sweeps.

set -uo pipefail
REPO="${HOME}/projects/C2D2-Model"
cd "${REPO}"
module purge
module load julia/1.10.2
echo "Job ${SLURM_JOB_ID} started: $(date) on $(hostname) — block='${1:-both}'"
julia --project=julia --threads=1 julia/experiments/tnwpaper/run/run_generic_inertial.jl "${1:-}"
echo "Job finished: $(date)"
