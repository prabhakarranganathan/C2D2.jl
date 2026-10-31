#!/bin/bash
# Submit from inside this folder:   sbatch job.slurm
# One array task per input file (steady runs first). Finished runs are skipped on resubmission.
#SBATCH --job-name=c2d2_phitest
#SBATCH --array=1-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%a.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
C2D2_ROOT="${C2D2_ROOT:-/path/to/C2D2.jl}"
shopt -s nullglob
FILES=( inputs/styext_*.toml inputs/caber_*.toml )
F="${FILES[$((SLURM_ARRAY_TASK_ID-1))]}"
echo "task $SLURM_ARRAY_TASK_ID: $F"
julia --startup-file=no c2d2_run.jl "$C2D2_ROOT" "$F"
