#!/bin/bash
#SBATCH --job-name=tnw_inert_oh0p01
#SBATCH --account=YOUR_PROJECT
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # 1 thread — physics modules are single-thread
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tnw_inert_oh0p01_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@institution.edu

# ──────────────────────────────────────────────────────────────────────────────
# TnW generic-behaviour — INERTIO-CAPILLARY device locus, Oh = 0.01.
# Plan_Generic sweep_ids 21–24 (Oh-primary pivot, 2026-06-12):
#   S21  styext steady curve, constant drag, phi=0.5
#   S22  styext steady curve, c2d2 drag,     phi=[0.01,0.1,0.5,1]  (phi-dependent)
#   S23  caber_inertial, constant drag, phi ladder, De_R=[0.01..100], Oh=0.01
#   S24  caber_inertial, c2d2 drag,     phi ladder, De_R=[0.01..100], Oh=0.01
# X_R=0.51, NK=3000, hsK=0.05.  De values are De_R (Rayleigh-time Deborah);
# the polymer prefactor U_R = Oh*U_1/De_R is formed per-cell, tracing the locus.
#
# Requires repo at commit >= fbd69cd (step-budget cap fix — low-De_R cells are
# reject-dominated and would otherwise ignore any wall cap).
#
# Output lands under the repo's Lustre clone:
#   ~/YOUR_PROJECT/C2D2-Model/julia/experiments/tnwpaper/outputs/generic_behaviour/
#       generic_inertial_oh0p01/hsK_0p05/<sweep folders>/
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${HOME}/projects/C2D2-Model"          # symlink → ~/YOUR_PROJECT/C2D2-Model (Lustre)
TNW="${REPO}/julia/experiments/tnwpaper"
PLAN="${TNW}/run/generic_inertial_oh0p01_plan.toml"

# SLURM --output is relative to the submit dir; keep logs beside this script.
mkdir -p "${TNW}/m3/logs"
cd "${REPO}/julia"

# Sanity: output target must be on Lustre, not the 20 GB NFS home.
fstype=$(df -PT "${TNW}/outputs" 2>/dev/null | awk 'NR==2{print $2}' || echo "?")
echo "tnwpaper/outputs fstype: ${fstype}  (expect lustre, NOT nfs4)"

module purge
module load julia/1.10.2

echo "Job started: $(date) on $(hostname)"
echo "Plan: ${PLAN}"

# Optional (commented): regenerate the plan from the xlsx as the source of truth.
# Needs XLSX.jl + DataFrames in the M3 Manifest; the committed plan is bit-identical.
# julia --project=. experiments/tnwpaper/run/export_plan_from_xlsx.jl \
#   generic_inertial_oh0p01 "${PLAN}" Plan_Generic

julia --project=. --threads=1 \
  experiments/tnwpaper/run/run_generic_validation.jl "${PLAN}"

echo "Job finished: $(date)"
