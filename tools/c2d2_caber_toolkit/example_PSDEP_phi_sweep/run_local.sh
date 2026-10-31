#!/usr/bin/env bash
# Run every input in this folder on this machine.
#   bash run_local.sh        one after another, in one Julia session
#   bash run_local.sh 4      four inputs at a time
#   bash run_local.sh 1 --force    redo runs that already finished
# Finished runs are skipped, so you can stop and restart at any time.
cd "$(dirname "$0")"
C2D2_ROOT="${C2D2_ROOT:-/path/to/C2D2.jl}"
JULIA="${JULIA:-julia}"
NPROC="${1:-1}"
shift || true
mkdir -p logs
if [ "$NPROC" -le 1 ]; then
  $JULIA --startup-file=no c2d2_run.jl "$C2D2_ROOT" inputs "$@" 2>&1 | tee logs/run_local.log
else
  # steady runs first (they are quick), then the CaBER runs
  printf "%s\n" inputs/styext_*.toml inputs/caber_*.toml | \
    xargs -P "$NPROC" -I{} sh -c '$0 --startup-file=no c2d2_run.jl "$1" "$2" $3 > "logs/$(basename "$2" .toml).log" 2>&1; grep -E "ok |FAILED|already done" "logs/$(basename "$2" .toml).log" | tail -n 1' "$JULIA" "$C2D2_ROOT" {} "$*"
fi
