#!/bin/bash
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "${HOVER_VM_PYTHON:-/home/s/full_csdp_v2/.venv/bin/python}" -u -B -m px4_offboard.run --assets stable_retest/slack5cm/assets "$@"
