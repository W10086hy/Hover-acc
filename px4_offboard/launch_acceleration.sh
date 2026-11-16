#!/bin/bash
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
source /opt/ros/noetic/setup.bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python_bin="${HOVER_VM_PYTHON:-python3}"
if [ -z "${HOVER_VM_PYTHON:-}" ] && [ -x /home/s/full_csdp_v2/.venv/bin/python ]; then
    python_bin=/home/s/full_csdp_v2/.venv/bin/python
fi
exec "$python_bin" -u -B -m px4_offboard.mavros_policy "$@"
