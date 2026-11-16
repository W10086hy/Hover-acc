#!/bin/bash
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
out="${1:?Usage: run_dual_suite.sh OUTPUT_ROOT [mavros|mavlink]}"
source="${2:-mavros}"
extra=()
if [ -n "${3:-}" ]; then extra=(--pitch-rate-k "$3"); fi
if [ -n "${4:-}" ]; then extra+=(--p3-calibration "$4"); fi
case "$source" in mavros|mavlink) ;; *) echo 'Invalid policy source' >&2; exit 2;; esac
mkdir -p "$out"
for wind in 1 5; do
    seed=2026114501
    ramp=0
    if [ "$wind" = 5 ]; then seed=2026114503; ramp=5; fi
    for interface in acceleration rate_thrust; do
        case_dir="$out/${interface}_wind${wind}"
        if [ ! -f "$case_dir/audit.json" ]; then
            bash px4_offboard/launch_dual_sitl.sh --interface "$interface" --policy-source "$source" --output "$case_dir" "${extra[@]}" \
                --assets px4_offboard/acceleration_dual/assets --seconds 120 --wind "$wind" \
                --wind-ramp-s "$ramp" --seed "$seed" >"$out/${interface}_wind${wind}.log" 2>&1
        fi
        "${HOVER_VM_PYTHON:-/home/s/full_csdp_v2/.venv/bin/python}" -B -m px4_offboard.dual_audit "$case_dir" \
            --interface "$interface" --policy-source "$source" --seed "$seed" --wind-mps "$wind" \
            --wind-ramp-s "$ramp" --requested-duration-s 120 --acceleration-filter-s .065 \
            --assets px4_offboard/acceleration_dual/assets "${extra[@]}"
    done
done
