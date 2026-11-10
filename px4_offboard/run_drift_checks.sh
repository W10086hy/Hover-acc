#!/usr/bin/env bash
set -euo pipefail
cd /home/s/hover_bptt_px4
out=${1:?Pass the selected drift experiment directory}
exec > "$out/vm_progress.log" 2>&1
python=/home/s/full_csdp_v2/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$python" -m zipfile -e "$out/vm_assets.zip" "$out"
"$python" -u -B -m px4_offboard.run --assets "$out/assets" --output "$out/results/px4_hover_wind5" \
  --duration 120 --cart-speed 0 --wind-peak 5 --seed 20272982 > "$out/hover_wind5.log" 2>&1
"$python" -u -B -m px4_offboard.run --assets "$out/assets" --output "$out/results/px4_track4_wind1" \
  --duration 120 --cart-speed 4 --wind-peak 1 --seed 20272982 > "$out/track4_wind1.log" 2>&1
tar -cf "$out/px4_results.tar" -C "$out" \
  --exclude='*/etc' --exclude='*/test_data' results
printf '%s\n' complete
