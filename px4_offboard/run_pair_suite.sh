#!/usr/bin/env bash
# Run the frozen 5 m P2/P3 comparison serially because PX4 uses fixed ports.
set -eu
assets="$1"
output="$2"
python="${HOVER_VM_PYTHON:-/home/s/full_csdp_v2/.venv/bin/python}"
if pgrep -x px4 >/dev/null; then
    echo 'Existing PX4 instance is using the SITL ports' >&2
    exit 1
fi
mkdir "$output"
for wind in 1 5; do
    for mode in p2 p2_p3; do
        options=()
        if [ "$mode" = p2 ]; then options+=(--no-residual); fi
        case_path="$output/wind${wind}_${mode}"
        echo "BEGIN $case_path"
        "$python" -u -B -m px4_offboard.run --assets "$assets" --output "$case_path" \
            --duration 120 --cart-speed 0 --wind-peak "$wind" --seed 20683131 "${options[@]}"
        if ! "$python" -u -B -m px4_offboard.audit "$case_path" --assets "$assets"; then
            echo "AUDIT_FAILED $case_path"
        fi
    done
done
