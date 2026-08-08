#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

Ks=(1)
BASE_DIRS=(outputs_final_final)
# BASE_DIRS=(outputs_mmd2)

CONFIGS=(

    # "edm_default|edm_gmm2d|default|cuda:1|50000"
    # "simple_ou|simple_ou|default|cuda:0|50000"
    # "cev_security_price|cev_security_price|default|cuda:0|50000"
    # "cev_security_price_gamma0.5|cev_security_price|default|cuda:0|250000"
    # "coupled_double_well_langevin|coupled_double_well_langevin|default|cuda:0|250000"

    # "smooth_threshold_autoregression|smooth_threshold_autoregression|default|cuda:0|50000"
    # "smooth_threshold_autoregression|smooth_threshold_autoregression|default|cuda:1|250000"

    # "simple_ou_mmd|simple_ou|mmd|cuda:1|250000"
    # "cev_security_price_gamma0.5_mmd|cev_security_price|mmd|cuda:0|250000"
    # "coupled_double_well_langevin_mmd|coupled_double_well_langevin|mmd|cuda:1|250000"
    # "smooth_threshold_autoregression_mmd|smooth_threshold_autoregression|mmd|cuda:1|50000"
    "ddpm_cifar10_hf_mmd|ddpm_cifar10_hf|mmd|cuda:1|5000"
)
N_PARALLEL=1
N_RUNS=25
CI_LEVEL=0.99
DEBUG=0

for i in "${!Ks[@]}"
do
    k="${Ks[$i]}"
    base_dir="${BASE_DIRS[$i]}"
    for config in "${CONFIGS[@]}"
    do
        (
            IFS='|' read -r output_name runner config_name device reference_sample_batch_size <<< "$config"
            output_dir="$base_dir/$output_name"
            mkdir -p "$output_dir"
            debug_flag=""
            if [ "$DEBUG" -eq 1 ]; then
                debug_flag="--debug"
            fi
            uv run python src/ensure_samples.py --runner "$runner" --config "$config_name" --batch_size "$reference_sample_batch_size" --device "$device" --no_compile $debug_flag || exit 1
            uv run python src/compare.py --runner "$runner" --config "$config_name" --output_dir "$output_dir" --device "$device" --n_parallel $N_PARALLEL --n_runs $N_RUNS --crossfit_q_folds "$k" --no_compile $debug_flag || exit 1
            manifest="$output_dir/compare_outputs.json"
            while IFS= read -r csv; do
                uv run python src/plots.py "$csv" --ci "$CI_LEVEL" || exit 1
                sleep 1
            done < <(uv run python - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    payload = json.load(f)
for path in payload["csv_files"]:
    print(path)
PY
            )
        ) &
    done
    wait
done

wait
