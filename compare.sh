#!/bin/bash
#PBS -N splitting_ddpm_compare
#PBS -o out1.log
#PBS -e err1.log
#PBS -l nodes=gpu-h100:ppn=50

# cd $PBS_O_WORKDIR

CONFIGS=(
    # "outputs/ddpm|ddpm_gmm2d|default|cuda:1|50000"
    # "outputs/ddpm_mnist|ddpm_mnist|default|cuda:1|12500"
    # "outputs/edm_default|edm_gmm2d|default|cuda:0|50000"
    "outputs/simple_ou|simple_ou|default|cuda:1|50000"
    # "outputs/cev_security_price|cev_security_price|default|cuda:0|50000"
    # "outputs/smooth_threshold_autoregression|smooth_threshold_autoregression|default|cuda:0|50000"
)
N_PARALLEL=1
CI_LEVEL=0.9
N_RUNS=10
DEBUG=1

for config in "${CONFIGS[@]}"
do
    (
        IFS='|' read -r output_dir runner config_name device reference_sample_batch_size <<< "$config"
        mkdir -p "$output_dir"
        debug_flag=""
        if [ "$DEBUG" -eq 1 ]; then
            debug_flag="--debug"
        fi
        # uv run python src/ensure_samples.py --runner "$runner" --config "$config_name" --batch_size "$reference_sample_batch_size" --device "$device" $debug_flag || exit 1
        uv run python src/compare.py --runner "$runner" --config "$config_name" --output_dir "$output_dir" --device "$device" --n_parallel $N_PARALLEL --n_runs $N_RUNS $debug_flag || exit 1
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
    )
done

wait
