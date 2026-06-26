#!/bin/bash
#PBS -N splitting_ddpm_compare
#PBS -o out1.log
#PBS -e err1.log
#PBS -l nodes=gpu-h100:ppn=50

# cd $PBS_O_WORKDIR

CONFIGS=(
    # "outputs/ddpm_joint_only|ddpm_gmm2d|joint_only|cuda:0"
    # "outputs/ddpm_independent_only|ddpm_gmm2d|independent_only|cuda:1"
    # "outputs/edm_default|edm_gmm2d|default|cuda:0"
    # "outputs/ddpm_default|ddpm_gmm2d|default|cuda:1"
    "outputs2/simple_ou|simple_ou|default|cuda:0"
    "outputs2/cev_security_price|cev_security_price|default|cuda:0"
    "outputs2/smooth_threshold_autoregression|smooth_threshold_autoregression|default|cuda:0"
)
N_PARALLEL=50
CI_LEVEL=0.5
N_RUNS=100

for config in "${CONFIGS[@]}"
do
    (
        IFS='|' read -r output_dir runner config_name device <<< "$config"
        mkdir -p "$output_dir"
        uv run python src/ensure_samples.py --runner "$runner" --config "$config_name" --device "$device" --debug || exit 1
        uv run python src/compare.py --runner "$runner" --config "$config_name" --output_dir "$output_dir" --device "$device" --n_parallel $N_PARALLEL --n_runs $N_RUNS --debug || exit 1
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
