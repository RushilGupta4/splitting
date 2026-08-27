#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

# BASE_DIRS=(outputs_paper_final)
BASE_DIRS=(outputs_paper_final_2)

# name|runner|config|device|reference_batch|num_queries|k_max|mlp_workers
CONFIGS=(
    "edm_default|edm_gmm2d|default|cuda:1|50000|1024|64|25"
    # "simple_ou|simple_ou|default|cuda:0|1000000|1024|64|25"
    # "coupled_double_well_langevin|coupled_double_well_langevin|default|cuda:1|1000000|1024|64|25"

    # "simple_ou_mmd|simple_ou|mmd|cuda:0|1000000|1024|64|25"/gao
    # "coupled_double_well_langevin_mmd|coupled_double_well_langevin|mmd|cuda:1|1000000|1024|64|25"
    # "ddpm_cifar10_hf_mmd|ddpm_cifar10_hf|mmd|cuda:1|5000|4096|512|1"
)
N_PARALLEL=100
N_RUNS=2500
# N_PARALLEL=1
# N_RUNS=50
CI_LEVEL=0.95
DEBUG=0
K=1

for base_dir in "${BASE_DIRS[@]}"
do
    for config in "${CONFIGS[@]}"
    do
        (
            IFS='|' read -r output_name runner config_name device reference_sample_batch_size num_queries k_max mlp_workers <<< "$config"
            output_dir="$base_dir/$output_name"
            mkdir -p "$output_dir"
            debug_flag=""
            if [ "$DEBUG" -eq 1 ]; then
                debug_flag="--debug"
            fi
            uv run python src/ensure_samples.py --runner "$runner" --config "$config_name" --batch_size "$reference_sample_batch_size" --device "$device" $debug_flag || exit 1
            uv run python src/compare.py --runner "$runner" --config "$config_name" --output_dir "$output_dir" --device "$device" --n_parallel $N_PARALLEL --n_runs $N_RUNS --crossfit_q_folds "$K" --crossfit_q_num_queries "$num_queries" --crossfit_q_k_max "$k_max" --crossfit_q_mlp_run_parallelism "$mlp_workers" $debug_flag || exit 1
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

# N_PARALLEL=1000
# N_RUNS=10000
# for base_dir in "${BASE_DIRS[@]}"
# do
#     output_dir="$base_dir/ou_oracle"
#     mkdir -p "$output_dir"
#     debug_flag=""
#     if [ "$DEBUG" -eq 1 ]; then
#         debug_flag="--debug"
#     fi
#     uv run python src/ensure_samples.py --runner ou_oracle --config default --batch_size 1000000 --device cuda:1 $debug_flag || exit 1
#     uv run python src/compare.py --runner ou_oracle --config default --output_dir "$output_dir" --device cuda:1 --n_parallel $N_PARALLEL --n_runs $N_RUNS $debug_flag || exit 1
# done
