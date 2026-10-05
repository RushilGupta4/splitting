#!/usr/bin/env bash
# Reproduce the paper experiments. For each experiment this
#   1. builds (or reuses) the cached reference samples,
#   2. runs the comparison sweep (compare.py),
#   3. plots every summary CSV the sweep wrote.
#
# Usage:
#   ./compare.sh                     # all experiments below
#   ./compare.sh simple_ou edm_default   # a subset, by name
#   DEVICE=cuda:1 DEBUG=1 ./compare.sh simple_ou
#
# edm_default needs the trained checkpoint from ./train.sh.
set -euo pipefail

cd "$(dirname -- "$0")"

BASE_DIR="${BASE_DIR:-outputs_paper_final_5}"
DEVICE="${DEVICE:-cuda:0}"
DEBUG="${DEBUG:-0}"
CI_LEVEL=0.95

# name|runner|config|reference_batch|n_runs|n_parallel|extra_args
EXPERIMENTS=(
    # "simple_ou|simple_ou|default|1000000|2500|100|"
    # "coupled_double_well_langevin|coupled_double_well_langevin|default|1000000|2500|100|"
    # "edm_default|edm_gmm2d|default|50000|2500|100|"
    # "ou_oracle|ou_oracle|default|1000000|10000|1000|"

    "ddpm_cifar10_hf_timing|ddpm_cifar10_hf|timing|5000|10|1|--timing"
    "ldm_ffhq_timing|ldm_ffhq|timing|1024|10|1|--timing"
    "ddpm_cifar10_hf_mmd|ddpm_cifar10_hf|mmd|5000|50|5|"
    "ldm_ffhq_mmd|ldm_ffhq|mmd|1024|50|2|"
)

debug_flag=""
if [ "$DEBUG" -eq 1 ]; then
    debug_flag="--debug"
fi

for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name runner config ref_batch n_runs n_parallel extra_args <<< "$experiment"
    if [ "$#" -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then
        continue
    fi
    output_dir="$BASE_DIR/$name"

    uv run python src/ensure_samples.py --runner "$runner" --config "$config" \
        --batch_size "$ref_batch" --device "$DEVICE" $debug_flag
    uv run python src/compare.py --runner "$runner" --config "$config" \
        --output_dir "$output_dir" --device "$DEVICE" \
        --n_runs "$n_runs" --n_parallel "$n_parallel" $extra_args $debug_flag
    uv run python -c 'import json, sys; print(*json.load(open(sys.argv[1]))["csv_files"], sep="\n")' \
        "$output_dir/compare_outputs.json" |
        while IFS= read -r csv; do
            uv run python src/plots.py "$csv" --ci "$CI_LEVEL"
        done
done
