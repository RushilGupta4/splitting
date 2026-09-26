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
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="${BASE_DIR:-outputs_paper_final}"
DEVICE="${DEVICE:-cuda:0}"
DEBUG="${DEBUG:-0}"
CI_LEVEL=0.95

# name|runner|config|reference_batch|n_runs|n_parallel|phase1
# phase1 is "ks" (CrossFit-Q Phase 1, used by the KS configs) or "mmd" (sibling
# Phase 1, configured entirely in the runner's configs.py).
EXPERIMENTS=(
    "simple_ou|simple_ou|default|1000000|2500|100|ks"
    "coupled_double_well_langevin|coupled_double_well_langevin|default|1000000|2500|100|ks"
    "edm_default|edm_gmm2d|default|50000|2500|100|ks"
    "ddpm_cifar10_hf_mmd|ddpm_cifar10_hf|mmd|5000|50|5|mmd"
    "ldm_ffhq_mmd|ldm_ffhq|mmd|1024|50|2|mmd"
    "ou_oracle|ou_oracle|default|1000000|10000|1000|ks"
)

# CrossFit-Q Phase 1 settings used for every KS experiment.
CROSSFIT_ARGS=(
    --crossfit_q_folds 1
    --crossfit_q_num_queries 1024
    --crossfit_q_k_max 64
    --crossfit_q_mlp_run_parallelism 25
)

debug_flag=""
if [ "$DEBUG" -eq 1 ]; then
    debug_flag="--debug"
fi

for experiment in "${EXPERIMENTS[@]}"
do
    IFS='|' read -r name runner config ref_batch n_runs n_parallel phase1 <<< "$experiment"
    if [ "$#" -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then
        continue
    fi

    output_dir="$BASE_DIR/$name"
    mkdir -p "$output_dir"

    phase1_args=()
    if [ "$phase1" = "ks" ]; then
        phase1_args=("${CROSSFIT_ARGS[@]}")
    fi

    uv run python src/ensure_samples.py --runner "$runner" --config "$config" \
        --batch_size "$ref_batch" --device "$DEVICE" $debug_flag || exit 1
    uv run python src/compare.py --runner "$runner" --config "$config" \
        --output_dir "$output_dir" --device "$DEVICE" \
        --n_runs "$n_runs" --n_parallel "$n_parallel" \
        ${phase1_args[@]+"${phase1_args[@]}"} $debug_flag || exit 1

    while IFS= read -r csv; do
        uv run python src/plots.py "$csv" --ci "$CI_LEVEL" || exit 1
    done < <(uv run python - "$output_dir/compare_outputs.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    payload = json.load(f)
for path in payload["csv_files"]:
    print(path)
PY
    )
done
