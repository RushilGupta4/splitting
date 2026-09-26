#!/usr/bin/env bash
# Train the score model for the 2D Gaussian-mixture EDM benchmark (edm_gmm2d).
# This is the only runner that needs training; the checkpoint is written to
# checkpoints/edm_gmm2d/model_final.pt. The other runners are either analytic
# SDEs or download pretrained models.
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${DEVICE:-cuda:0}"

uv run python src/train.py --runner edm_gmm2d --device "$DEVICE" \
    --num_samples 1000000 --batch_size 50000 --epochs 200 --lr 1e-4 \
    --hidden_dim 128 --num_blocks 4 --sampling_steps 50 \
    --sigma_min 0.002 --sigma_max 80.0 --rho 7.0 --sigma_data 1.0 \
    --P_mean -1.2 --P_std 1.2
