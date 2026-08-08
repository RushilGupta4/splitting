#!/bin/bash
#PBS -N splitting_ddpm_train
#PBS -o train_out.log
#PBS -e train_err.log
#PBS -l nodes=gpu-h100:ppn=50

# cd $PBS_O_WORKDIR

TRAIN_CONFIGS=(
    "edm_gmm2d|cuda:1|--num_samples 1000000 --batch_size 50000 --epochs 200 --lr 1e-4 --hidden_dim 128 --num_blocks 4 --sampling_steps 50 --sigma_min 0.002 --sigma_max 80.0 --rho 7.0 --sigma_data 1.0 --P_mean -1.2 --P_std 1.2"
)

for config in "${TRAIN_CONFIGS[@]}"
do
    (
        IFS='|' read -r runner device extra_args <<< "$config"
        uv run python src/train.py --runner "$runner" --device "$device" $extra_args || exit 1
    ) &
done

wait
