#!/bin/bash

OPTS=("gd_pc_blocks")
MULT=1
WARMUP=0
LR=1e-4

for OPT in "${OPTS[@]}"; do
    echo "------------------------------------------------"
    echo "Submitting batch for optimizer: $OPT"

    # 1. Extract Base Optimizer (e.g., "gd_pc_blocks" -> "gd")
    BASE_OPT=${OPT%%_*}

    # 2. Build extra flags based on the string content
    EXTRA_FLAGS=""
    [[ "$OPT" == *"_pc"* ]] && EXTRA_FLAGS="$EXTRA_FLAGS --use_pc"
    [[ "$OPT" == *"_blocks"* ]] && EXTRA_FLAGS="$EXTRA_FLAGS --use_blocks"

    echo "MLP / Telescope"
    sbatch --job-name="mlp_tele_${OPT}" run_es.sh python run_es.py \
        --model mlp --dataset telescope --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean
    sleep 2

    echo "CNN / occupancy"
    sbatch --time=00:10:00 --job-name="cnn_occ_${OPT}" run_es.sh python run_es.py \
        --model cnn --dataset occupancy --hidden_size 8 --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 20 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean
    sleep 2

    echo "Transf / occupancy"
    sbatch --time=00:10:00 --job-name="transf_occ_${OPT}" run_es.sh python run_es.py \
        --model transf --dataset occupancy --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 20 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean
    sleep 2

    echo "CNN / air"
    sbatch --time=00:15:00 --job-name="cnn_air_${OPT}" run_es.sh python run_es.py \
        --model cnn --dataset air --hidden_size 5 --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean
    sleep 2

    echo "Transf / air"
    sbatch --time=00:15:00 --job-name="transf_air_${OPT}" run_es.sh python run_es.py \
        --model transf --dataset air --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean
    sleep 2

    # Schedule the same commands using the early stopping criteria with clean gradients
    echo "MLP / Telescope"
    sbatch --job-name="mlp_tele_${OPT}" run_es.sh python run_es.py \
        --model mlp --dataset telescope --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --use_clean_es
    sleep 2

    echo "CNN / occupancy"
    sbatch --time=00:10:00 --job-name="cnn_occ_${OPT}" run_es.sh python run_es.py \
        --model cnn --dataset occupancy --hidden_size 8 --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 20 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --use_clean_es
    sleep 2

    echo "Transf / occupancy"
    sbatch --time=00:10:00 --job-name="transf_occ_${OPT}" run_es.sh python run_es.py \
        --model transf --dataset occupancy --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 20 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --use_clean_es
    sleep 2

    echo "CNN / air"
    sbatch --time=00:15:00 --job-name="cnn_air_${OPT}" run_es.sh python run_es.py \
        --model cnn --dataset air --hidden_size 5 --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --use_clean_es
    sleep 2

    echo "Transf / air"
    sbatch --time=00:15:00 --job-name="transf_air_${OPT}" run_es.sh python run_es.py \
        --model transf --dataset air --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --use_clean_es
    sleep 2

done

echo "------------------------------------------------"
echo "Submission loop complete."