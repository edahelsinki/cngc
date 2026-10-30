#!/bin/bash

# Optimizers to run - supports suffixes like _pc, _blocks, or _pc_blocks
OPTS=("gd" "adam" "muon" "gd_pc_blocks" "sophia" "sophiac")
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

    # 3. Explicit Submission Blocks
    echo "MLP / Income"
    sbatch --job-name="mlp_inc_${OPT}" run_opt_wallclock.sh python run_opt.py \
        --model mlp --dataset income --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --warmup_epochs $WARMUP --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
    sleep 2

    echo "Linear / Income"
    sbatch --job-name="lin_inc_${OPT}" run_opt_wallclock.sh python run_opt.py \
        --model linear --dataset income --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --warmup_epochs $WARMUP --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
    sleep 2

    echo "MLP / Bank"
    sbatch --job-name="mlp_bnk_${OPT}" run_opt_wallclock.sh python run_opt.py \
        --model mlp --dataset bank --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --warmup_epochs $WARMUP --batch_size 128 --n_epochs 120 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
    sleep 2

    echo "Linear / Bank"
    sbatch --job-name="lin_bnk_${OPT}" run_opt_wallclock.sh python run_opt.py \
        --model linear --dataset bank --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --warmup_epochs $WARMUP --batch_size 128 --n_epochs 120 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
    sleep 2

    echo "MLP / Telescope"
    sbatch --job-name="mlp_tele_${OPT}" run_opt_wallclock.sh python run_opt.py \
        --model mlp --dataset telescope --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --warmup_epochs $WARMUP --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
    sleep 2

    echo "Linear / Telescope"
    sbatch --job-name="lin_tele_${OPT}" run_opt_wallclock.sh python run_opt.py \
        --model linear --dataset telescope --optimizer "$BASE_OPT" $EXTRA_FLAGS \
        --warmup_epochs $WARMUP --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
    sleep 2

    if [[ "$OPT" != "muon" ]]; then
        echo "CNN / occupancy"
        sbatch --time=00:10:00 --job-name="cnn_occ_${OPT}" run_opt_wallclock.sh python run_opt.py \
            --model cnn --dataset occupancy --hidden_size 8 --optimizer "$BASE_OPT" $EXTRA_FLAGS \
            --warmup_epochs $WARMUP --batch_size 128 --n_epochs 20 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
        sleep 2

        echo "Transf / occupancy"
        sbatch --time=00:10:00 --job-name="transf_occ_${OPT}" run_opt_wallclock.sh python run_opt.py \
            --model transf --dataset occupancy --optimizer "$BASE_OPT" $EXTRA_FLAGS \
            --warmup_epochs $WARMUP --batch_size 128 --n_epochs 20 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
        sleep 2

        echo "CNN / air"
        sbatch --time=00:15:00 --job-name="cnn_air_${OPT}" run_opt_wallclock.sh python run_opt.py \
            --model cnn --dataset air --hidden_size 5 --optimizer "$BASE_OPT" $EXTRA_FLAGS \
            --warmup_epochs $WARMUP --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
        sleep 2

        echo "Transf / air"
        sbatch --time=00:15:00 --job-name="transf_air_${OPT}" run_opt_wallclock.sh python run_opt.py \
            --model transf --dataset air --optimizer "$BASE_OPT" $EXTRA_FLAGS \
            --warmup_epochs $WARMUP --batch_size 128 --n_epochs 100 --lr $LR --norm_X --noise_mult $MULT --binary_noise --subtract_clean --res_folder_root results_nn_opt_time
        sleep 2
    else
        echo "Skipping $OPT for cnns and transformers."
    fi

done

echo "------------------------------------------------"
echo "Submission loop complete."