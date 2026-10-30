#!/bin/bash
#SBATCH --job-name=NGC_comparison
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128Gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/ngc-%j-%x-%a.out
#SBATCH --error=logs/ngc-%j-%x-%a.err
#SBATCH --array=0-9

# Seeds to use for the experiment
SEEDS=(42 69 71 12 7 123 456 789 812 947)

CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Add env to Path
export PATH="/env/torch210/bin:$PATH"

# Setup
mkdir -p logs

# Execute the command
srun python compare_ngc.py --dataset air --model cnn --norm_X --hidden_size 20 --use_h --seed $CURRENT_SEED --batch_size 8
