#!/bin/bash
#SBATCH --job-name=Noise_test
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/noise-%j-%x-%a.out
#SBATCH --error=logs/noise-%j-%x-%a.err
#SBATCH --array=0-9

# Seeds to use for the experiment
SEEDS=(42 69 71 12 7 123 456 789 812 947)

CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Add env to Path
export PATH="/env/torch210/bin:$PATH"
mkdir -p logs

# Execute the command
srun python run_binary_vs_gaussian.py --seed $CURRENT_SEED