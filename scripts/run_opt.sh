#!/bin/bash
#SBATCH --job-name=Optimization_PC
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/opt_pc-%j-%x-%a.out
#SBATCH --error=logs/opt_pc-%j-%x-%a.err
#SBATCH --array=0-9

# Seeds to use for the experiment
SEEDS=(42 69 71 12 7 123 456 789 812 947)

CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Add env to Path
export PATH="/env/torch210/bin:$PATH"
mkdir -p logs

FULL_CMD="$@ --seed $CURRENT_SEED"
echo "Running Command: $FULL_CMD"

# Execute the command
eval $FULL_CMD