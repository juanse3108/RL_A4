#!/bin/bash
#SBATCH --job-name=rl_a4_ppo
#SBATCH --partition=cpu-skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl_a4_ppo_%j.out
#SBATCH --error=logs/rl_a4_ppo_%j.err

set -e

cd /zfsstore/user/s4567846/projects/rl_projects/a4
mkdir -p logs results/raw results/plots

source /zfsstore/user/s4567846/projects/rl_projects/a3/rl_env/bin/activate

echo "Job started on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Virtual env: $VIRTUAL_ENV"

python -u src/train.py \
  --total-timesteps 1000000 \
  --eval-interval 5000 \
  --eval-episodes 5 \
  --seed 0 \
  --cpu

python -u src/plot_results.py --skip-baseline

echo "Job finished successfully"