#!/bin/bash
#SBATCH --job-name=rl_a4_ppo_all
#SBATCH --partition=cpu-skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl_a4_ppo_all_%j.out
#SBATCH --error=logs/rl_a4_ppo_all_%j.err

set -e

cd /zfsstore/user/s4567846/projects/rl_projects/a4
mkdir -p logs results/raw results/plots

source /zfsstore/user/s4567846/projects/rl_projects/a3/rl_env/bin/activate

echo "Job started on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Virtual env: $VIRTUAL_ENV"

# Optional: keep seed 0 if already finished.
# Remove only if you want a fully fresh 3-seed run.
# rm -f results/raw/ppo_train_seed_*.csv results/raw/ppo_eval_seed_*.csv results/plots/*.png

for SEED in 1 2
do
    echo "--------------------------------"
    echo "Starting PPO seed ${SEED} at $(date)"
    echo "--------------------------------"

    python -u src/train.py \
      --total-timesteps 1000000 \
      --eval-interval 5000 \
      --eval-episodes 5 \
      --seed ${SEED} \
      --cpu

    echo "Finished PPO seed ${SEED} at $(date)"
done

echo "Generating plots from all available seeds..."
python -u src/plot_results.py --skip-baseline

echo "Available raw results:"
ls -lh results/raw

echo "Available plots:"
ls -lh results/plots

echo "Job finished successfully at $(date)"
