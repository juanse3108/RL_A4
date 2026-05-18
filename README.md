# Proximal Policy Optimization on CartPole

This repository contains an implementation of Proximal Policy Optimization (PPO-Clipped) for the `CartPole-v1` environment. The project was developed as a reinforcement learning assignment to study how PPO extends the advantage-based actor-critic methods used in previous assignments.

The implementation focuses on a simple PPO variant with:

- actor-critic architecture
- rollout collection
- discounted returns
- advantage estimation
- advantage normalization
- PPO clipped actor loss
- critic mean squared error loss
- mini-batch updates
- multiple PPO epochs per rollout
- evaluation across multiple random seeds

## Repository Structure

```text
.
├── BaselineDataCartPole.csv      # Baseline data from previous assignments
├── README.md
├── requirements.txt
├── results/
│   ├── raw/                      # Raw training and evaluation CSV files
│   └── plots/                    # Generated learning curve plots
└── src/
    ├── models.py                 # Actor and critic networks
    ├── ppo.py                    # PPO update logic and rollout utilities
    ├── train.py                  # Main training script
    ├── plot_results.py           # Plotting script
    └── summarize_results.py      # Evaluation summary script
````
## How to Run

Install the required packages:
```
pip install -r requirements.txt
```
Run a short test:
```
python src/train.py --total-timesteps 10000 --eval-interval 5000 --eval-episodes 2 --cpu
```
Run one full PPO experiment:
```
python -u src/train.py --seed 0 --cpu
```
To reproduce the three-seed experiment, run:
```
python -u src/train.py --seed 0 --cpu
python -u src/train.py --seed 1 --cpu
python -u src/train.py --seed 2 --cpu
```
Generate the plot:
```
python src/plot_results.py
```
Summarize the results:
```
python src/summarize_results.py
````

## Outputs

Raw CSV files are saved in:
```
results/raw/
```
Plots are saved in:
```
results/plots/
```
The final experiment was run with three seeds. PPO reached the maximum mean evaluation return of 500 after 26,624 environment steps and ended with all seeds achieving the maximum score.

## References

* OpenAI Spinning Up: Proximal Policy Optimization
https://spinningup.openai.com/en/latest/algorithms/ppo.html
* https://github.com/juanse3108/RL_A3
* Public PPO CartPole references used for comparison/checking:
    * https://github.com/bnelo12/PPO-Implemnetation
    * https://github.com/RPC2/PPO