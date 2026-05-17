import argparse
import os
import glob

import pandas as pd
import matplotlib.pyplot as plt


def plot_ppo_only(raw_dir, plots_dir):
    eval_files = glob.glob(os.path.join(raw_dir, "ppo_eval_seed_*.csv"))

    if not eval_files:
        raise FileNotFoundError("No PPO evaluation CSV files found.")

    df = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True)

    summary = (
        df.groupby("step")["eval_return"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)

    plt.figure()
    plt.plot(summary["step"], summary["mean"], label="PPO mean")
    plt.fill_between(
        summary["step"],
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        alpha=0.2,
        label="±1 std"
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("PPO on CartPole-v1")
    plt.legend()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "ppo_learning_curve.png"), dpi=300)
    plt.close()

def plot_ppo_vs_baseline(raw_dir, plots_dir, baseline_path):
    eval_files = glob.glob(os.path.join(raw_dir, "ppo_eval_seed_*.csv"))

    if not eval_files:
        raise FileNotFoundError("No PPO evaluation CSV files found.")

    ppo_df = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True)
    ppo_summary = ppo_df.groupby("step")["eval_return"].mean().reset_index()

    baseline_df = pd.read_csv(baseline_path)

    plt.figure()
    plt.plot(ppo_summary["step"], ppo_summary["eval_return"], label="PPO")

    # Try to infer common baseline column names.
    step_candidates = ["step", "steps", "env_steps", "environment_steps", "timesteps"]
    return_candidates = ["return", "reward", "eval_return", "mean_return", "episode_return"]

    step_col = next((c for c in step_candidates if c in baseline_df.columns), None)
    return_col = next((c for c in return_candidates if c in baseline_df.columns), None)

    if step_col is None or return_col is None:
        print("Baseline columns found:", list(baseline_df.columns))
        raise ValueError("Could not infer baseline step/return columns.")

    plt.plot(baseline_df[step_col], baseline_df[return_col], label="Baseline")

    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("PPO vs Baseline on CartPole-v1")
    plt.legend()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "ppo_vs_baseline.png"), dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="results/raw")
    parser.add_argument("--plots-dir", type=str, default="results/plots")
    parser.add_argument("--baseline-path", type=str, default="BaselineDataCartPole.csv")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_ppo_only(args.raw_dir, args.plots_dir)

    if not args.skip_baseline:
        plot_ppo_vs_baseline(args.raw_dir, args.plots_dir, args.baseline_path)
