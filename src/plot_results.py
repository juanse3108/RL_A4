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

    ppo_summary = (
        ppo_df.groupby("step")["eval_return"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ppo_summary["std"] = ppo_summary["std"].fillna(0.0)

    baseline_df = pd.read_csv(baseline_path)

    plt.figure()

    plt.plot(
        ppo_summary["step"],
        ppo_summary["mean"],
        label="PPO mean"
    )
    plt.fill_between(
        ppo_summary["step"],
        ppo_summary["mean"] - ppo_summary["std"],
        ppo_summary["mean"] + ppo_summary["std"],
        alpha=0.2,
        label="PPO ±1 std"
    )

    baseline_summary = (
        baseline_df.groupby("env_step")["Episode_Return_smooth"]
        .mean()
        .reset_index()
        .sort_values("env_step")
    )

    plt.plot(
        baseline_summary["env_step"],
        baseline_summary["Episode_Return_smooth"],
        linestyle="--",
        label="Baseline smooth"
    )

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
