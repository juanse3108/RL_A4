import glob
import os
import pandas as pd


def summarize_ppo(raw_dir="results/raw"):
    eval_files = sorted(glob.glob(os.path.join(raw_dir, "ppo_eval_seed_*.csv")))

    if not eval_files:
        raise FileNotFoundError("No PPO evaluation files found in results/raw.")

    df = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True)

    summary = (
        df.groupby("step")["eval_return"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)

    print("PPO evaluation files:")
    for f in eval_files:
        print(f"  - {f}")

    print("\nFirst evaluation points:")
    print(summary.head(10).to_string(index=False))

    print("\nLast evaluation points:")
    print(summary.tail(10).to_string(index=False))

    best_row = summary.loc[summary["mean"].idxmax()]
    print("\nBest mean evaluation return:")
    print(best_row.to_string())

    reached = summary[summary["mean"] >= 500.0]
    print("\nFirst step where mean evaluation return reaches 500:")
    if len(reached) > 0:
        print(reached.iloc[0].to_string())
    else:
        print("Mean evaluation return never reached 500.")

    print("\nFinal evaluation point:")
    print(summary.iloc[-1].to_string())

    return summary


if __name__ == "__main__":
    summarize_ppo()
