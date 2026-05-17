import argparse
import os

import gymnasium as gym
import torch

from networks import Actor, Critic
from ppo import collect_rollout, compute_rollout_targets, ppo_update
from utils import append_row, evaluate_policy, set_seed


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    set_seed(args.seed, env)
    set_seed(args.seed + 1000, eval_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim, args.hidden_size).to(device)
    critic = Critic(state_dim, args.hidden_size).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.learning_rate,
    )

    os.makedirs(args.raw_dir, exist_ok=True)

    train_csv = os.path.join(args.raw_dir, f"ppo_train_seed_{args.seed}.csv")
    eval_csv = os.path.join(args.raw_dir, f"ppo_eval_seed_{args.seed}.csv")

    train_fields = ["step", "episode", "episode_return", "episode_length", "seed"]
    eval_fields = ["step", "eval_return", "seed"]

    total_steps = 0
    episode_count = 0
    next_eval_step = args.eval_interval

    while total_steps < args.total_timesteps:
        rollout_steps = min(args.rollout_steps, args.total_timesteps - total_steps)

        rollout = collect_rollout(
            env=env,
            actor=actor,
            critic=critic,
            rollout_steps=rollout_steps,
            device=device,
        )

        total_steps += rollout_steps

        for ep_return, ep_length in zip(
            rollout["episode_returns"],
            rollout["episode_lengths"],
        ):
            episode_count += 1
            append_row(
                train_csv,
                train_fields,
                {
                    "step": total_steps,
                    "episode": episode_count,
                    "episode_return": ep_return,
                    "episode_length": ep_length,
                    "seed": args.seed,
                },
            )

        rollout = compute_rollout_targets(
            rollout,
            gamma=args.gamma,
            normalize_advantages=True,
        )

        losses = ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            rollout=rollout,
            clip_epsilon=args.clip_epsilon,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            value_coef=args.value_coef,
        )

        if total_steps >= next_eval_step:
            eval_return = evaluate_policy(
                eval_env,
                actor,
                device,
                eval_episodes=args.eval_episodes,
            )

            append_row(
                eval_csv,
                eval_fields,
                {
                    "step": total_steps,
                    "eval_return": eval_return,
                    "seed": args.seed,
                },
            )

            print(
                f"step={total_steps} "
                f"episodes={episode_count} "
                f"eval_return={eval_return:.2f} "
                f"actor_loss={losses['actor_loss']:.4f} "
                f"critic_loss={losses['critic_loss']:.4f}"
            )

            next_eval_step += args.eval_interval

    env.close()
    eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=str, default="results/raw")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
