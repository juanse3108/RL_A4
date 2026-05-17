import csv
import os
import random

import numpy as np
import torch


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)


def compute_discounted_returns(rewards, dones, gamma):
    returns = []
    running_return = 0.0

    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            running_return = 0.0
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    return torch.tensor(returns, dtype=torch.float32)


def evaluate_policy(env, actor, device, eval_episodes=5):
    actor.eval()
    returns = []

    with torch.no_grad():
        for _ in range(eval_episodes):
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            done = False
            episode_return = 0.0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor(state_tensor)
                action = torch.argmax(logits, dim=-1).item()

                step_out = env.step(action)
                if len(step_out) == 5:
                    state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_out

                episode_return += reward

            returns.append(episode_return)

    actor.train()
    return float(np.mean(returns))


def append_row(path, fieldnames, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
