import numpy as np
import torch

from utils import compute_discounted_returns


def collect_rollout(env, actor, critic, rollout_steps, device):
    states = []
    actions = []
    rewards = []
    dones = []
    old_log_probs = []
    values = []

    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    episode_returns = []
    episode_lengths = []
    current_return = 0.0
    current_length = 0

    actor.eval()
    critic.eval()

    for _ in range(rollout_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob_tensor = actor.act(state_tensor)
            value_tensor = critic(state_tensor)

        action = action_tensor.item()

        step_out = env.step(action)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_out

        states.append(state)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)
        old_log_probs.append(log_prob_tensor.item())
        values.append(value_tensor.item())

        current_return += reward
        current_length += 1

        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)

            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            current_return = 0.0
            current_length = 0
        else:
            state = next_state

    actor.train()
    critic.train()

    rollout = {
        "states": torch.tensor(np.array(states), dtype=torch.float32, device=device),
        "actions": torch.tensor(actions, dtype=torch.long, device=device),
        "rewards": rewards,
        "dones": dones,
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        "values": torch.tensor(values, dtype=torch.float32, device=device),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }

    return rollout


def compute_rollout_targets(rollout, gamma, normalize_advantages=True):
    returns = compute_discounted_returns(
        rollout["rewards"],
        rollout["dones"],
        gamma,
    ).to(rollout["values"].device)

    advantages = returns - rollout["values"]

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rollout["returns"] = returns
    rollout["advantages"] = advantages

    return rollout


def ppo_update(
    actor,
    critic,
    optimizer,
    rollout,
    clip_epsilon,
    ppo_epochs,
    mini_batch_size,
    value_coef,
):
    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    returns = rollout["returns"]
    advantages = rollout["advantages"]

    n_samples = states.shape[0]

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    update_steps = 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(n_samples, device=states.device)

        for start in range(0, n_samples, mini_batch_size):
            mb_idx = indices[start:start + mini_batch_size]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            dist = actor.get_dist(mb_states)
            new_log_probs = dist.log_prob(mb_actions)
            new_values = critic(mb_states)

            ratios = torch.exp(new_log_probs - mb_old_log_probs)

            unclipped = ratios * mb_advantages
            clipped = torch.clamp(
                ratios,
                1.0 - clip_epsilon,
                1.0 + clip_epsilon,
            ) * mb_advantages

            actor_loss = -torch.mean(torch.min(unclipped, clipped))
            critic_loss = torch.mean((new_values - mb_returns) ** 2)

            loss = actor_loss + value_coef * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            update_steps += 1

    return {
        "actor_loss": total_actor_loss / max(update_steps, 1),
        "critic_loss": total_critic_loss / max(update_steps, 1),
    }
