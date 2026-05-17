import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        return logits

    def get_dist(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)

    def act(self, state):
        dist = self.get_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)
