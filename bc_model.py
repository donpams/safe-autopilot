import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """Small MLP mapping the 8-dim observation to the 2-dim action.
    Tanh output matches the environment's action_space bound of [-1, 1].
    """
    def __init__(self, obs_dim=8, act_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
