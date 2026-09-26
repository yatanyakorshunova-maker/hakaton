from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        input_dim = obs_dim + action_dim + 3

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.memory = nn.GRUCell(hidden_size, hidden_size)

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def initial(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.hidden_size,
            device=device,
        )

    def _features(
        self,
        observation,
        previous_action,
        previous_reward,
        previous_done,
        trial_progress,
    ):
        action = F.one_hot(
            previous_action.long(),
            self.action_dim,
        ).float()

        reward = torch.tanh(
            previous_reward.unsqueeze(-1) / 10.0
        )

        done = previous_done.float().unsqueeze(-1)

        progress = trial_progress.float().unsqueeze(-1)

        return torch.cat(
            (
                observation,
                action,
                reward,
                done,
                progress,
            ),
            dim=-1,
        )

    def step(
        self,
        observation,
        previous_action,
        previous_reward,
        previous_done,
        trial_progress,
        trial_start,
        memory,
    ):
        reset_mask = trial_start.float().unsqueeze(-1)
        memory = memory * (1.0 - reset_mask)

        features = self._features(
            observation,
            previous_action,
            previous_reward,
            previous_done,
            trial_progress,
        )

        encoded = self.encoder(features)

        memory = self.memory(
            encoded,
            memory,
        )

        logits = self.actor(memory)
        value = self.critic(memory).squeeze(-1)

        return logits, value, memory

    def sequence(
        self,
        observation,
        previous_action,
        previous_reward,
        previous_done,
        trial_progress,
        trial_start,
        memory,
    ):
        logits = []
        values = []

        for i in range(len(observation)):
            current_logits, current_value, memory = self.step(
                observation[i],
                previous_action[i],
                previous_reward[i],
                previous_done[i],
                trial_progress[i],
                trial_start[i],
                memory,
            )

            logits.append(current_logits)
            values.append(current_value)

        return (
            torch.stack(logits),
            torch.stack(values),
        )
