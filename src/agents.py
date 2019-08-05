from copy import deepcopy
from enum import Enum

import gin
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

from src.utils import discount_return

mode = Enum('Player', 'SENDER RECVER')


class Policy(nn.Module):
    def __init__(self, mode, *args, **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, state):
        pass

    def loss(self, rewards, **kwargs):
        mean_reward = rewards.mean().item()
        round_reward = rewards.mean(dim=0).tolist()
        logs = {'reward': mean_reward,
                'round_reward': round_reward}

        return None, logs


@gin.configurable
class UniformBias(Policy):
    def __init__(self, bias, **kwargs):
        super().__init__(**kwargs)
        self.bias = Uniform(0, bias)

    def forward(self, state):
        target = state[:,0]
        output = target + self.bias.sample()
        return output.float()


@gin.configurable
class DeterministicGradient(Policy):
    def __init__(self, input_size, output_size, lr, weight_decay, device,
                 output_range=None, **kwargs):
        super().__init__(**kwargs)
        self.output_range = output_range
        self.policy = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, state):
        action = self.policy(state)

        if self.output_range:
            action = action % self.output_range

        return action

    def loss(self, rewards):
        _, logs = super().loss(rewards)
        loss = - rewards.mean()

        logs['loss'] = loss.item()
        # for sample_in in [0, 15, 30]:
            # tensor_in = torch.tensor(sample_in).unsqueeze(0).float().to(loss.device)
            # logs[str(sample_in)] = self.policy(tensor_in).item()

        return loss, logs


@gin.configurable
class CategoricalPG(Policy):
    def __init__(self, input_size, output_size, lr, weight_decay, gamma, ent_reg, device, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.policy = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr,
                              weight_decay=weight_decay)

        self.ent_reg = ent_reg
        self.gamma = gamma
        self.baseline = 0.
        self.n_update = 0.
        self.entropy = []
        self.log_probs = []

    def forward(self, state):
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        self.entropy.append(dist.entropy())

        if self.training:
            sample = dist.sample()
        else:
            sample = probs.argmax(dim=1)

        self.log_probs.append(dist.log_prob(sample))

        return sample.float().unsqueeze(1)

    def loss(self, rewards, **kwargs):
        _, logs = super().loss(rewards)
        logprobs = torch.stack(self.log_probs, dim=1)
        entropy = torch.stack(self.entropy, dim=1).mean()

        # discount_return(rewards, self.gamma)
        returns = rewards - self.baseline
        loss = (-logprobs * returns).mean() - self.ent_reg * entropy

        logs['loss'] = loss.item()
        for sample_in in [0, 15, 30]:
            tensor_in = torch.zeros(1, self.input_size).to(loss.device)
            tensor_in[0] = sample_in
            greedy_out = torch.argmax(self.policy(tensor_in)).item()
            logs[str(sample_in)] = greedy_out

        if self.training:
            self.n_update += 1.
            self.baseline += (returns.detach().mean().item() - self.baseline) / (self.n_update)

        self.entropy = []
        self.log_probs = []

        return loss, logs




