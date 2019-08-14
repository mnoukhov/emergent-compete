from copy import deepcopy
from enum import Enum

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

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
class Deterministic(Policy):
    def __init__(self, input_size, output_size, hidden_size,
                 lr, output_range=None, **kwargs):
        super().__init__(**kwargs)
        self.output_range = output_range
        self.policy = nn.Sequential(
            nn.Embedding(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size))
        self.lr = lr

    def forward(self, state):
        action = self.policy(state)

        # if self.output_range:
            # action = action % self.output_range

        return action

    def functional_forward(self, state, weights):
        out = F.embedding(state, weights[0], weights[1])
        out = F.relu(out)
        out = F.linear(out, weights[2], weights[3])
        out = F.relu(out)
        out = F.linear(out, weights[4], weights[5])

        # if self.output_range:
            # out = out % self.output_range

        return out

    def loss(self, raw_loss):
        _, logs = super().loss(-raw_loss)
        loss = raw_loss.mean()

        logs['loss'] = loss.item()
        # for sample_in in [0, 15, 30]:
            # tensor_in = torch.tensor(sample_in).unsqueeze(0).float().to(loss.device)
            # logs[str(sample_in)] = self.policy(tensor_in).item()

        return loss, logs


@gin.configurable
class Reinforce(Policy):
    def __init__(self, input_size, output_size, hidden_size,
                 lr, ent_reg, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1))

        self.ent_reg = ent_reg
        self.lr = lr
        self.baseline = 0.
        self.n_update = 0.

    def forward(self, state):
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if self.training:
            sample = dist.sample()
        else:
            sample = probs.argmax(dim=1)

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def loss(self, raw_loss, logprobs, entropy):
        _, logs = super().loss(-raw_loss)

        policy_loss = ((raw_loss.detach() - self.baseline) * logprobs).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()
        # for sample_in in [0, 15, 30]:
            # tensor_in = torch.zeros(1, self.input_size).to(loss.device)
            # tensor_in[0] = sample_in
            # greedy_out = torch.argmax(self.policy(tensor_in)).item()
            # logs[str(sample_in)] = greedy_out

        if self.training:
            self.n_update += 1.
            self.baseline += (raw_loss.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs




