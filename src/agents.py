from enum import Enum

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

mode = Enum('Player', 'SENDER RECVER')

def relaxedembedding(x, weight, *args):
    if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
        return F.embedding(x, weight, *args)
    else:
        return torch.matmul(x, weight)


class RelaxedEmbedding(nn.Embedding):
    def forward(self, x):
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            return F.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return torch.matmul(x, self.weight)


class Policy(nn.Module):
    retain_graph = False
    lola = False

    def __init__(self, mode, *args, **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, state):
        pass

    def loss(self, error, **kwargs):
        logs = {'error': error.mean().item()}

        return None, logs


@gin.configurable
class Deterministic(Policy):
    retain_graph = True

    def __init__(self, input_size, output_size, hidden_size, lr, **kwargs):
        super().__init__(**kwargs)
        self.policy = nn.Sequential(
            RelaxedEmbedding(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))
        self.lr = lr

    def forward(self, state):
        action = self.policy(state)

        return action, torch.tensor(0.), torch.tensor(0.)

    def functional_forward(self, x, weights):
        out = relaxedembedding(x, weights[0])
        out = F.relu(out)
        out = F.linear(out, weights[1], weights[2])
        out = F.relu(out)
        out = F.linear(out, weights[3], weights[4])

        return out, torch.tensor(0.), torch.tensor(0.)

    def loss(self, error, *args):
        _, logs = super().loss(error)
        loss = error.mean()

        logs['loss'] = loss.item()

        return loss, logs


@gin.configurable
class Reinforce(Policy):
    def __init__(self, input_size, output_size, hidden_size,
                 lr, gamma, ent_reg, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1))

        self.gamma = gamma
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
            sample = logits.argmax(dim=1)

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def functional_forward(self, x, weights):
        out = F.linear(x, weights[0], weights[1])
        out = F.relu(out)
        out = F.linear(out, weights[2], weights[3])
        out = F.relu(out)
        out = F.linear(out, weights[4], weights[5])
        logits = F.log_softmax(out, dim=1)

        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if self.training:
            sample = dist.sample()
        else:
            sample = logits.argmax(dim=1)

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def loss(self, errors, logprobs, entropy):
        _, logs = super().loss(errors)

        # LOLA proposes this weird discounting
        discounts = torch.tensor([self.gamma**t for t in range(error.size(0))])
        discount_errors = discounts * errors

        policy_loss = ((discount_errors.detach() - self.baseline) * logprobs).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs




