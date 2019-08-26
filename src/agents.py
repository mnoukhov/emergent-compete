from enum import Enum

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

mode = Enum('Player', 'SENDER RECVER')


class RelaxedEmbedding(nn.Embedding):
    def forward(self, x):
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            return F.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return torch.matmul(x, self.weight)


class Policy(nn.Module):
    retain_graph = False

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

    def functional_forward(self, state, weights):
        out = F.embedding(state, weights[0], weights[1])
        out = F.relu(out)
        out = F.linear(out, weights[2], weights[3])
        out = F.relu(out)
        out = F.linear(out, weights[4], weights[5])

        return out

    def loss(self, error, *args):
        _, logs = super().loss(error)
        loss = error.mean()

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
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
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
            sample = logits.argmax(dim=1)

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def loss(self, error, logprobs, entropy):
        _, logs = super().loss(error)

        policy_loss = ((error.detach() - self.baseline) * logprobs).mean()
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
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs




