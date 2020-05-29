import math
from enum import Enum

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Normal, MultivariateNormal

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

    def __init__(self, input_size, output_size, hidden_size,
                 lr, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.policy = RelaxedEmbedding(input_size, output_size)
        elif self.num_layers == 2:
            self.policy = nn.Sequential(
                RelaxedEmbedding(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size))
        else:
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
        if self.num_layers == 1:
            out = relaxedembedding(x, weights[0])
        elif self.num_layers == 2:
            out = relaxedembedding(x, weights[0])
            out = F.relu(out)
            out = F.linear(out, weights[1], weights[2])
        else:
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
                 lr, ent_reg, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.policy = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LogSoftmax(dim=1))
        elif self.num_layers == 2:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.LogSoftmax(dim=1))
        else:
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

    def functional_forward(self, x, weights):
        if self.num_layers == 1:
            out = F.linear(x, weights[0], weights[1])
            logits = F.log_softmax(out, dim=1)
        elif self.num_layers == 2:
            out = F.linear(x, weights[0], weights[1])
            out = F.relu(out)
            out = F.linear(out, weights[2], weights[3])
            logits = F.log_softmax(out, dim=1)
        else:
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

    def forward_dist(self, state):
        logits = self.policy(state)
        return Categorical(logits=logits)

    def loss(self, error, logprobs, entropy):
        _, logs = super().loss(error)

        policy_loss = ((error.detach() - self.baseline) * logprobs).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()
        logs['entropy'] = entropy.mean().item()

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs


@gin.configurable
class Gaussian(Policy):
    def __init__(self, input_size, output_size, hidden_size,
                 lr, ent_reg, dim=1, num_layers=2, min_var=1e-7,
                 retain_graph=False, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        if self.num_layers == 2:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU())
        elif self.num_layers == 3:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU())
        else:
            self.policy = None

        self.mean = nn.Linear(hidden_size, dim)
        self.var = nn.Sequential(
            nn.Linear(hidden_size, dim),
            nn.ReLU())

        self.ent_reg = ent_reg
        self.lr = lr
        self.dim = dim
        self.min_var = min_var
        self.retain_graph = retain_graph
        self.baseline = 0.
        self.n_update = 0.

    def forward(self, state):
        device = state.device
        logits = self.policy(state)
        mean = self.mean(logits)
        var = self.var(logits) + self.min_var
        if self.dim > 1:
            covar = var.unsqueeze(1) * torch.eye(self.dim).to(device)
            dist = MultivariateNormal(mean, covar)
        else:
            dist = Normal(mean, var)

        entropy = dist.entropy()

        if self.training:
            sample = dist.rsample()
        else:
            sample = mean

        logprobs = dist.log_prob(sample)

        if self.dim > 1:
            sample = sample.mean(dim=1, keepdim=True)

        return sample, logprobs, entropy

    def functional_forward(self, x, weights):
        if self.num_layers == 2:
            out = F.linear(x, weights[0], weights[1])
            out = F.relu(out)
            mean = F.linear(out, weights[2], weights[3])
            var = F.relu(F.linear(out, weights[4], weights[5])) + 1e-7
        elif self.num_layers == 3:
            out = F.linear(x, weights[0], weights[1])
            out = F.relu(out)
            out = F.linear(out, weights[2], weights[3])
            out = F.relu(out)
            mean = F.linear(out, weights[4], weights[5])
            var = F.relu(F.linear(out, weights[6], weights[7])) + 1e-7
        else:
            out = x
            mean = F.linear(out, weights[0], weights[1])
            var = F.relu(F.linear(out, weights[2], weights[3])) + 1e-7

        dist = Normal(mean, var)
        entropy = dist.entropy()

        if self.training:
            sample = dist.rsample()
        else:
            sample = mean

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def forward_dist(self, state):
        logits = self.policy(state)
        mean = self.mean(logits)
        var = self.var(logits) + self.min_var

        iso_mean = torch.mean(mean, dim=1, keepdim=True)
        iso_var = torch.mean(var, dim=1, keepdim=True)
        def cdf(value):
            return 0.5 * (1 + torch.erf((value - iso_mean) * iso_var.reciprocal() / math.sqrt(2)))

        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(iso_var)

        return iso_mean, iso_var, cdf, entropy

    def loss(self, error, logprobs, entropy):
        _, logs = super().loss(error)

        if self.retain_graph:
            loss = error.mean()
        else:
            policy_loss = ((error.detach() - self.baseline) * logprobs).mean()
            entropy_loss = -entropy.mean() * self.ent_reg
            loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()
        logs['entropy'] = entropy.mean().item()

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs
