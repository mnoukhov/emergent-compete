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
            nn.ReLU(),
            nn.Linear(5*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))
        self.input_logit = nn.Embedding(input_size, hidden_size)
        self.output_logit = nn.Linear(output_size, hidden_size)
        self.error_logit = nn.Linear(1, hidden_size)
        self.first_logit = nn.Embedding(2, hidden_size)

        self.lr = lr

    def forward(self, input_, prev_input, prev_output, prev_error, first_round):
        input_logit = self.input_logit(input_)
        prev_input_logit = self.input_logit(prev_input)
        prev_output_logit = self.output_logit(prev_output)
        prev_error_logit = self.error_logit(prev_error)
        first_round_logit = self.first_logit(first_round)

        state = torch.cat((input_logit,
                           prev_input_logit,
                           prev_output_logit,
                           prev_error_logit,
                           first_round_logit), 1)
        action = self.policy(state)

        return action, torch.tensor(0.), torch.tensor(0.)

    def functional_forward(self, input_, prev_input, prev_output, prev_error, first_round, weights):
        input_logit = F.embedding(input_, weights[6])
        prev_input_logit = F.embedding(prev_input, weights[6])
        prev_output_logit = F.linear(prev_output, weights[7], weights[8])
        prev_error_logit = F.linear(prev_error, weights[9], weights[10])
        first_round_logit = F.embedding(first_round, weights[11])

        state = torch.cat((input_logit,
                           prev_input_logit,
                           prev_output_logit,
                           prev_error_logit,
                           first_round_logit), 1)

        out = F.relu(state)
        out = F.linear(out, weights[0], weights[1])
        out = F.relu(out)
        out = F.linear(out, weights[2], weights[3])
        out = F.relu(out)
        out = F.linear(out, weights[4], weights[5])

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
            nn.ReLU(),
            nn.Linear(5*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1))
        self.input_logit = nn.Linear(input_size, hidden_size)
        self.output_logit = nn.Embedding(output_size, hidden_size)
        self.error_logit = nn.Linear(1, hidden_size)
        self.first_logit = nn.Embedding(2, hidden_size)

        self.gamma = gamma
        self.ent_reg = ent_reg
        self.lr = lr
        self.baseline = 0.
        self.n_update = 0.

    def forward(self, input_, prev_input, prev_output, prev_error, first_round):
        input_logit = self.input_logit(input_)
        prev_input_logit = self.input_logit(prev_input)
        prev_output_logit = self.output_logit(prev_output)
        prev_error_logit = self.error_logit(prev_error)
        first_round_logit = self.first_logit(first_round)

        state = torch.cat((input_logit,
                           prev_input_logit,
                           prev_output_logit,
                           prev_error_logit,
                           first_round_logit), 1)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if self.training:
            sample = dist.sample()
        else:
            sample = logits.argmax(dim=1)

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def functional_forward(self, input_, prev_input, prev_output, prev_error, first_round, weights):
        input_logit = F.linear(input_, weights[6], weights[7])
        prev_input_logit = F.linear(prev_input, weights[6], weights[7])
        prev_output_logit = F.embedding(prev_output, weights[8])
        prev_error_logit = F.linear(prev_error, weights[9], weights[10])
        first_round_logit = F.embedding(first_round, weights[11])

        state = torch.cat((input_logit,
                           prev_input_logit,
                           prev_output_logit,
                           prev_error_logit,
                           first_round_logit), 1)

        out = F.relu(state)
        out = F.linear(out, weights[0], weights[1])
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

        E = 0
        discount_errors = []
        for round_ in reversed(range(errors.size(0))):
            E = errors[round_] + self.gamma * E
            discount_errors.insert(0, E)

        discount_errors = torch.stack(discount_errors)

        policy_loss = ((discount_errors.detach() - self.baseline) * logprobs).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()

        if self.training:
            self.n_update += 1.
            self.baseline += (discount_errors.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs




