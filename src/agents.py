from abc import abstractmethod
import random

import gin
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class Policy(object):
    def __init__(self, mode, *args, **kwargs):
        self.mode = mode
        self.rewards = []
        self.logs = []
        self.round_logs = []

    @property
    def avg_grad(self):
        return 0.0

    @property
    def avg_reward(self):
        return torch.cat(self.rewards).mean().item()

    @property
    def avg_loss(self):
        return 0.0

    def reset(self):
        del self.rewards[:]

    def log_reward(self):
        rewards = torch.cat(self.rewards, dim=1)
        round_avg = rewards.mean(dim=0).tolist()
        avg = rewards.mean().item()
        self.logs.append(avg)
        self.round_logs.append(round_avg)

    @abstractmethod
    def action(self, state):
        pass

    def update(self, *args, **kwargs):
        pass


@gin.configurable
class Human(Policy):
    def action(self, state):
        obs, _, _ = state

        if self.mode == 0:
            prompt = 'target {}: '.format(obs.item())
        elif self.mode == 1:
            prompt = 'message {}: '.format(obs.item())

        action = int(input(prompt))

        return torch.tensor(action)


@gin.configurable
class NoComm(Policy):
    def action(self, state):
        return torch.ones_like(state[0])


@gin.configurable
class UniformBias(Policy):
    def __init__(self, n, bias, **kwargs):
        super().__init__(**kwargs)
        self.bias = Uniform(0, bias)
        self.n = n

    def action(self, state):
        target, _, _ = state
        message = target + self.bias.sample()
        return torch.clamp(message, 1, self.n)


@gin.configurable
class NaiveQNet(Policy):
    """Q Learning with epsilon-greedy

    state = target * prev_message * prev_guess
    """

    def __init__(self, n, gamma, alpha, decay, epsilon):
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.epsilon = epsilon
        self.Q = torch.rand(self.states, self.n)

    @property
    def states(self):
        return self.n**3 + 1

    def action(self, state):
        state_idx = self._to_idx(state)
        logits = self.Q[state_idx]
        return self._epsilon_greedy(logits)

    def _epsilon_greedy(self, logits):
        if random.random() <= self.epsilon:
            return torch.randint(self.n - 1, size=())
        else:
            return torch.argmax(logits)

    def update(self, round, prev_state, action, next_state, reward):
        next_state_idx = self._to_idx(next_state)
        V = torch.max(self.Q[next_state_idx])

        # alpha = 1 / (1 / self.alpha + round * self.decay)
        alpha = self.alpha
        prev_state_idx = self._to_idx(prev_state)

        self.Q[prev_state_idx, action] = ( (1 - alpha) * self.Q[prev_state_idx, action]
                                          + alpha * (reward + self.gamma * V))

    def _to_idx(self, state):
        if any(s == -1 for s in state):
            return 0
        else:
            return self.n**2 * state[0] + self.n*state[1] + state[2] + 1


@gin.configurable
class OneShotQNet(NaiveQNet):
    @property
    def states(self):
        return self.n + 1

    def _to_idx(self, state):
        if any(s == -1 for s in state):
            return 0
        else:
            return state[0] + 1


@gin.configurable
class DeterministicGradient(Policy):
    def __init__(self, n, lr, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.policy = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def action(self, state):
        input_ = torch.cat(state, dim=1)
        action = 1 + self.policy(input_) * (self.n - 1)

        return action

    def update(self):
        retain_graph = (self.mode == 0)
        loss = -torch.mean(torch.cat(self.rewards, dim=1))

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()


@gin.configurable
class PolicyGradient(Policy):
    def __init__(self, n, lr, gamma, std, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.gamma = gamma
        self.policy = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid())
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.std = std
        self.log_probs = []
        self.grads = []
        self.losses = []

    @property
    def avg_grad(self):
        return torch.stack(self.grads).abs().mean().item()

    @property
    def avg_loss(self):
        return torch.stack(self.losses).mean().item()

    def reset(self):
        super().reset()
        del self.log_probs[:]
        del self.grads[:]
        del self.losses[:]

    def action(self, state):
        # input_ = torch.cat(state, dim=1)
        input_ = state[0]
        out = self.policy(input_) * self.n

        mean = out
        dist = Normal(mean, self.std)
        sample = dist.sample()
        self.log_probs.append(dist.log_prob(sample))

        action = sample.round() % self.n
        return action

    def update(self):
        # discount future
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.cat(returns, dim=1)
        log_probs = torch.cat(self.log_probs, dim=1)
        loss = - torch.mul(returns, log_probs).mean(dim=1).sum()
        self.losses.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.grads.append(torch.stack([x.grad.mean() for x in self.policy.parameters()]).abs().mean().detach())
        self.optimizer.step()


@gin.configurable
class A2C(Policy):
    def __init__(self, n, gamma, alpha, decay, epsilon):
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.epsilon = epsilon
        self.policy = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid())
        self.V = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid())
        self.Q = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid())

    def step(self, state):
        input_ = torch.stack(state, dim=0)
        logits
        logits = self.Q[state_idx]
        return self._epsilon_greedy(logits)
