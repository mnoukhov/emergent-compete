from abc import abstractmethod
import random

import gin
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class Policy(object):
    @abstractmethod
    def step(self, state):
        pass

    @abstractmethod
    def update(self, round, prev_state, action, next_state, reward):
        pass


@gin.configurable
class Human(Policy):
    def __init__(self, mode):
        # 0 = sender, 1 = receiver
        self.mode = mode

    def step(self, state):
        obs, _, _ = state

        if self.mode == 0:
            prompt = 'target {}: '.format(obs.item())
        elif self.mode == 1:
            prompt = 'message {}: '.format(obs.item())

        action = int(input(prompt))

        return torch.tensor(action)

    def update(self, *args):
        pass


@gin.configurable
class RuleBasedSender(Policy):
    def __init__(self, n, bias):
        self.bias_dist = Uniform(0, bias)
        self.n = n

    def step(self, state):
        target, _, _ = state
        message = target + self.bias_dist.sample().int()
        return torch.clamp(message, 0, self.n - 1)

    def update(self, *args):
        pass


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
        return self.n**3

    def step(self, state):
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

        self.Q[prev_state_idx, action] = reward

        # self.Q[prev_state_idx, action] = ( (1 - alpha) * self.Q[prev_state_idx, action]
                                          # + alpha * (reward + self.gamma * V))

    def _to_idx(self, state):
        return self.n**2 * state[0] + self.n*state[1] + state[2]


@gin.configurable
class OneShotQNet(NaiveQNet):
    @property
    def states(self):
        return self.n

    def _to_idx(self, state):
        return state[0]


@gin.configurable
class A2C(Policy):
    def __init__(self, n, gamma, alpha, decay, epsilon):
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.epsilon = epsilon
        self.V = torch.nn.Linear(3,1)

    def step(self, state)
        input_ = torch.stack(state, dim=0)
        logits
        logits = self.Q[state_idx]
        return self._epsilon_greedy(logits)

    def _epsilon_greedy(self, logits):
        if random.random() <= self.epsilon:
            return torch.randint(self.states - 1, size=())
        else:
            return torch.argmax(logits)

    def _to_idx(self, state):
        return self.n**2 * state[0] + self.n*state[1] + state[2]


@gin.configurable
class PolicyGradient(Policy):
    def __init__(self, n, gamma, alpha, decay, epsilon):
        pass

    def step(self, state):
        pass

    def update(self, round, prev_state, action, next_state, reward):
        pass
