from abc import abstractmethod
import random

import gin
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class Policy(object):
    def __init__(self):
        self.rewards = []
        self.logs = []

    def reset(self):
        del self.rewards[:]

    @abstractmethod
    def action(self, state):
        pass

    def update(self):
        pass

    def avg_reward(self):
        rewards = torch.stack(self.rewards)
        return rewards.mean().item()


@gin.configurable
class Human(Policy):
    def __init__(self, mode):
        super().__init__()
        # 0 = sender, 1 = receiver
        self.mode = mode

    def action(self, state):
        obs, _, _ = state

        if self.mode == 0:
            prompt = 'target {}: '.format(obs.item())
        elif self.mode == 1:
            prompt = 'message {}: '.format(obs.item())

        action = int(input(prompt))

        return torch.tensor(action)


@gin.configurable
class RuleBasedSender(Policy):
    def __init__(self, n, bias):
        super().__init__()
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
    def __init__(self, n, lr):
        super().__init__()
        self.n = n
        self.policy = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid()
        )
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def action(self, state):
        input_ = state[0]
        action = 1 + self.policy(input_) * (self.n - 1)

        return action

    def update(self):
        loss = -torch.mean(torch.stack(self.rewards))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


@gin.configurable
class PolicyGradient(Policy):
    def __init__(self, n, lr, gamma):
        super().__init__()
        self.n = n
        self.gamma = gamma
        self.policy = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid()
        )
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        # self.optimizer = SGD(self.policy.parameters(), lr=lr)
        self.log_probs = []

    def reset(self):
        del self.rewards[:]
        del self.log_probs[:]

    def action(self, state):
        # input_ = torch.stack(state)
        input_ = state[0].reshape(1,1)
        out = self.policy(input_) * self.n

        mean = out
        std = 1
        dist = Normal(mean, std)

        sample = dist.sample()[0]
        self.log_probs.append(dist.log_prob(sample))

        action = torch.clamp(sample, 0, self.n)
        return action

    def update(self):
        # discount future
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(self.log_probs)
        loss = - torch.sum(torch.mul(returns, log_probs))

        self.optimizer.zero_grad()
        __import__('pdb').set_trace()
        loss.backward()
        self.optimizer.step()

# @gin.configurable
# class A2C(Policy):
    # def __init__(self, n, gamma, alpha, decay, epsilon):
        # self.n = n
        # self.gamma = gamma
        # self.alpha = alpha
        # self.decay = decay
        # self.epsilon = epsilon
        # self.V = torch.nn.Linear(3,1)

    # def step(self, state):
        # input_ = torch.stack(state, dim=0)
        # logits
        # logits = self.Q[state_idx]
        # return self._epsilon_greedy(logits)

    # def _epsilon_greedy(self, logits):
        # if random.random() <= self.epsilon:
            # return torch.randint(self.states - 1, size=())
        # else:
            # return torch.argmax(logits)

    # def _to_idx(self, state):
        # return self.n**2 * state[0] + self.n*state[1] + state[2]
