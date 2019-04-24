from abc import abstractmethod
import random

import gin
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam, SGD
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from utils import discount_return


class Policy(nn.Module):
    def __init__(self, mode, gamma=1.0):
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        self.rewards = []
        self.logger = {
            'ep_reward': [],
            'round_reward': [],
        }

    def last(self, metric):
        values = self.logger.get(metric, None)
        if values:
            return values[-1]
        else:
            return 0.

    def reset(self):
        del self.rewards[:]

    def action(self, state):
        pass

    def update(self, *args, **kwargs):
        rewards = torch.cat(self.rewards, dim=1)
        self.logger['ep_reward'].append(rewards.mean().item())
        self.logger['round_reward'].append(rewards.mean(dim=0).tolist())


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
    def __init__(self, num_actions, bias, **kwargs):
        super().__init__(**kwargs)
        self.bias = Uniform(0, bias)
        self.num_actions = num_actions

    def action(self, state):
        target = state[0]
        message = target + self.bias.sample()
        return message % self.num_actions


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
    def __init__(self, mode, num_actions, lr):
        super().__init__(mode)
        self.num_actions = num_actions
        self.policy = nn.Linear(3,1)
        init.uniform_(self.policy.weight, 1/9, 1/3)
        init.uniform_(self.policy.bias, 1/9, 1/3)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        self.logger.update({
            'loss': [],
            '20': [],
            'preact grad': [],
            'act grad': []
        })
        self.grad = []
        self.act_grad = []
        self.preact_grad = []
        self.policy.weight.register_hook(self.grad.append)

    def reset(self):
        super().reset()
        del self.grad[:]
        del self.act_grad[:]
        del self.preact_grad[:]

    def action(self, state):
        input_ = torch.cat(state, dim=1)
        action = self.policy(input_)
        action.register_hook(self.preact_grad.append)
        clamped = torch.clamp(action, 0, self.num_actions)
        clamped.register_hook(self.act_grad.append)

        return clamped

    def update(self):
        super().update()
        loss = -torch.stack(self.rewards).mean()
        self.logger['loss'].append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger['act grad'].append(torch.stack(self.act_grad).norm(2).item())
        self.logger['preact grad'].append(torch.stack(self.preact_grad).norm(2).item())
        self.logger['20'].append(self.policy(torch.tensor([20., 0., 0.]))[0].item())
        # self.logger['grad'].append(torch.stack(self.grad).norm(2).item())


@gin.configurable
class PolicyGradient(Policy):
    def __init__(self, num_actions, lr, gamma, ent_reg, min_std, mode):
        super().__init__(mode)
        self.num_actions = num_actions
        self.gamma = gamma
        self.min_std = torch.tensor(min_std)
        self.policy = nn.Linear(3,2)
        init.uniform_(self.policy.weight, 1/9, 1)
        init.uniform_(self.policy.bias, 1/9, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.ent_reg = ent_reg

        self.logger.update({
            'loss': [],
            'grad': [],
            '20': [],
            'entropy': [],
        })
        self.log_probs = []
        self.weight_grad = []
        self.entropy = 0
        self.policy.weight.register_hook(self.weight_grad.append)


    def reset(self):
        super().reset()
        del self.log_probs[:]
        del self.weight_grad[:]
        self.entropy = 0

    def action(self, state):
        state = torch.cat(state, dim=1)
        out = self.policy(state)
        mean, std = out.chunk(2, dim=1)

        mean = torch.clamp(mean, 0, self.num_actions)
        std = torch.max(std, self.min_std)
        dist = Normal(mean, std)
        sample = dist.sample()

        self.entropy += dist.entropy()
        self.log_probs.append(dist.log_prob(sample))

        return sample.round() % self.num_actions

    def update(self):
        super().update()
        out20 = self.policy(torch.tensor([20., 0., 0.]))[0]

        returns = torch.cat(discount_return(self.rewards, self.gamma), dim=1)
        logprobs = torch.cat(self.log_probs, dim=1)
        entropy = torch.mean(self.entropy)
        loss = -(returns * logprobs).mean() - self.ent_reg * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger['loss'].append(loss.item())
        self.logger['grad'].append(torch.stack(self.weight_grad).norm(2).item())
        self.logger['entropy'].append(entropy.item())
        self.logger['20'].append(out20.item())


@gin.configurable
class A2C(Policy):
    def __init__(self, num_actions, gamma, lr, min_std, ent_reg, mode):
        super().__init__(mode)
        self.num_actions = num_actions
        self.gamma = gamma
        self.min_std = torch.tensor(min_std)
        self.ent_reg = ent_reg

        self.actor = nn.Linear(3, 2)
        self.critic = nn.Linear(3, 1)
        init.uniform_(self.actor.weight, 1/9, 1/3)
        init.uniform_(self.actor.bias, 1/9, 1/3)
        init.uniform_(self.critic.weight, 1/9, 1/3)
        init.uniform_(self.critic.bias, 1/9, 1/3)

        self.optimizer = Adam(self.parameters(), lr=lr)

        self.values = []
        self.log_probs = []
        self.entropy = 0
        self.logger.update({
            'entropy': [],
            'loss': [],
            '20': []
        })

    def reset(self):
        super().reset()
        del self.values[:]
        del self.log_probs[:]
        self.entropy = 0

    def action(self, state):
        # state = state[0]
        state = torch.cat(state, dim=1)
        mean, std = self.actor(state).chunk(2, dim=1)
        mean = torch.clamp(mean, 0, self.num_actions)
        std = torch.max(std, self.min_std)
        dist = Normal(mean, std)
        sample = dist.sample()

        value = self.critic(state)

        self.entropy += dist.entropy()
        self.log_probs.append(dist.log_prob(sample))
        self.values.append(value)

        return sample.round() % self.num_actions


    def update(self):
        super().update()
        out20 = self.actor(torch.tensor([20., 0., 0.]))[0]

        returns = torch.cat(discount_return(self.rewards, self.gamma), dim=1)
        logprobs = torch.cat(self.log_probs, dim=1)
        values = torch.cat(self.values, dim=1)
        entropy = self.entropy.mean()

        adv = returns - values
        actor_loss = (-logprobs * adv).mean()
        critic_loss = (0.5 * adv**2).mean()
        loss = actor_loss + critic_loss + self.ent_reg * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger['entropy'].append(entropy.item())
        self.logger['loss'].append(loss.item())
        self.logger['20'].append(out20.item())

