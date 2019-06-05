from abc import abstractmethod
from copy import deepcopy
from enum import Enum
import random

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.memory import ReplayBuffer
from src.utils import (discount_return, soft_update, hard_update)

mode = Enum('Player', 'SENDER RECVER')


class Policy(nn.Module):
    def __init__(self, mode, *args, **kwargs):
        super().__init__()
        self.rewards = []
        self.mode = mode
        self.logger = {
            'ep_reward': [],
            'round_reward': [],
        }
        self.writer = SummaryWriter(comment=mode.name)

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

    def update(self, log=True):
        rewards = torch.stack(self.rewards, dim=1)
        mean_reward = rewards.mean().item()
        round_reward = rewards.mean(dim=0).tolist()
        self.logger['ep_reward'].append(mean_reward)
        self.logger['round_reward'].append(round_reward)

        if log:
            self.writer.add_scalar('reward', mean_reward)
            for r, reward in enumerate(round_reward):
                self.writer.add_scalar(f'reward/{r}', reward)
            print("WROTE")

@gin.configurable
class Human(Policy):
    def action(self, state, mode):
        obs = state[0]

        if self.mode == mode.SENDER:
            prompt = 'target {}: '.format(obs.item())
        elif self.mode == mode.RECVER:
            prompt = 'message {}: '.format(obs.item())

        action = float(input(prompt))

        return torch.tensor([action])


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
        target = state[:,0]
        message = target + self.bias.sample()
        return message % self.num_actions


@gin.configurable
class DeterministicGradient(Policy):
    def __init__(self, num_actions, lr):
        super().__init__()
        self.num_actions = num_actions
        self.policy = nn.Sequential(
            nn.Linear(7, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1))
        self.optimizer = Adam(self.parameters(), lr=lr)

        self.logger.update({
            'loss': [],
            '20 start': [],
            '20 same': [],
            'weight grad': [],
            'weights': [],
            'biases': [],
        })
        self.weight_grad = []
        self.act_grad = []
        self.preact_grad = []

    def reset(self):
        super().reset()
        del self.weight_grad[:]
        del self.act_grad[:]
        del self.preact_grad[:]

    def action(self, state):
        action = self.policy(state).squeeze()
        return action % self.num_actions

    def update(self):
        super().update()
        loss = -torch.stack(self.rewards).mean()
        self.logger['loss'].append(loss.item())
        # start20 = self.policy(torch.tensor([20., 0., 0., 0., 0., 1.]))[0]
        # self.logger['20 start'].append(start20.item())
        # same20 = self.policy(torch.tensor([20., 10., 10., 10., 10., 0.,]))[0]
        # self.logger['20 same'].append(same20.item())

        self.optimizer.zero_grad()
        loss.backward()
        norm = sum(p.grad.data.norm(2) ** 2 for p in self.policy.parameters())**0.5
        self.logger['weight grad'].append(norm)
        # self.logger['weights'].append(self.policy.weight.data[0].tolist())
        # self.logger['biases'].append(self.policy.bias.data[0].tolist())
        self.optimizer.step()


@gin.configurable
class PolicyGradient(Policy):
    def __init__(self, num_actions, lr, weight_decay, gamma, ent_reg, min_std):
        super().__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.min_std = torch.tensor(min_std)
        self.policy = nn.Sequential(
            nn.Linear(7, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 2))

        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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
        # self.policy.weight.register_hook(self.weight_grad.append)

    def reset(self):
        super().reset()
        del self.log_probs[:]
        del self.weight_grad[:]
        self.entropy = 0

    def action(self, state):
        out = self.policy(state)
        mean, std = out.chunk(2, dim=1)
        mean = mean.squeeze()
        std = std.squeeze()

        std = torch.max(std, self.min_std)
        dist = Normal(mean, std)
        sample = dist.sample()

        self.entropy += dist.entropy()
        self.log_probs.append(dist.log_prob(sample))

        return sample % self.num_actions

    def update(self):
        super().update()
        # out20 = self.policy(torch.tensor([20., 0., 0., 0.]))[0]

        returns = torch.stack(discount_return(self.rewards, self.gamma), dim=1)
        logprobs = torch.stack(self.log_probs, dim=1)
        entropy = torch.mean(self.entropy)
        loss = -(returns * logprobs).mean() - self.ent_reg * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger['loss'].append(loss.item())
        # self.logger['grad'].append(torch.stack(self.weight_grad).norm(2).item())
        self.logger['entropy'].append(entropy.item())
        # self.logger['20'].append(out20.item())


@gin.configurable
class CategoricalPG(PolicyGradient):
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(num_actions, *args, **kwargs)
        self.policy = nn.Sequential(
            nn.Linear(7, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, self.num_actions))

    def action(self, state):
        logits = self.policy(state)
        probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        sample = dist.sample()

        self.entropy += dist.entropy()
        self.log_probs.append(dist.log_prob(sample))

        return sample.float()


@gin.configurable
class A2C(Policy):
    def __init__(self, num_actions, gamma, lr, min_std, ent_reg):
        super().__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.min_std = torch.tensor(min_std)
        self.ent_reg = ent_reg

        self.actor = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 2))
        self.critic =  nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1))
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
        mean, std = self.actor(state).chunk(2, dim=1)
        mean, std = mean.squeeze(), std.squeeze()
        std = torch.max(std, self.min_std)
        dist = Normal(mean, std)
        sample = dist.sample()

        value = self.critic(state).squeeze()
        self.entropy += dist.entropy()
        self.log_probs.append(dist.log_prob(sample))
        self.values.append(value)

        return sample % self.num_actions

    def update(self):
        super().update()
        in20 = torch.zeros(7)
        in20[0] = 20.
        out20 = self.actor(in20)[0] % self.num_actions

        returns = torch.stack(discount_return(self.rewards, self.gamma), dim=1)
        logprobs = torch.stack(self.log_probs, dim=1)
        values = torch.stack(self.values, dim=1)
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


@gin.configurable
class DDPG(Policy):
    def __init__(self, num_actions, gamma, tau,
                 actor_lr, critic_lr, batch_size,
                 warmup_episodes, **kwargs):
        super().__init__(**kwargs)
        self.actor = Actor(12)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(12,1)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer()
        self.noise = Normal(0, scale=0.2)

        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.warmup_episodes = warmup_episodes
        self.batch_size = batch_size
        self.logger.update({
            'loss': [],
        })

    def action(self, state):
        batch_size = state.shape[0]
        with torch.no_grad():
            action = self.actor(state).squeeze()

        if self.training:
            action += self.noise.sample(sample_shape=(batch_size,))

        return action % self.num_actions

    def update(self, ep, **kwargs):
        super().update(**kwargs)
        if ep < self.warmup_episodes:
            return

        # hardcoded for recver
        state, _, action, _, reward, next_state = self.memory.sample(self.batch_size)

        current_Q = self.critic(state, action)
        next_action = self.actor_target(next_state).squeeze()
        next_Q = self.critic_target(next_state, next_action)
        target_Q = reward.unsqueeze(1) + self.gamma * next_Q
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        current_action = self.actor(state).squeeze()
        critic_reward = self.critic(state, current_action)
        actor_loss = -critic_reward.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.target_update()

        self.logger['loss'].append(actor_loss.item() + critic_loss.item())
        self.writer.add_scalar('actor loss', actor_loss.item(), global_step=ep)
        self.writer.add_scalar('critic loss', critic_loss.item(), global_step=ep)

    def target_update(self):
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)


class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1))

    def forward(self, input):
        return self.net(input)


class Critic(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_agents, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, state, action):
        action = action.unsqueeze(1)
        inputs = torch.cat((state, action), dim=1)
        return self.net(inputs)
