from abc import abstractmethod
from copy import deepcopy
from enum import Enum
import random

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.mode = mode
        self.logger = {
            'ep_reward': [],
            'round_reward': [],
        }
        # self.writer = SummaryWriter(comment=f'/{mode.name}/')


    def last(self, metric):
        values = self.logger.get(metric, None)
        if values:
            return values[-1]
        else:
            return 0.

    def action(self, state):
        pass

    def update(self, ep, rewards, log=True):
        rewards = torch.stack(rewards, dim=1)
        mean_reward = rewards.mean().item()
        round_reward = rewards.mean(dim=0).tolist()
        self.logger['ep_reward'].append(mean_reward)
        self.logger['round_reward'].append(round_reward)

        # if log is True:
            # self.writer.add_scalar('reward', mean_reward, global_step=ep)
            # for r, reward in enumerate(round_reward):
                # self.writer.add_scalar(f'reward/{r}', reward, global_step=ep)


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
    def __init__(self, num_actions, lr, weight_decay, device, **kwargs):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.device = device
        self.policy = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)).to(device)
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        self.logger.update({
            'loss': [],
        })

    def action(self, state):
        action = self.policy(state).squeeze()
        return action

    def update(self, ep, rewards, log, retain_graph=False):
        super().update(ep, rewards, log)
        loss = -torch.stack(rewards).mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        norm = sum(p.grad.data.norm(2) ** 2 for p in self.policy.parameters())**0.5
        self.optimizer.step()

        self.logger['loss'].append(loss.item())
        if log:
            self.scheduler.step(loss)
            # self.writer.add_scalar('loss', loss.item(), global_step=ep)
            # self.writer.add_scalar('grad norm', norm, global_step=ep)


@gin.configurable
class PolicyGradient(Policy):
    def __init__(self, num_actions, lr, weight_decay, gamma, ent_reg, min_std=0, **kwargs):
        super().__init__(**kwargs)
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

        self.entropy = 0
        self.log_probs = []


@gin.configurable
class CategoricalPG(Policy):
    def __init__(self, num_actions, lr, weight_decay, gamma, ent_reg, device, **kwargs):
        super().__init__(**kwargs)
        self.policy = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr,
                              weight_decay=weight_decay)

        self.ent_reg = ent_reg
        self.gamma = gamma
        self.entropy = 0
        self.log_probs = []
        self.logger.update({'loss': []})

    def action(self, state):
        logits = self.policy(state)
        probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        sample = dist.sample()

        self.entropy += dist.entropy()
        self.log_probs.append(dist.log_prob(sample))

        return sample.float()

    def update(self, ep, rewards, log):
        super().update(ep, rewards, log)

        # returns = torch.stack(discount_return(rewards, self.gamma), dim=1)
        returns = torch.stack(rewards, dim=1)
        logprobs = torch.stack(self.log_probs, dim=1)
        entropy = torch.mean(self.entropy)
        loss = -(returns * logprobs).mean() - self.ent_reg * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger['loss'].append(loss.item())

        self.entropy = 0
        self.log_probs = []


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
                 warmup_episodes, device, **kwargs):
        super().__init__(**kwargs)
        self.actor = Actor(7).to(device)
        self.actor_target = deepcopy(self.actor).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(7,1).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer()
        self.noise = Normal(0, scale=0.1)

        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.warmup_episodes = warmup_episodes
        self.batch_size = batch_size
        self.device = device
        self.logger.update({
            'loss': [],
        })

    def action(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            action = self.actor(state).squeeze().cpu()

        if self.training:
            batch_size = state.shape[0]
            action += self.noise.sample(sample_shape=(batch_size,))

        return action % self.num_actions

    def update(self, ep, rewards, log):
        super().update(ep, rewards, log)
        if ep < self.warmup_episodes:
            return

        # hardcoded for recver
        state, _, action, _, reward, next_state = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)


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
        if log:
            self.writer.add_scalar('actor loss', actor_loss.item(), global_step=ep)
            self.writer.add_scalar('critic loss', critic_loss.item(), global_step=ep)

    def target_update(self):
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)


class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, input):
        return self.net(input)


class Critic(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_agents, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, state, action):
        action = action.unsqueeze(1)
        inputs = torch.cat((state, action), dim=1)
        return self.net(inputs)
