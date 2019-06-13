#TODO change state
# state = prev 2 actions rewards critic state should have both actions
from copy import deepcopy

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.agents import mode, Policy
from src.utils import soft_update
from src.memory import ReplayBuffer


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

    def forward(self, state, message, action):
        message = message.unsqueeze(1)
        action = action.unsqueeze(1)
        inputs = torch.cat((state, message, action), dim=1)
        return self.net(inputs)


@gin.configurable
class MADDPG(Policy):
    def __init__(self, num_agents, num_actions, state_dim,
                 tau, gamma, lr, batch_size, device, warmup_episodes,
                 target_update_freq, opponent=None,**kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.tau = tau
        self.gamma = gamma
        self.warmup_episodes = warmup_episodes
        self.target_update_freq = target_update_freq
        self.device = device
        self.batch_size = batch_size

        self.actor = Actor(state_dim)
        self.actor_target = deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr)

        self.critic = Critic(num_agents, state_dim)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr)

        self.memory = ReplayBuffer()
        self.noise = Normal(0, scale=1)

        self.logger.update({'loss': []})

    def action(self, state):
        with torch.no_grad():
            action = self.actor(state).squeeze()

        if self.training:
            device = state.device
            batch_size = state.shape[0]
            action += self.noise.sample(sample_shape=(batch_size,)).to(device)

        return action % self.num_actions

    def update(self, ep, rewards, log):
        super().update(ep, rewards, log)
        if ep < self.warmup_episodes:
            return

        state, message, action, send_reward, recv_reward, next_state = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        message = message.to(self.device)
        action = action.to(self.device)
        send_reward = send_reward.to(self.device)
        recv_reward = recv_reward.to(self.device)
        next_state = next_state.to(self.device)

        # update critic
        current_Q = self.critic(state, message, action)
        if self.mode == mode.SENDER:
            next_message = self.actor_target(state).squeeze()
            next_recv_state = state.clone()
            next_recv_state[:,0] = next_message
            next_action = self.opponent.action(next_recv_state)
            reward = send_reward
        else:
            next_message = next_state[:,0]
            next_action = self.actor_target(state).squeeze()
            reward = recv_reward

        target_Q = reward.unsqueeze(1) + self.gamma * self.critic_target(next_state, next_message, next_action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        new_state = state.clone()
        if self.mode == mode.SENDER:
            new_message = self.actor(state).squeeze()
            new_state[:,0] = new_message
            new_action = self.opponent.action(new_state)
        else:
            new_message = new_state[:,0]
            new_action = self.actor(new_state).squeeze()
        reward = self.critic(new_state, new_message, new_action)
        actor_loss = -reward.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update targets
        if ep % self.target_update_freq == 0:
            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)

        self.logger['loss'].append(actor_loss.item() + critic_loss.item())
