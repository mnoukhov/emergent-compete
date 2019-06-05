#TODO change state
# state = prev 2 actions rewards critic state should have both actions
from copy import deepcopy

import gin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.agents import mode, Policy
from src.utils import soft_update, hard_update
from src.memory import ReplayBuffer, Experience


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


#TODO fix net
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

    def forward(self, states, message, action):
        message = message.unsqueeze(1)
        action = action.unsqueeze(1)
        inputs = torch.cat((states, message, action), dim=1)
        return self.net(inputs)


@gin.configurable
class MADDPG(Policy):
    def __init__(self, num_agents, num_actions, state_dim,
                 tau, gamma, lr, batch_size, opponent,
                 warmup_episodes, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.tau = tau
        self.gamma = gamma
        self.opponent = opponent

        self.actor = Actor(state_dim)
        self.actor_target = deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr)

        self.critic = Critic(num_agents, state_dim)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr)

        self.memory = ReplayBuffer()
        self.noise = Normal(0, scale=0.2)

        self.loss_fn = nn.MSELoss()

    def action(self, state):
        batch_size = state.shape[0]
        with torch.no_grad():
            action = self.actor(state).squeeze()

        if self.training:
            action += self.noise.sample(sample_shape=(batch_size,))

        return action % self.num_actions

    def update(self):
        super().update()
        state, message, action, send_reward, recv_reward, next_state = self.memory_iter.next()

        # update critic
        current_Q = self.critic(state, message, action)
        if mode == mode.SENDER:
            next_message = self.actor_target(state).squeeze()
            next_recv_state = state.clone()
            next_recv_state[:,0] = next_message
            next_action = self.opponent.action(next_recv_state).detach()
            reward = send_reward
        else:
            next_message = message
            next_action = self.actor_target(next_message).squeeze()
            reward = recv_reward
        target_Q = reward + self.gamma * self.critic_target(next_state, next_message, next_action)
        critic_loss = self.loss_fn(current_Q, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backwards()
        self.critic_optim.step()

        # update actor
        new_state = state.clone()
        if mode == mode.SENDER:
            new_message = self.actor(state).squeeze()
            new_state[:,0] = new_message
            new_action = self.opponent(new_state)
        elif mode == mode.RECVER:
            new_action = self.actor(new_state)
        reward = self.critic(new_state, new_message, new_action)
        actor_loss = -reward.mean()

        self.actor_optim.zero_grad()
        actor_loss.backwards()
        self.actor_optim.step()

        self.logger['loss'].append(actor_loss.item() + critic_loss.item())

    def target_update(self):
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)
