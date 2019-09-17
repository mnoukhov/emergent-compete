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
from src.memory import ReplayBuffer
from src.utils import discount_return, soft_update

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

    def update(self, ep, rewards, log, **kwargs):
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
        # if log:
            # self.writer.add_scalar('actor loss', actor_loss.item(), global_step=ep)
            # self.writer.add_scalar('critic loss', critic_loss.item(), global_step=ep)

    def target_update(self):
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)


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

        self.actor = Actor(state_dim).to(device)
        self.actor_target = deepcopy(self.actor).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr)

        self.critic = MACritic(num_agents, state_dim).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
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

        return action

    def update(self, ep, rewards, log, **kwargs):
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

class MACritic(nn.Module):
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
class DeterExactLOLA(Deterministic):
    lola = True

    def __init__(self, agent, order, lola_lr, **kwargs):
        super().__init__()
        self.agent = agent(**kwargs)
        self.order = order
        self.optimizer = self.agent.optimizer
        self.lola_lr = lola_lr
        self.other = None
        self.loss_fn = None

    def forward(self, state):
        return self.agent(state)

    def loss(self, batch):
        #TODO change to make loss interface same as others
        other = self.other
        circle_loss = self.loss_fn

        agent_params = [param.clone().detach().requires_grad_()
                        for param in self.agent.parameters()]
        other_params = [param.clone().detach().requires_grad_()
                        for param in other.parameters()]
        send_targets, recv_targets = batch

        for step in range(self.order):
            messages, _, _ = self.agent.functional_forward(send_targets, agent_params)
            actions, _, _ = other.functional_forward(messages, other_params)

            agent_neg_rewards = circle_loss(actions, send_targets)
            agent_loss, _ = self.agent.loss(agent_neg_rewards)
            agent_grads = grad(agent_loss, agent_params, create_graph=True)

            other_neg_rewards = circle_loss(actions, recv_targets)
            other_loss, _ = other.loss(other_neg_rewards)
            other_grads = grad(other_loss, other_params, create_graph=True)

            # update
            agent_params = [param - grad * self.lola_lr
                            for param, grad in zip(agent_params, agent_grads)]
            other_params = [param - grad * self.lola_lr
                            for param, grad in zip(other_params, other_grads)]


        messages, _, _ = self.agent(send_targets)
        actions, _, _ = other.functional_forward(messages, other_params)
        lola_rewards = -circle_loss(actions, send_targets)

        return self.agent.loss(lola_rewards)
