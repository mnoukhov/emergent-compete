import copy
from enum import Enum

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

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

    def __init__(self, input_size, output_size, hidden_size,
                 lr, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.policy = RelaxedEmbedding(input_size, output_size)
        elif self.num_layers == 2:
            self.policy = nn.Sequential(
                RelaxedEmbedding(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size))
        else:
            self.policy = nn.Sequential(
                RelaxedEmbedding(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size))
        self.lr = lr

    def forward(self, state):
        action = self.policy(state)

        return action, torch.tensor(0.), torch.tensor(0.)

    def functional_forward(self, x, weights):
        if self.num_layers == 1:
            out = relaxedembedding(x, weights[0])
        elif self.num_layers == 2:
            out = relaxedembedding(x, weights[0])
            out = F.relu(out)
            out = F.linear(out, weights[1], weights[2])
        else:
            out = relaxedembedding(x, weights[0])
            out = F.relu(out)
            out = F.linear(out, weights[1], weights[2])
            out = F.relu(out)
            out = F.linear(out, weights[3], weights[4])

        return out, torch.tensor(0.), torch.tensor(0.)

    def loss(self, error, *args):
        _, logs = super().loss(error)
        loss = error.mean()

        logs['loss'] = loss.item()

        return loss, logs


@gin.configurable
class Reinforce(Policy):
    def __init__(self, input_size, output_size, hidden_size,
                 lr, ent_reg, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.policy = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LogSoftmax(dim=1))
        elif self.num_layers == 2:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.LogSoftmax(dim=1))
        else:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.LogSoftmax(dim=1))

        self.ent_reg = ent_reg
        self.lr = lr
        self.baseline = 0.
        self.n_update = 0.

    def forward(self, state):
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if self.training:
            sample = dist.sample()
        else:
            sample = logits.argmax(dim=1)

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def functional_forward(self, x, weights):
        if self.num_layers == 1:
            out = F.linear(x, weights[0], weights[1])
            logits = F.log_softmax(out, dim=1)
        elif self.num_layers == 2:
            out = F.linear(x, weights[0], weights[1])
            out = F.relu(out)
            out = F.linear(out, weights[2], weights[3])
            logits = F.log_softmax(out, dim=1)
        else:
            out = F.linear(x, weights[0], weights[1])
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

    def forward_dist(self, state):
        logits = self.policy(state)
        return Categorical(logits=logits)

    def loss(self, error, logprobs, entropy):
        _, logs = super().loss(error)

        policy_loss = ((error.detach() - self.baseline) * logprobs).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()
        logs['entropy'] = entropy.mean().item()

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs


@gin.configurable
class Gaussian(Policy):
    retain_graph = True

    def __init__(self, input_size, output_size, hidden_size,
                 lr, ent_reg, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        if self.num_layers == 2:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU())
        elif self.num_layers == 3:
            self.policy = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU())
        else:
            self.policy = None

        self.mean = nn.Linear(hidden_size, output_size)
        self.var = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU())

        self.ent_reg = ent_reg
        self.lr = lr
        self.baseline = 0.
        self.n_update = 0.

    def forward(self, state):
        logits = self.policy(state)
        mean = self.mean(logits)
        var = self.var(logits) + 1e-7
        dist = Normal(mean, var)
        entropy = dist.entropy()

        if self.training:
            sample = dist.rsample()
        else:
            sample = mean

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def functional_forward(self, x, weights):
        if self.num_layers == 2:
            out = F.linear(x, weights[0], weights[1])
            out = F.relu(out)
            mean = F.linear(out, weights[2], weights[3])
            var = F.relu(F.linear(out, weights[4], weights[5])) + 1e-7
        elif self.num_layers == 3:
            out = F.linear(x, weights[0], weights[1])
            out = F.relu(out)
            out = F.linear(out, weights[2], weights[3])
            out = F.relu(out)
            mean = F.linear(out, weights[4], weights[5])
            var = F.relu(F.linear(out, weights[6], weights[7])) + 1e-7
        else:
            out = x
            mean = F.linear(out, weights[0], weights[1])
            var = F.relu(F.linear(out, weights[2], weights[3])) + 1e-7

        dist = Normal(mean, var)
        entropy = dist.entropy()

        if self.training:
            sample = dist.rsample()
        else:
            sample = mean

        logprobs = dist.log_prob(sample)

        return sample, logprobs, entropy

    def forward_dist(self, state):
        logits = self.policy(state)
        mean = self.mean(logits)
        var = self.var(logits) + 1e-7
        return Normal(mean, var)

    def loss(self, error, logprobs, entropy):
        _, logs = super().loss(error)

        # grad_loss = error.mean()
        policy_loss = ((error.detach() - self.baseline) * logprobs).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        loss = policy_loss + entropy_loss

        logs['loss'] = loss.item()
        logs['entropy'] = entropy.mean().item()

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        return loss, logs


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)


        def forward(self, state, action):
            q = F.relu(self.l1(torch.cat([state, action], 1)))
            q = F.relu(self.l2(q))
            return self.l3(q)


@gin.configurable
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action=36, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

