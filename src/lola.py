from copy import deepcopy

import gin
from torch import nn
from torch.optim import Adam

from src.agents import Policy



@gin.configurable
class DeterExactLOLA(nn.Module):
    def __init__(self, agent, order, lola_lr, **kwargs):
        super().__init__()
        self.agent = agent(**kwargs)
        self.order = order
        self.optimizer = self.agent.optimizer
        # self.target_range = torch.arange(env.num_targets).to(device)
        self.lola_lr = lola_lr

        self.other = None
        self.env = None

    def forward(self, state):
        return self.agent(state)

    def loss(self, rewards):
        targets = self.env.send_targets[0]
        agent = deepcopy(self.agent)
        agent.load_state_dict(self.agent.state_dict())
        other = deepcopy(self.other)
        other.load_state_dict(self.other.state_dict())
        agent_optimizer = Adam(agent.parameters(), lr=self.lola_lr)
        other_optimizer = Adam(other.parameters(), lr=self.lola_lr)
        for step in range(self.order):
            messages = agent(targets.unsqueeze(1))
            actions = other(messages).squeeze()

            agent_rewards = self.env._reward(actions, targets)
            agent_loss, _ = agent.loss(agent_rewards)
            agent_optimizer.zero_grad()
            agent_loss.backward(retain_graph=True, create_graph=True)
            agent_optimizer.step()

            other_targets = (targets + self.env.bias) % self.env.num_targets
            other_rewards = self.env._reward(actions, other_targets)
            other_loss, _ = other.loss(other_rewards)
            other_optimizer.zero_grad()
            other_loss.backward(create_graph=True)
            other_optimizer.step()

        messages = self.agent(targets.unsqueeze(1))
        actions = other(messages.unsqueeze(1))
        lola_rewards = self.env._reward(actions, targets)

        return self.agent.loss(lola_rewards)
