from copy import deepcopy

import gin
from torch import nn
from torch.autograd import grad
from torch.optim import Adam

from src.agents import Policy



@gin.configurable
class DeterExactLOLA(nn.Module):
    def __init__(self, agent, order, lola_lr, **kwargs):
        super().__init__()
        self.agent = agent(**kwargs)
        self.order = order
        self.optimizer = self.agent.optimizer
        self.lola_lr = lola_lr

        self.other = None
        self.env = None

    def forward(self, state):
        return self.agent(state)

    def loss(self, rewards):
        targets = self.env.send_targets[0]
        agent = deepcopy(self.agent)
        agent_params = {n:p for n,p in self.agent.named_parameters()}
        set_params(agent, agent_params)

        other = deepcopy(self.other)
        other_params = {n:p for n,p in self.agent.named_parameters()}
        set_params(other, other_params)

        for step in range(self.order):
            messages = agent(targets.unsqueeze(1))
            actions = other(messages).squeeze()

            agent_rewards = self.env._reward(actions, targets)
            agent_loss, _ = agent.loss(agent_rewards)
            agent_grads = grad(agent_loss, agent.parameters(), create_graph=True)

            new_params = [param - grad * self.lola_lr
                          for param, grad in zip(agent_params, agent_grads)]
            # new_agent = deepcopy(agent)
            params = new_params
            agent = new_agent

            other_targets = (targets + self.env.bias) % self.env.num_targets
            other_rewards = self.env._reward(actions, other_targets)
            other_loss, _ = other.loss(other_rewards)
            other_grads = grad(other_loss, other.parameters(), create_graph=True)
            new_params = [param - grad * self.lola_lr
                          for param, grad in zip(other_params, other_grads)]
            # new_other = deepcopy(other)
            set_params(other, new_params)
            params = new_params
            other = new_other

        messages = self.agent(targets.unsqueeze(1))
        actions = other(messages.unsqueeze(1))
        lola_rewards = self.env._reward(actions, targets)

        return self.agent.loss(lola_rewards)
