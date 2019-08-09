import gin
from torch import nn
from torch.autograd import grad
from torch.optim import Adam

from src.game import CircleLoss



@gin.configurable
class DeterExactLOLA(nn.Module):
    def __init__(self, agent, order, lola_lr, **kwargs):
        super().__init__()
        self.agent = agent(**kwargs)
        self.order = order
        self.optimizer = self.agent.optimizer
        self.lola_lr = lola_lr

    def forward(self, state):
        return self.agent(state)

    def loss(self, batch, other):
        agent_params = [param.clone().detach().requires_grad_()
                        for param in self.agent.parameters()]
        other_params = [param.clone().detach().requires_grad_()
                        for param in other.parameters()]
        circle_loss = CircleLoss()
        send_targets, recv_targets = batch

        for step in range(self.order):
            messages = self.agent.functional_forward(send_targets, agent_params)
            actions = other.functional_forward(messages, other_params)

            agent_rewards = -circle_loss(actions, send_targets)
            agent_loss, _ = self.agent.loss(agent_rewards)
            agent_grads = grad(agent_loss, agent_params, create_graph=True)

            other_rewards = -circle_loss(actions, recv_targets)
            other_loss, _ = other.loss(other_rewards)
            other_grads = grad(other_loss, other_params, create_graph=True)

            # update
            agent_params = [param - grad * self.lola_lr
                            for param, grad in zip(agent_params, agent_grads)]
            other_params = [param - grad * self.lola_lr
                            for param, grad in zip(other_params, other_grads)]


        messages = self.agent(send_targets)
        actions = other.functional_forward(messages, other_params)
        lola_rewards = -circle_loss(actions, send_targets)

        return self.agent.loss(lola_rewards)
