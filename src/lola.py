import gin
import torch
from torch import nn
from torch.autograd import grad
from torch.optim import Adam

from src.agents import Deterministic, Reinforce


def dice(x):
    return torch.exp(x - x.detach())


@gin.configurable
class DiceLOLASender(Reinforce):
    lola = True

    def __init__(self, order, recver_lola_lr, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.recver_lola_lr = recver_lola_lr

    def loss(self, error, message, logprobs, entropy, batch, recver, loss_fn):
        _, logs = super(Reinforce, self).loss(error)
        sender = self

        recver_params = [param.clone().detach().requires_grad_()
                         for param in recver.parameters()]
        sender_targets, recver_targets = batch

        for step in range(self.order):
            actions, _, _ = recver.functional_forward(message, recver_params)

            recver_error = loss_fn(actions, recver_targets).squeeze()
            recver_loss = recver_error.mean()
            recver_grads = grad(recver_loss, recver_params, create_graph=True)

            # update
            recver_params = [param - grad * self.recver_lola_lr
                            for param, grad in zip(recver_params, recver_grads)]

        actions, _, _ = recver.functional_forward(message, recver_params)

        error = loss_fn(actions, sender_targets).squeeze()
        dice_loss = (error.detach() * dice(logprobs)).mean()
        entropy_loss = -entropy.mean() * sender.ent_reg
        baseline = ((1 - dice(logprobs)) * self.baseline).mean()
        loss = dice_loss + entropy_loss + baseline

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

        logs['lola_error'] = error.mean().item()
        logs['loss'] = loss.item()

        return loss, logs


@gin.configurable
class ExactLOLARecver(Deterministic):
    lola = True

    def __init__(self, order, sender_lola_lr, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.sender_lola_lr = sender_lola_lr

    def loss(self, error, messages, sender_logprobs, sender_entropy,
             batch, sender, loss_fn):
        _, logs = super(Deterministic, self).loss(error)
        recver = self

        sender_params = [param.clone().detach().requires_grad_()
                         for param in sender.parameters()]
        sender_targets, recver_targets = batch

        for step in range(self.order):
            # we can redo our actions because we're deterministic
            actions, _, _ = recver(messages)

            sender_error = loss_fn(actions, sender_targets).squeeze()
            sender_dice_loss = (sender_error.detach() * dice(sender_logprobs)).mean()
            sender_entropy_loss = -sender_entropy.mean() * sender.ent_reg
            # assume a fixed baseline
            sender_baseline = ((1 - dice(sender_logprobs)) * sender.baseline).mean()

            sender_loss = sender_dice_loss + sender_entropy_loss + sender_baseline
            sender_grads = grad(sender_loss, sender_params, create_graph=True)

            # update opponent
            sender_params = [param - grad * self.sender_lola_lr
                             for param, grad in zip(sender_params, sender_grads)]

            messages, sender_logprobs, sender_entropy = sender.functional_forward(sender_targets, sender_params)


        if self.order > 0:
            # otherwise we use the inputs we got in the actual game
            messages, logprobs, entropy = sender(sender_targets)

        actions, _, _ = recver(messages)
        error = loss_fn(actions, recver_targets).squeeze()
        loss = error.mean()

        logs['lola_error'] = error
        logs['loss'] = loss

        return loss, logs


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
