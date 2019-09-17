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

    def loss(self, error, batch, recver, start_rng_state, loss_fn, grounded=False):
        _, logs = super(Reinforce, self).loss(error)
        sender = self

        recver_params = [param.clone().detach().requires_grad_()
                         for param in recver.parameters()]
        sender_targets, recver_targets = batch

        # TODO should you reset rng state at every step?
        torch.set_rng_state(start_rng_state)

        for step in range(self.order):
            message, _, _ = sender(sender_targets)
            action, _, _ = recver.functional_forward(message.detach(), recver_params)

            if grounded:
                action = message.reshape(action.shape).float() + action

            recver_error = loss_fn(action, recver_targets).squeeze()
            recver_loss = recver_error.mean()
            recver_grads = grad(recver_loss, recver_params, create_graph=True)

            # update
            recver_params = [param - grad * self.recver_lola_lr
                            for param, grad in zip(recver_params, recver_grads)]

        message, logprobs, entropy = sender(sender_targets)
        action, _, _ = recver.functional_forward(message.detach(), recver_params)

        if grounded:
            action = message.reshape(action.shape).float() + action

        error = loss_fn(action, sender_targets).squeeze()
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

    def loss(self, error, batch, sender, sender_rng_state, loss_fn, grounded=False):
        _, logs = super(Deterministic, self).loss(error)
        recver = self

        sender_params = [param.clone().detach().requires_grad_()
                         for param in sender.parameters()]
        sender_targets, recver_targets = batch

        torch.set_rng_state(sender_rng_state)

        for step in range(self.order):
            message, sender_logprobs, sender_entropy = sender.functional_forward(sender_targets, sender_params)
            action, _, _ = recver(message)

            if grounded:
                action = message.reshape(action.shape).float() + action

            sender_error = loss_fn(action, sender_targets).squeeze()
            sender_dice_loss = (sender_error.detach() * dice(sender_logprobs)).mean()
            sender_entropy_loss = -sender_entropy.mean() * sender.ent_reg
            # assume a fixed baseline
            sender_baseline = ((1 - dice(sender_logprobs)) * sender.baseline).mean()
            sender_loss = sender_dice_loss + sender_entropy_loss + sender_baseline

            sender_grads = grad(sender_loss, sender_params, create_graph=True)

            # update opponent
            sender_params = [param - grad * self.sender_lola_lr
                             for param, grad in zip(sender_params, sender_grads)]


        message, _, _ = sender.functional_forward(sender_targets, sender_params)
        action, _, _ = recver(message)

        if grounded:
            action = message.reshape(action.shape).float() + action

        error = loss_fn(action, recver_targets).squeeze()
        loss = error.mean()

        logs['lola_error'] = error.mean().item()
        logs['loss'] = loss.item()

        return loss, logs
