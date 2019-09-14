import gin
import torch
from torch import nn
from torch.autograd import grad
from torch.optim import Adam

from src.agents import Deterministic, Reinforce


def magic_box(x):
    return torch.exp(x - x.detach())


@gin.configurable
class DiceLOLASender(Reinforce):
    lola = True

    def __init__(self, order, sender_lola_lr, recver_lola_lr, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.sender_lola_lr = sender_lola_lr
        self.recver_lola_lr = recver_lola_lr

    def loss(self, error, messages, logprobs, entropys, batch, recver, loss_fn):
        _, logs = super(Reinforce, self).loss(error)
        sender = self
        num_rounds = error.size(0)

        recver_params = [param.clone().detach().requires_grad_()
                         for param in recver.parameters()]
        sender_targets, recver_targets = batch

        for step in range(self.order):
            recver_error_list = []
            for round_ in range(num_rounds):
                sender_target = sender_targets[round_]
                recver_target = sender_targets[round_]
                if step == 0:
                    # we can use the actual messages our agent sent that round
                    message = messages[round_]
                else:
                    message, _, _ = sender(sender_target)

                actions, _, _ = recver.functional_forward(message, recver_params)
                recver_error_list.append(loss_fn(actions, recver_target).squeeze(1))

            # update recver
            recver_errors = torch.stack(recver_error_list, dim=1)
            recver_loss = recver_errors.mean()
            recver_grads = grad(recver_loss, recver_params, create_graph=True)
            recver_params = [param - grad * self.recver_lola_lr
                            for param, grad in zip(recver_params, recver_grads)]

        error_list = []
        message_list = []
        logprob_list = []
        entropy_list = []
        for round_ in range(num_rounds):
            sender_target = sender_targets[round_]
            if self.order == 0:
                # use the actual messages sent by the agent
                message = messages[round_]
                logprob = logprobs[round_]
                entropy = entropys[round_]
            else:
                message, logprob, entropy = sender(sender_target)

            actions, _, _ = recver.functional_forward(message, recver_params)
            error = loss_fn(actions, sender_target).squeeze(1)

            error_list.append(error)
            message_list.append(message)
            logprob_list.append(logprob)
            entropy_list.append(entropy)


        errors = torch.stack(error_list, dim=0)
        messages = torch.stack(error_list, dim=0)
        logprobs = torch.stack(logprob_list, dim=0)
        entropys = torch.stack(entropy_list, dim=0)

        # LOLA uses discounted errors and cumsum logprobs
        # but this should be equivalent to discounting the future as usual
        discounts = torch.tensor([self.gamma**t for t in range(num_rounds)]).unsqueeze(1)
        discount_error = discounts * errors
        logprob_cumsum = torch.cumsum(logprobs, dim=0)

        dice_loss = torch.sum(discount_error.detach() * magic_box(logprob_cumsum), dim=0).mean()
        entropy_loss = -entropy.mean() * sender.ent_reg
        baseline = torch.sum((1 - magic_box(logprobs)) * self.baseline, dim=0).mean()
        loss = dice_loss + entropy_loss + baseline


        if self.training:
            self.n_update += 1.
            self.baseline += (discount_error.detach().mean().item() - self.baseline) / (self.n_update)

        logs['lola_error'] = error.mean().item()
        logs['loss'] = loss.item()

        return loss, logs


@gin.configurable
class DiceLOLAReceiver(Deterministic):
    lola = True

    def __init__(self, sender, order, sender_lola_lr, recver_lola_lr, **kwargs):
        super().__init__(**kwargs)
        self.sender = sender(**kwargs)
        self.order = order
        self.sender_lola_lr = sender_lola_lr
        self.recver_lola_lr = recver_lola_lr

        self.n_update = 0
        self.baseline = 0

    @property
    def lr(self):
        return self.sender.lr

    def forward(self, state):
        return self.sender(state)

    def loss(self, error, logprobs, entropy, batch, recver, loss_fn):
        _, logs = super().loss(error)
        sender = self.sender

        recver_params = [param.clone().detach().requires_grad_()
                         for param in recver.parameters()]
        sender_targets, recver_targets = batch

        for step in range(self.order):
            messages, sender_logprobs, sender_entropy = sender(sender_targets)
            actions, _, _ = recver.functional_forward(messages, recver_params)

            sender_error = loss_fn(actions, sender_targets).squeeze()
            sender_dice_loss = (sender_error.detach() * magic_box(sender_logprobs)).mean()
            sender_entropy_loss = -sender_entropy.mean() * sender.ent_reg
            sender_baseline = ((1 - magic_box(sender_logprobs)) * self.baseline).mean()

            sender_loss = sender_dice_loss + sender_entropy_loss + sender_baseline
            sender_grads = grad(sender_loss, sender_params, create_graph=True)

            recver_error = loss_fn(actions, recver_targets).squeeze()
            recver_loss = recver_error.mean()
            recver_grads = grad(recver_loss, recver_params, create_graph=True)

            # update
            recver_params = [param - grad * self.recver_lola_lr
                            for param, grad in zip(recver_params, recver_grads)]

        messages, logprobs, entropy = sender(sender_targets)
        actions, _, _ = recver.functional_forward(messages, recver_params)

        error = loss_fn(actions, sender_targets).squeeze()
        dice_loss = (error.detach() * magic_box(logprobs)).mean()
        entropy_loss = -entropy.mean() * sender.ent_reg
        baseline = ((1 - magic_box(logprobs)) * self.baseline).mean()
        loss = dice_loss + entropy_loss + baseline

        if self.training:
            self.n_update += 1.
            self.baseline += (error.detach().mean().item() - self.baseline) / (self.n_update)

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
