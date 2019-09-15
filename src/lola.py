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

    def __init__(self, order, recver_lola_lr, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.recver_lola_lr = recver_lola_lr

    def loss(self, error, messages, logprobs, entropys, batch, recver, loss_fn):
        _, logs = super(Reinforce, self).loss(error)
        sender = self
        num_rounds = error.size(0)
        batch_size = error.size(1)

        recver_params = [param.clone().detach().requires_grad_()
                         for param in recver.parameters()]
        sender_targets, recver_targets = batch

        # Update opponent with lookaheads
        for step in range(self.order):
            recver_error_list = []
            prev_sender_target = torch.zeros(batch_size, 1)
            prev_message = torch.zeros(batch_size).long()
            prev_action = torch.zeros(batch_size, 1)
            prev_sender_error = torch.zeros(batch_size, 1)
            prev_recver_error = torch.zeros(batch_size, 1)

            for round_ in range(num_rounds):
                sender_target = sender_targets[round_]
                recver_target = sender_targets[round_]
                first_round = torch.ones(batch_size).long() if round_ == 0 else torch.zeros(batch_size).long()
                if step == 0:
                    # we can use the actual messages our agent sent that round
                    message = messages[round_]
                else:
                    message, _, _, _ = sender(sender_target,
                                              prev_sender_target,
                                              prev_message,
                                              prev_sender_error,
                                              first_round)

                action, _, _ = recver.functional_forward(message,
                                                         prev_message,
                                                         prev_action,
                                                         prev_recver_error,
                                                         first_round,
                                                         recver_params)
                sender_error = loss_fn(action, sender_target)
                recver_error = loss_fn(action, recver_target)

                recver_error_list.append(recver_error.squeeze(1))

                prev_sender_target = sender_target
                prev_message = message.clone().detach()
                prev_action = action.clone().detach()
                prev_sender_error = sender_error.clone().detach()
                prev_recver_error = recver_error.clone().detach()

            recver_errors = torch.stack(recver_error_list, dim=1)
            recver_loss = recver_errors.mean()
            recver_grads = grad(recver_loss, recver_params, create_graph=True)
            recver_params = [param - grad * self.recver_lola_lr
                            for param, grad in zip(recver_params, recver_grads)]

        # Play against updated opponent
        error_list = []
        logprob_list = []
        entropy_list = []

        prev_sender_target = torch.zeros(batch_size, 1)
        prev_message = torch.zeros(batch_size).long()
        prev_action = torch.zeros(batch_size, 1)
        prev_sender_error = torch.zeros(batch_size, 1)
        prev_recver_error = torch.zeros(batch_size, 1)

        for round_ in range(num_rounds):
            sender_target = sender_targets[round_]
            recver_target = recver_targets[round_]
            first_round = torch.ones(batch_size).long() if round_ == 0 else torch.zeros(batch_size).long()

            if self.order == 0:
                # we can use the actual messages our agent sent that round
                message = messages[round_]
                logprob = logprobs[round_]
                entropy = entropys[round_]
            else:
                message, logprob, entropy, _ = sender(sender_target,
                                                      prev_sender_target,
                                                      prev_message,
                                                      prev_sender_error,
                                                      first_round)

            action, _, _ = recver.functional_forward(message,
                                                     prev_message,
                                                     prev_action,
                                                     prev_recver_error,
                                                     first_round,
                                                     recver_params)

            sender_error = loss_fn(action, sender_target)
            recver_error = loss_fn(action, recver_target)

            error_list.append(sender_error.squeeze(1))
            logprob_list.append(logprob)
            entropy_list.append(entropy)

            prev_sender_target = sender_target
            prev_message = message.clone().detach()
            prev_action = action.clone().detach()
            prev_sender_error = sender_error.clone().detach()
            prev_recver_error = recver_error.clone().detach()

        errors = torch.stack(error_list, dim=0)
        logprobs = torch.stack(logprob_list, dim=0)
        entropy = torch.stack(entropy_list, dim=0)

        # LOLA-DiCE uses discounted errors and cumsum logprobs
        # but this should be equivalent to discounting the future as usual
        discounts = torch.tensor([self.gamma**t for t in range(num_rounds)]).unsqueeze(1)
        discount_error = discounts * errors
        logprob_cumsum = torch.cumsum(logprobs, dim=0)

        dice_loss = torch.sum(discount_error.detach() * magic_box(logprob_cumsum), dim=0).mean()
        entropy_loss = -entropy.mean() * self.ent_reg
        baseline = torch.sum((1 - magic_box(logprobs)) * self.baseline, dim=0).mean()
        loss = dice_loss + entropy_loss + baseline


        if self.training:
            self.n_update += 1.
            self.baseline += (discount_error.detach().mean().item() - self.baseline) / (self.n_update)

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

    def loss(self, error, batch, sender, sender_rng_state, loss_fn):
        _, logs = super(Deterministic, self).loss(error)
        recver = self
        num_rounds = error.size(0)
        batch_size = error.size(1)

        sender_params = [param.clone().detach().requires_grad_()
                         for param in sender.parameters()]
        sender_targets, recver_targets = batch

        torch.set_rng_state(sender_rng_state)

        # Update opponent with lookaheads
        for step in range(self.order):
            sender_error_list = []
            sender_logprob_list = []
            sender_entropy_list = []

            prev_sender_target = torch.zeros(batch_size, 1)
            prev_message = torch.zeros(batch_size).long()
            prev_action = torch.zeros(batch_size, 1)
            prev_sender_error = torch.zeros(batch_size, 1)
            prev_recver_error = torch.zeros(batch_size, 1)

            for round_ in range(num_rounds):
                sender_target = sender_targets[round_]
                recver_target = recver_targets[round_]
                first_round = torch.ones(batch_size).long() if round_ == 0 else torch.zeros(batch_size).long()
                # same as actual game because of rng state
                message, sender_logprob, sender_entropy, _ = sender.functional_forward(sender_target,
                                                                                       prev_sender_target,
                                                                                       prev_message,
                                                                                       prev_sender_error,
                                                                                       first_round,
                                                                                       sender_params)
                action, _, _ = recver(message,
                                      prev_message,
                                      prev_action,
                                      prev_recver_error,
                                      first_round)

                sender_error = loss_fn(action, sender_target)
                recver_error = loss_fn(action, recver_target)

                sender_error_list.append(sender_error.squeeze(1))
                sender_logprob_list.append(sender_logprob)
                sender_entropy_list.append(sender_entropy)

                prev_sender_target = sender_target
                prev_message = message.clone().detach()
                prev_action = action.clone().detach()
                prev_sender_error = sender_error.clone().detach()
                prev_recver_error = recver_error.clone().detach()

            errors = torch.stack(sender_error_list, dim=0)
            logprobs = torch.stack(sender_logprob_list, dim=0)
            entropys = torch.stack(sender_entropy_list, dim=0)

            discounts = torch.tensor([sender.gamma**t for t in range(num_rounds)]).unsqueeze(1)
            discount_error = discounts * errors
            logprob_cumsum = torch.cumsum(logprobs, dim=0)

            dice_loss = torch.sum(discount_error.detach() * magic_box(logprob_cumsum), dim=0).mean()
            entropy_loss = -entropys.mean() * sender.ent_reg
            baseline = torch.sum((1 - magic_box(logprobs)) * sender.baseline, dim=0).mean()
            sender_loss = dice_loss + entropy_loss + baseline
            sender_grads = grad(sender_loss, sender_params, create_graph=True)

            sender_params = [param - grad * self.sender_lola_lr
                            for param, grad in zip(sender_params, sender_grads)]

        # Play against updated opponent
        error_list = []

        prev_sender_target = torch.zeros(batch_size, 1)
        prev_message = torch.zeros(batch_size).long()
        prev_action = torch.zeros(batch_size, 1)
        prev_sender_error = torch.zeros(batch_size, 1)
        prev_recver_error = torch.zeros(batch_size, 1)

        for round_ in range(num_rounds):
            sender_target = sender_targets[round_]
            recver_target = recver_targets[round_]
            first_round = torch.ones(batch_size).long() if round_ == 0 else torch.zeros(batch_size).long()
            message, sender_logprob, sender_entropy, _ = sender.functional_forward(sender_target,
                                                                                   prev_sender_target,
                                                                                   prev_message,
                                                                                   prev_sender_error,
                                                                                   first_round,
                                                                                   sender_params)
            action, _, _ = recver(message,
                                  prev_message,
                                  prev_action,
                                  prev_recver_error,
                                  first_round)

            sender_error = loss_fn(action, sender_target)
            recver_error = loss_fn(action, recver_target)

            error_list.append(recver_error.squeeze(1))

            prev_sender_target = sender_target
            prev_message = message.clone().detach()
            prev_action = action.clone().detach()
            prev_sender_error = sender_error.clone().detach()
            prev_recver_error = recver_error.clone().detach()

        error = torch.stack(error_list, dim=0)
        loss = error.mean()

        logs['lola_error'] = error.mean().item()
        logs['loss'] = loss.item()

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
