import torch
import numpy as np


def log_grad_norm(log_list):
    def grad_norm_hook(module, grad_input, grad_output):
        __import__('pdb').set_trace()
        if not isinstance(grad_input, tuple):
            grad_input = [grad_input]
        log_list.append(total_norm(grad_input))

    return grad_norm_hook


def total_norm(iterable):
    return sum([x.norm(2).detach()**2 for x in iterable])**0.5


def discount_return(rewards, gamma):
    R = 0
    discounted = []
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)

    return discounted


def running_mean(x, N=100):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    afterN = (cumsum[N:] - cumsum[:-N]) / float(N)
    beforeN = cumsum[1:N] / np.arange(1, N)
    return np.concatenate([beforeN, afterN])


def circle_diff(x, y, circ):
    # x - y on a circle
    diff = x - y
    diff[diff > circ/2] -= circ
    diff[diff < -circ/2] += circ
    return diff


def soft_update(src, trg, tau):
    for trg_param, src_param in zip(trg.parameters(), src.parameters()):
        trg_param.data.copy_((1 - tau) * trg_param.data + tau * src_param.data)


def hard_update(source, target):
    for trg_param, src_param in zip(trg.parameters(), src.parameters()):
        trg_param.data.copy_(src_param.data)


