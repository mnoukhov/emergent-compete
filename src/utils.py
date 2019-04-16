import torch


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


