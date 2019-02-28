import random

import gin
import torch
import torch.nn as nn

class Policy(object):
    pass

class Human(Policy):
    def __init__(self, mode):
        # 0 = sender, 1 = receiver
        self.mode = mode

    def __call__(self, obs):
        if obs.shape[0] > 1:
            raise Exception("use batch size = 1 for human player")

        if self.mode == 0:
            prompt = 'target {}: '.format(obs[0].item())
        elif self.mode == 1:
            prompt = 'message {}: '.format(obs[0].item())

        action = int(input(prompt))

        return torch.LongTensor([action])


@gin.configurable
class QNet(Policy):
    def __init__(self, action_range, epsilon=0.99):
        super().__init__()
        self.action_range = action_range
        self.net = torch.nn.Linear(1, action_range)
        self.epsilon = epsilon

    def __call__(self, obs):
        logits = self.net(obs)

        return self._epsilon_greedy(logits)

    def _epsilon_greedy(self, logits):
        if random.random() <= self.epsilon:
            return torch.randint(self.action_range - 1, size=(logits.shape[0],))
        else:
            return torch.argmax(logits, dim=1)

