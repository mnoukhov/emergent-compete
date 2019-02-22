from abc import ABC, abstractmethod

import torch


class Policy(ABC):
    @abstractmethod
    def __call__(self, obs):
        action = None
        return action


class Human(Policy):
    def __init__(self, mode=0):
        # 0 = sender, 1 = receiver
        self.mode = mode

    def __call__(self, obs):
        if mode == 0:
            prompt = 'target {}, bias {}'.format(*obs)
        else:
            prompt = 'message {}'.format(obs)

        action = input(prompt)

        return action

