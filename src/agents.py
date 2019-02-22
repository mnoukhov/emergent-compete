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

    def __call__(self, *obs):
        if self.mode == 0:
            prompt = 'target {}: '.format(*obs)
        elif self.mode == 1:
            prompt = 'message {}: '.format(*obs)

        action = input(prompt)

        return int(action)
