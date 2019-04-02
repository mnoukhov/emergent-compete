""" Iterated Sender Reciever Game

each round consists of three steps:
    1. agent 1 observes the target and bias
    2. agent 1 "acts", agent 2 observes action
    3. agent 2 acts determining reward

steps 3 and next round's step 1 are combined

"""
import gin
import gym
from gym.spaces import Discrete
import numpy as np
import torch


class DiscreteRange(Discrete):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.range = high - low
        super().__init__(self.range)

    def sample(self):
        return super().sample() + self.low

    def contains(self, x):
        return super().contains(x - self.low)


@gin.configurable
class IteratedSenderRecver(gym.Env):
    def __init__(self,
                 batch_size,
                 num_rounds,
                 num_targets,
                 max_bias,
                 min_bias=0):
        self.num_rounds = num_rounds
        self.num_targets = num_targets
        self.action_space = Discrete(num_targets)
        self.observation_space = Discrete(num_targets)
        self.bias_space = DiscreteRange(min_bias, max_bias)
        self.batch_size = batch_size

    def _generate_bias(self):
        return torch.randint(self.bias_space.low, self.bias_space.high,
                             size=(self.batch_size,1)).float()

    def _generate_target(self):
        return torch.randint(self.action_space.n,
                             size=(self.batch_size,1)).float()

    def reset(self):
        self.round = 0
        self.bias = self._generate_bias()
        self.recv_target = self._generate_target()
        self.send_target = (self.recv_target + self.bias) % self.num_targets

        return self.send_target

    def _distance(self, pred, target):
        dist = torch.abs(pred - target)
        circle_dist = torch.min(dist, self.num_targets - dist)
        return circle_dist ** 2

    def step(self, action):
        self.round += 1

        rewards = [- self._distance(action, self.send_target),
                   - self._distance(action, self.recv_target)]
        done = (self.round >= self.num_rounds)

        self.round_info = {
            'round': self.round,
            'send_target': self.send_target[0].item(),
            'recv_target': self.recv_target[0].item(),
            'guess': action[0].item(),
            'send_loss': (-rewards[0][0].item()) **0.5,
            'recv_loss': (-rewards[1][0].item()) **0.5,
        }

        self.recv_target = self._generate_target()
        self.send_target = (self.recv_target + self.bias) % self.num_targets

        return self.send_target, rewards, done

    def render(self, message=-1.0):
        print('--- round {} ---'.format(self.round_info['round']))
        print('targetS {:<2}   targetR {:2}'.format(
            self.round_info['send_target'],
            self.round_info['recv_target']))
        print('message {: <6.2f}   guess {:<5.2f}'.format(
            message,
            self.round_info['guess']))
        print('losses  {: <5.2f}          {:<4.2f}'.format(
            self.round_info['send_loss'],
            self.round_info['recv_loss']))
