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

import math


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

        self.send_diffs = []
        self.recv_diffs = []

    def _generate_bias(self):
        return torch.randint(self.bias_space.low, self.bias_space.high + 1,
                             size=(self.batch_size,1)).float()

    def _generate_target(self):
        return torch.randint(self.action_space.n,
                             size=(self.batch_size,1)).float()


    def _reward(self, pred, target=None):
        if target is None:
            target = torch.tensor(0.)
        diff = torch.abs(pred - target)
        dist = torch.min(diff, self.num_targets - diff)
        return 1 - 2 * dist / self.num_targets

    def _cos_reward(self, pred, target=None):
        if target is None:
            target = torch.tensor(0.)
        norm_dist = (pred - target) / self.num_targets
        radian_dist = 2*math.pi*norm_dist
        return 0.5*(1 + torch.cos(radian_dist))

    def reset(self):
        self.round = 0
        self.bias = self._generate_bias()
        self.recv_target = self._generate_target()
        self.send_target = (self.recv_target + self.bias) % self.num_targets

        return self.send_target

    def step(self, action):
        self.round += 1

        rewards = [self._reward(action, self.send_target),
                   self._reward(action, self.recv_target)]
        done = (self.round >= self.num_rounds)

        action = action % self.num_targets
        send_diffs = torch.abs(action - self.send_target)
        send_diffs = torch.min(self.num_targets - send_diffs, send_diffs)
        recv_diffs = torch.abs(action - self.recv_target)
        recv_diffs = torch.min(self.num_targets - recv_diffs, recv_diffs)
        if done:
            self.send_diffs.append(send_diffs.mean().item())
            self.recv_diffs.append(recv_diffs.mean().item())

        self.round_info = {
            'round': self.round,
            'send_target': self.send_target[0].item(),
            'recv_target': self.recv_target[0].item(),
            'guess': action[0].item(),
            'send_reward': rewards[0][0].item(),
            'recv_reward': rewards[1][0].item(),
        }

        self.recv_target = self._generate_target()
        self.send_target = (self.recv_target + self.bias) % self.num_targets

        return self.send_target, rewards, done, [send_diffs, recv_diffs]

    def render(self, message=-1.0):
        print('--- round {} ---'.format(self.round_info['round']))
        print('targetS {:<2}   targetR {:2}'.format(
            self.round_info['send_target'],
            self.round_info['recv_target']))
        print('message {: <6.2f}   guess {:<5.2f}'.format(
            message,
            self.round_info['guess']))
        print('rewards   {: <5.2f}          {:<4.2f}'.format(
            self.round_info['send_reward'],
            self.round_info['recv_reward']))
