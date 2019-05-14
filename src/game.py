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

from src.utils import circle_diff
# import math


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
        return self.bias_space.low + torch.rand(size=(self.batch_size,)) * self.bias_space.range

    def _generate_target(self):
        return torch.rand(size=(self.batch_size,)) * self.action_space.n

    def _reward(self, pred, target=None):
        if target is None:
            target = torch.tensor(0.)
        dist = torch.abs(circle_diff(pred, target, self.num_targets))
        # dist = torch.min(diff, self.num_targets - diff)
        return (1 - (2 * dist / self.num_targets))

    # def _reward(self, pred, target=None):
        # if target is None:
            # target = torch.tensor(0.)
        # diff = torch.abs(pred - target)
        # dist = torch.min(diff, self.num_targets - diff)
        # return 1 - 2 * dist / self.num_targets

    # def _reward(self, pred, target=None):
        # if target is None:
            # target = torch.tensor(0.)
        # norm_dist = (pred - target) / self.num_targets
        # radian_dist = 2*math.pi*norm_dist
        # return 0.5*(1 + torch.cos(radian_dist))

    # def _reward(self, pred, target=None):
        # if target is None:
            # target = torch.tensor(0.)
        # norm_dist = (pred - target) / self.num_targets
        # return 1-norm_dist**2

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

        send_diffs = circle_diff(self.send_target, action, self.num_targets)
        recv_diffs = circle_diff(self.recv_target, action, self.num_targets)
        if done:
            self.send_diffs.append(send_diffs.abs().mean().item())
            self.recv_diffs.append(recv_diffs.abs().mean().item())

        self.round_info = {
            'round': self.round,
            'send_target': self.send_target[0].item(),
            'recv_target': self.recv_target[0].item(),
            'guess': action[0].item(),
            'send_reward': rewards[0][0].item(),
            'recv_reward': rewards[1][0].item(),
            'send_diff': send_diffs[0].item(),
            'recv_diff': recv_diffs[0].item(),
        }

        self.recv_target = self._generate_target()
        self.send_target = (self.recv_target + self.bias) % self.num_targets

        return self.send_target, rewards, done, [send_diffs, recv_diffs]

    def render(self, message=-1.0):
        print('--- round {} ---'.format(self.round_info['round']))
        print('targetS {:<5.2f}   targetR {:<5.2f}'.format(
            self.round_info['send_target'],
            self.round_info['recv_target']))
        print('message {:<5.2f}   guess   {:<4.2f}'.format(
            message,
            self.round_info['guess']))
        print('rewards  {: <5.2f}          {:<4.2f}'.format(
            self.round_info['send_reward'],
            self.round_info['recv_reward']))
        if self.round_info['send_diff']:
            print('diffs    {: <5.2f}          {:<4.2f}'.format(
                self.round_info['send_diff'],
                self.round_info['recv_diff']))
