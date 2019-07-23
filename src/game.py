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


class DiscreteRange(Discrete):
    def __init__(self, low, high=None):
        self.low = low
        self.high = high if high is not None else low
        self.range = self.high - self.low
        super().__init__(self.range)

    def sample(self):
        return super().sample() + self.low

    def contains(self, x):
        return super().contains(x - self.low)


@gin.configurable
class ISR(gym.Env):
    def __init__(self,
                 batch_size,
                 num_rounds,
                 num_targets,
                 min_bias,
                 max_bias=None):
        self.num_rounds = num_rounds
        self.num_targets = torch.tensor(num_targets, dtype=torch.float)
        self.action_space = Discrete(num_targets)
        self.observation_space = Discrete(num_targets)
        self.bias_space = DiscreteRange(min_bias, max_bias)
        self.batch_size = batch_size

        self.send_diffs = []
        self.recv_diffs = []

    def _generate_bias(self):
        return torch.randint(low=self.bias_space.low,
                             high=self.bias_space.high+1,
                             size=(1, self.batch_size),
                             dtype=torch.float)

    def _generate_targets(self):
        return torch.randint(high=self.action_space.n,
                             size=(self.num_rounds, self.batch_size),
                             dtype=torch.float)


    def _reward(self, pred, target=None):
        if target is None:
            target = torch.tensor(0.)
        return -circle_diff(pred, target, self.num_targets)

    def reset(self):
        self.round = 0
        self.bias = self._generate_bias()
        self.recv_targets = self._generate_targets()
        self.send_targets = (self.recv_targets + self.bias) % self.num_targets

        return self.send_targets[0]

    def step(self, message, action):
        action = action.cpu().clamp(0, self.num_targets)
        send_target = self.send_targets[self.round]
        recv_target = self.recv_targets[self.round]
        rewards = [self._reward(action, send_target),
                   self._reward(action, recv_target)]
        done = (self.round >= self.num_rounds - 1)

        if done:
            self.send_diffs.append(rewards[0].mean().item())
            self.recv_diffs.append(rewards[1].mean().item())

        self.round_info = {
            'round': self.round,
            'send_target': send_target[0].item(),
            'recv_target': recv_target[0].item(),
            'message': message[0].tolist(),
            'action': action[0].item(),
            'send_reward': rewards[0][0].item(),
            'recv_reward': rewards[1][0].item(),
            # 'send_diff': send_diffs[0].item(),
            # 'recv_diff': recv_diffs[0].item(),
        }

        self.round += 1
        next_target = self.send_targets[self.round] if not done else None
        return next_target, rewards, done

    def render(self):
        message = self.round_info['message']
        if isinstance(message, list):
            message_str = " ".join(f'{m:5.2f}' for m in message)
        else:
            message_str = f'{message:5.2f}'

        print('--- round {} ---'.format(self.round_info['round']))
        print('targetS {:<5.2f}   targetR {:<5.2f}'.format(
            self.round_info['send_target'],
            self.round_info['recv_target']))
        print('message {}   action  {:<4.2f}'.format(
            message_str,
            self.round_info['action']))
        print('rewards  {: <5.2f}          {:<4.2f}'.format(
            self.round_info['send_reward'],
            self.round_info['recv_reward']))
        # print('diffs    {: <5.2f}          {:<4.2f}'.format(
            # self.round_info['send_diff'],
            # self.round_info['recv_diff']))
