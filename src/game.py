""" Iterated Sender Reciever Game

each round consists of three steps:
    1. agent 1 observes the target and bias
    2. agent 1 "acts", agent 2 observes action
    3. agent 2 acts determining reward

steps 3 and next round's step 1 are combined

"""
import gin
import gym
from gym.spaces import Discrete, Tuple
import numpy as np
import torch


@gin.configurable
class IteratedSenderRecver(gym.Env):
    def __init__(self,
                 batch_size,
                 num_rounds,
                 max_obs,
                 max_bias,
                 min_bias=0):
        self.num_rounds = num_rounds
        self.action_space = Discrete(max_obs - max_bias)
        self.bias_space = Discrete(max_bias)
        self.observation_space = Discrete(max_obs)
        self.batch_size = batch_size

    def _generate_bias(self):
        return torch.randint(self.bias_space.n,
                             size=(self.batch_size,1)).float()

    def _generate_target(self):
        return torch.randint(1, self.observation_space.n - self.bias_space.n,
                             size=(self.batch_size,1)).float()

    def reset(self):
        self.round = 0
        self.bias = self._generate_bias()
        self.target = self._generate_target()

        return self.target + self.bias

    def step(self, action):
        self.round += 1

        # rewards = [-torch.abs(action - self.target - self.bias),
                   # -torch.abs(action - self.target)]
        rewards = [-(action - self.target - self.bias)**2 / 100,
                   -(action - self.target)**2 / 100]
        done = (self.round >= self.num_rounds)

        self.round_info = {
            'round': self.round,
            'send_target': (self.target + self.bias)[0].item(),
            'recv_target': self.target[0].item(),
            'guess': action[0].item(),
            'send_loss': (-rewards[0][0].item() * 100) **0.5,
            'recv_loss': (-rewards[1][0].item() * 100) **0.5,
        }

        self.target = self._generate_target()
        obs = self.target + self.bias

        return obs, rewards, done

    def render(self, message=-1.0):
        print('--- round {} ---'.format(self.round_info['round']))
        print('targetS {:<2}   targetR {:2}'.format(
            self.round_info['send_target'],
            self.round_info['recv_target']))
        print('message {:<5.2f}    guess {:5.2f}'.format(
            message,
            self.round_info['guess']))
        print('losses  {:<4.2f}          {:4.2f}'.format(
            self.round_info['send_loss'],
            self.round_info['recv_loss']))
