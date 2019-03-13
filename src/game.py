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
                 num_rounds,
                 max_obs,
                 max_bias,
                 batch_size):
        self.num_rounds = num_rounds
        self.action_space = Discrete(max_obs)
        self.bias_space = Discrete(max_bias)
        self.observation_space = Discrete(max_obs)
        # self.batch_size = batch_size

    def _generate(self):
        bias = torch.randint(self.bias_space.n, size=())
        target = torch.randint(self.observation_space.n - bias, size=())

        return target.float(), bias.float()

    def reset(self):
        self.round = 0
        self.target, self.bias = self._generate()

        return self.target + self.bias

    def step(self, action):
        self.round += 1

        rewards = [-torch.abs(action - self.target - self.bias),
                   -torch.abs(action - self.target)]
        # rewards = [-(action - self.target)**2,
                   # -(action - self.target - self.bias)**2]
        done = (self.round >= self.num_rounds)

        self.round_info = [self.round,
                           self.target + self.bias,
                           self.target,
                           action]

        self.target, self.bias = self._generate()
        obs = self.target + self.bias

        return obs, rewards, done

    def render(self, message=None, rewards=None):
        print('--- round {} ---'.format(self.round_info[0]))
        print('targetS {:>2}   targetR {:>2}'.format(self.round_info[1].item(),
                                                     self.round_info[2].item()))
        if message is not None:
            print('message {:>2.2f}    guess {:>2.2f}'.format(message.item(),
                                                              self.round_info[3].item()))
        if rewards is not None:
            print('rewards{:>3.2f}         {:>3.2f}'.format(rewards[0].item(),
                                                            rewards[1].item()))
