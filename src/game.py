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


@gin.configurable
class ISR(gym.Env):
    NAME = 'ISR'

    def __init__(self,
                 num_rounds=5,
                 obs_range=100,
                 bias_range=20):
        self.round = None
        self.targets = None
        self.bias = None
        self.last_actions = None
        self.action_space = Discrete(obs_range)
        self.observation_space = Tuple(Discrete(obs_range),
                                       Discrete(bias_range))

    def reset(self):
        self.round = 0
        self.round_info = [targets[0], None, None]
        self.target = self.observation_space[1].sample()
        self.bias = self.observation_space[0][1].sample()

        return (self.target, self.bias)

    def step(self, action):
        rewards = [(action - self.target)**2,
                   (action - self.target - self.bias)**2]
        done = (self.round >= self.num_rounds)

        self.round += 1
        self.target = self.observation_space[1].sample()
        self.bias = self.observation_space[0][1].sample()
        self.guess = action
        obs = (self.target, self.bias)

        return obs, rewards, done

    def render(self, message=None, mode='human', close=False):
        print('target {} message {} guess {}'.format(self.target, message, self.guess))

