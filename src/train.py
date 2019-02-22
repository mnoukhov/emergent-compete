import gin
import torch

from .game import ISR
from .agents import Human

@gin.configurable
def train(agents, env, episodes):
    sender = agents[0]
    recver = agents[1]

    target, bias = env.reset()
    for _ in range(episodes):
        message = sender(target, bias)
        guess = recver(message)
        target, bias = env.step(guess)
        env.render(message=message)

    print('Game Over')

if __name__ == '__main__':
    gin.parse_config_file('configs/default.gin')
    train([Human(0), Human(1)], ISR())
