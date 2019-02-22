import gin
import torch

from src.game import ISR
from src.agents import Human

@gin.configurable
def train(agents, env, episodes):
    sender = agents[0]
    recver = agents[1]

    obs = env.reset()
    for _ in range(episodes):
        done = False
        while not done:
            message = sender(obs)
            guess = recver(message)
            obs, rewards, done = env.step(guess)

            env.render(message=message,
                       rewards=rewards)

    print('Game Over')

if __name__ == '__main__':
    gin.parse_config_file('configs/default.gin')
    train([Human(0), Human(1)], ISR())
