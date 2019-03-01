import gin
import torch

from src.game import ISR
from src.agents import Human, NaiveQNet

@gin.configurable
def train(agents, env, episodes):
    sender = agents[0]
    recver = agents[1]

    next_target = env.reset()
    next_action = [0,0]
    for _ in range(episodes):
        done = False
        while not done:
            prev_target = next_target
            prev_action = next_action

            message = sender.step([prev_target[0]] + prev_action)
            guess = recver.step([message] + prev_action)
            next_target, rewards, done = env.step(guess)
            next_action = [message, guess]

            sender.update(env.round - 1,
                          [prev_target[0]] + prev_action,
                          message,
                          [next_target[0]] + next_action,
                          rewards[0])
            recver.update(env.round - 1,
                          [prev_target[1]] + prev_action,
                          guess,
                          [next_target[1]] + next_action,
                          rewards[0])

            env.render(message=message,
                       rewards=rewards)

    print('Game Over')

if __name__ == '__main__':
    gin.parse_config_file('configs/default.gin')
    train([NaiveQNet(), NaiveQNet()], ISR())
