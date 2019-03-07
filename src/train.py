import argparse

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import game


@gin.configurable
def train(sender, recver, env, episodes, render=1000):
    next_target = env.reset()
    next_action = [0,0]
    logs = [[],[]]
    for e in range(episodes):
        done = False
        logs[0].append(0)
        logs[1].append(0)
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

            logs[0][-1] += rewards[0].item()
            logs[0][-1] += rewards[1].item()


        if e % render == 0:
            print('EPISODE ', e)
            env.render(message=message,
                        rewards=rewards)

    print('Game Over')
    x = list(range(episodes - 99))
    plt.plot(x, running_mean(logs[0], 100), 'b',
             x, running_mean(logs[0], 100), 'g')
    plt.show()

    print(gin.operative_config_str())

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', nargs='+', default=['default.gin'])
    parser.add_argument('--gin_param', nargs='+')
    args = parser.parse_args()

    gin_files = ['configs/{}'.format(gin_file) for gin_file in args.gin_file]
    gin.parse_config_files_and_bindings(gin_files, args.gin_param)
    train()
