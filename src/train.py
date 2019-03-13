import argparse

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import game


@gin.configurable
def train(sender, recver, env, episodes, render=1000):
    for e in range(episodes):
        done = False
        target = env.reset()
        sender.reset()
        recver.reset()
        prev_action = [torch.tensor(0.), torch.tensor(0.)]
        while not done:
            message = sender.action([target] + prev_action)
            guess = recver.action([message] + prev_action)

            target, rewards, done = env.step(guess)
            prev_action = [message, guess]

            sender.rewards.append(rewards[0])
            recver.rewards.append(rewards[1])

        sender.logs.append(sum(sender.rewards) / env.round)
        recver.logs.append(sum(recver.rewards) / env.round)

        sender.update()
        recver.update()

        if e % render == 0:
            print('EPISODE ', e)
            avg_rewards = [sender.logs[-1], recver.logs[-1]]
            env.render(message=message, rewards=rewards)
            print('')

    print('Game Over')
    x = list(range(episodes - 99))
    # plt.plot(x, running_mean(logs[0], 100), 'b',
             # x, running_mean(logs[1], 100), 'g')
    slogs = np.array(sender.logs)
    rlogs = np.array(recver.logs)

    avg_reward = (rlogs + slogs) / 2
    recv_advantage = rlogs - slogs
    plt.plot(x, running_mean(avg_reward, 100), 'b', label='avg reward')
    plt.plot(x, running_mean(recv_advantage, 100), 'g', label='recv advantage')
    plt.legend()
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
