import argparse
import os

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import game


@gin.configurable
def train(sender, recver, env, episodes, render):
    for e in range(episodes):
        target = env.reset()
        sender.reset()
        recver.reset()
        prev_action = [torch.zeros(env.batch_size, 1),
                       torch.zeros(env.batch_size, 1)]
        done = False
        while not done:
            message = sender.action([target] + prev_action)
            guess = recver.action([message] + prev_action)

            target, rewards, done = env.step(guess)
            prev_action = [message, guess]

            sender.rewards.append(rewards[0])
            recver.rewards.append(rewards[1])

            if e % render == 0:
                env.render(message=message, rewards=rewards)


        sender.logs.append(sender.avg_reward())
        recver.logs.append(recver.avg_reward())

        sender.update(retain_graph=True)
        recver.update()

        if e % render == 0:
            print('EPISODE ', e)
            print('AVG REW  {:2.2f}     {:2.2f}'.format(sender.avg_reward(), recver.avg_reward()))
            print('')

    print('Game Over')
    x = list(range(episodes))
    slogs = np.array(sender.logs)
    rlogs = np.array(recver.logs)
    plot(x, slogs, rlogs, env.bias_space.n)
    print(gin.operative_config_str())

@gin.configurable
def plot(x, slogs, rlogs, max_bias, savedir):
    if savedir is not None:
        savedir = os.path.join('experiments', savedir)
        os.makedirs(savedir, exist_ok=True)

    avg_reward = (rlogs + slogs) / 2
    recv_advantage = rlogs - slogs
    plt.plot(x, running_mean(avg_reward, 100), 'b', label='avg reward')
    plt.plot(x, running_mean(recv_advantage, 100), 'g', label='recv advantage')
    plt.plot(x, np.full_like(x, -max_bias // 2), 'r', label='avg bias')
    plt.legend()
    plt.show()
    if savedir:
        plt.savefig('{}/advantage.png'.format(savedir))
    plt.plot(x, running_mean(slogs, 100), 'r', label='sender')
    plt.plot(x, running_mean(rlogs, 100), 'b', label='recver')
    plt.legend()
    plt.show()
    if savedir:
        plt.savefig('{}/rewards.png'.format(savedir))


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    afterN = (cumsum[N:] - cumsum[:-N]) / float(N)
    beforeN = cumsum[1:N] / np.arange(1, N)
    return np.concatenate([beforeN, afterN])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', nargs='+', default=['default.gin'])
    parser.add_argument('--gin_param', nargs='+')
    args = parser.parse_args()

    gin_files = ['configs/{}'.format(gin_file) for gin_file in args.gin_file]
    gin.parse_config_files_and_bindings(gin_files, args.gin_param)
    train()
