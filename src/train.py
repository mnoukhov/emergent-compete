import argparse
import os

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import game


@gin.configurable
def train(Sender, Recver, env, episodes, render):
    sender = Sender(mode=0, n=env.observation_space.n)
    recver = Recver(mode=1, n=env.action_space.n)

    for e in range(episodes):
        target = env.reset()
        sender.reset()
        recver.reset()
        prev_action = [torch.zeros(env.batch_size,1),
                       torch.zeros(env.batch_size,1)]
        done = False

        while not done:
            message = sender.action([target] + prev_action)
            guess = recver.action([message] + prev_action)

            target, rewards, done = env.step(guess)
            prev_action = [message.detach(), guess.detach()]

            sender.rewards.append(rewards[0])
            recver.rewards.append(rewards[1])

            if e % render == 0:
                env.render(message=message[0].item())

        sender.log_reward()
        recver.log_reward()

        sender.update()
        recver.update()

        if e % render == 0:
            print('EPISODE ', e)
            print('AVG REW  {:2.2f}     {:2.2f}'.format(sender.avg_reward(), recver.avg_reward()))
            print('')

    print('Game Over')
    x = list(range(episodes))
    plot(x, sender, recver, env)
    print(gin.operative_config_str())


@gin.configurable
def plot(x, sender, recver, env, savedir):
    slogs = np.array(sender.logs)
    rlogs = np.array(recver.logs)
    target_var = env.observation_space.n**2 / 12
    bias_var = (env.bias_space.low + env.bias_space.range / 2)**2
    if savedir is not None:
        savedir = os.path.join('experiments', savedir)
        os.makedirs(savedir, exist_ok=True)

    avg_reward = (rlogs + slogs) / 2
    recv_advantage = rlogs - slogs
    plt.plot(x, running_mean(avg_reward, 100), 'b', label='avg reward')
    plt.plot(x, np.full_like(x, -target_var), 'r', label='nocomm baseline')
    plt.plot(x, np.full_like(x, -bias_var), 'y', label='midbias baseline')
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

    sround = np.array(sender.round_logs)
    rround = np.array(recver.round_logs)
    avg_round = (sround + rround) / 2
    for r in range(env.num_rounds):
        plt.plot(x, running_mean(avg_round[:,r]), label='avg_round-{}'.format(r))
        # plt.plot(x, running_mean(sround[:,r]), label='sender-{}'.format(r))
        # plt.plot(x, running_mean(rround[:,r]), label='recver-{}'.format(r))
    plt.legend()
    plt.show()
    if savedir:
        plt.savefig('{}/round_rewards.png'.format(savedir))


def running_mean(x, N=100):
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
