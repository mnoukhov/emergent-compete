import argparse
import os

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import game


@gin.configurable
def train(Sender, Recver, env, episodes, render, log):
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

            if render and e % render == 0:
                env.render(message=message[0].item())

        sender.log_reward()
        recver.log_reward()

        sender.update()
        recver.update()

        if log and e % log == 0:
            print(f'EPISODE {e}')
            print('REWRD   {:2.2f}     {:2.2f}'.format(sender.last('ep_reward'), recver.last('ep_reward')))
            print('LOSS    {:2.2f}     {:2.2f}'.format(sender.last('loss'), recver.last('loss')))
            print('GRADS   {:2.4f}     {:2.4f}'.format(sender.last('grad'), recver.last('grad')))
            print('')

    print('Game Over')
    x = list(range(episodes))
    plot(x, sender, recver, env)


@gin.configurable
def plot(x, sender, recver, env, savedir):
    if savedir is not None:
        savedir = os.path.join('experiments', savedir)
        os.makedirs(savedir, exist_ok=True)

    slogs = np.array(sender.logger['ep_reward'])
    rlogs = np.array(recver.logger['ep_reward'])
    avg_reward = (rlogs + slogs) / 2
    recv_advantage = rlogs - slogs
    plt.plot(x, running_mean(avg_reward, 100), 'b', label='avg reward')

    target_std_loss = env.observation_space.n / (12**0.5)
    bias_nash_loss = env.bias_space.low + (env.bias_space.range / 2)
    plt.plot(x, np.full_like(x, env._reward(target_std_loss), dtype=np.float), 'r', label='nocomm baseline')
    plt.plot(x, np.full_like(x, env._reward(bias_nash_loss), dtype=np.float), 'y', label='midbias baseline')
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

    print(gin.operative_config_str())
    if savedir:
        with open(f'{savedir}/config.gin','w') as f:
            f.write(gin.operative_config_str())
    # sround = np.array(sender.logger['round_reward'])
    # rround = np.array(recver.logger['round_reward'])
    # avg_round = (sround + rround) / 2
    # for r in range(env.num_rounds):
        # plt.plot(x, running_mean(avg_round[:,r]), label='avg_round-{}'.format(r))
        # # plt.plot(x, running_mean(sround[:,r]), label='sender-{}'.format(r))
        # # plt.plot(x, running_mean(rround[:,r]), label='recver-{}'.format(r))
    # plt.legend()
    # plt.show()
    # if savedir:
        # plt.savefig('{}/round_rewards.png'.format(savedir))


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
