import argparse
import os

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import game
from utils import running_mean


@gin.configurable
def train(Sender, Recver, env, episodes, render, log, savedir):
    sender = Sender(mode=0, num_actions=env.observation_space.n)
    recver = Recver(mode=1, num_actions=env.action_space.n)

    for e in range(episodes):
        target = env.reset()
        sender.reset()
        recver.reset()
        prev_target = torch.zeros(env.batch_size, 1)
        prev_message = torch.zeros(env.batch_size, 1)
        prev_guess = torch.zeros(env.batch_size, 1)
        prev_send_reward = torch.zeros(env.batch_size, 1)
        prev_recv_reward = torch.zeros(env.batch_size, 1)

        done = False
        while not done:
            message = sender.action([target, prev_target, prev_message, prev_send_reward])
            guess = recver.action([message, prev_message, prev_guess, prev_recv_reward])

            target, rewards, done, diffs = env.step(guess)
            prev_target = target.detach()
            prev_message = message.detach()
            prev_guess = guess.detach()

            sender.rewards.append(rewards[0])
            recver.rewards.append(rewards[1])
            prev_send_reward = diffs[0].detach()
            prev_recv_reward = diffs[1].detach()

            if render and e % render == 0:
                env.render(message=message[0].item())

        sender.update()
        recver.update()

        if log and e % log == 0:
            print(f'EPISODE {e}')
            print('REWD    {:2.2f}     {:2.2f}'.format(sender.last('ep_reward'), recver.last('ep_reward')))
            print('LOSS    {:2.2f}     {:2.2f}'.format(sender.last('loss'), recver.last('loss')))
            print('DIFF    {:2.2f}     {:2.2f}'.format(env.send_diffs[-1], env.recv_diffs[-1]))
            print('')

    print('Game Over')
    x = list(range(episodes))
    plot_and_save(x, sender, recver, env, savedir)


def plot_and_save(x, sender, recver, env, savedir):
    if savedir is not None:
        savedir = os.path.join('results', savedir)
        os.makedirs(savedir, exist_ok=True)

    # REWARDS
    srew = np.array(sender.logger['ep_reward'])
    rrew = np.array(recver.logger['ep_reward'])
    avg_rew = (rrew + srew) / 2
    nocomm_loss = torch.tensor(env.observation_space.n / 4)
    nocomm_rew = env._reward(nocomm_loss)
    midbias_loss = env.bias_space.low + (env.bias_space.range / 2)
    midbias_rew = env._reward(midbias_loss)
    plt.plot(x, running_mean(avg_rew, 100), label='avg reward')
    plt.plot(x, running_mean(srew, 100), label='sender')
    plt.plot(x, running_mean(rrew, 100), label='recver')
    plt.axhline(nocomm_rew, label='nocomm baseline')
    plt.axhline(midbias_rew, label='midbias baseline')
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/rewards.png')
    plt.show()

    # REWARD PER ROUND
    # sround = np.array(sender.logger['round_reward'])
    # rround = np.array(recver.logger['round_reward'])
    # avg_round = (sround + rround) / 2
    # for r in range(env.num_rounds):
        # # plt.plot(x, running_mean(avg_round[:,r]), label='avg_round-{}'.format(r))
        # # plt.plot(x, running_mean(sround[:,r]), label='sender-{}'.format(r))
        # plt.plot(x, running_mean(rround[:,r]), label='recver-{}'.format(r))
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/round.png')
    # plt.show()

    # ABS DIFF AT ROUND 5
    plt.plot(x, running_mean(env.send_diffs), label='sender')
    plt.plot(x, running_mean(env.recv_diffs), label='recver')
    plt.title('Absolute diff at Round 5')
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/diff.png')
    plt.show()

    # GRADS
    # plt.plot(x, running_mean(sender.logger['preact grad']), label='sender-grad')
    plt.plot(x, running_mean(recver.logger['preact grad']), label='recver-norm')
    # plt.plot(x, running_mean(sender.logger['grad var']), label='sender-var')
    plt.plot(x, running_mean(recver.logger['grad var']), label='recver-var')
    plt.plot(x, running_mean(recver.logger['grad mean']), label='recver-mean')
    plt.legend()
    plt.show()

    # OUTPUT FOR 20
    # plt.plot(x, sender.logger['20'], label='sender')
    # plt.plot(x, recver.logger['20'], label='recver')
    # plt.title('Output for Input=20')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/20.png')
    # plt.show()

    # ENTROPY
    # plt.plot(x, recver.logger['entropy'], label='entropy')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/entropy.png')
    # plt.show()


    print(gin.operative_config_str())
    if savedir:
        with open(f'{savedir}/config.gin','w') as f:
            f.write(gin.operative_config_str())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', nargs='+', default=['default.gin'])
    parser.add_argument('--gin_param', nargs='+')
    args = parser.parse_args()

    gin_files = ['configs/{}'.format(gin_file) for gin_file in args.gin_file]
    gin.parse_config_files_and_bindings(gin_files, args.gin_param)
    train()
