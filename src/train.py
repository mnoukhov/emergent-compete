import argparse
import math
import os

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import src.agents
from src.agents import mode
import src.maddpg
import src.game
from src.utils import running_mean, circle_diff


@gin.configurable
def train(Sender, Recver, env, episodes, render_freq, log_freq, savedir, device):
    sender = Sender(num_actions=env.observation_space.n,
                    mode=mode.SENDER)
    recver = Recver(num_actions=env.action_space.n,
                    mode=mode.RECVER,
                    opponent=sender,
                    device=device)

    for e in range(episodes):
        target = env.reset()
        sender.reset()
        recver.reset()
        recv_reward = torch.zeros(env.batch_size)
        send_reward = torch.zeros(env.batch_size)
        prev_target = torch.zeros(env.batch_size)
        prev_message = torch.zeros(env.batch_size)
        prev_action = torch.zeros(env.batch_size)
        prev_recv_reward = torch.zeros(env.batch_size)
        prev_send_reward = torch.zeros(env.batch_size)
        prev2_target = torch.zeros(env.batch_size)
        prev2_message = torch.zeros(env.batch_size)
        prev2_action = torch.zeros(env.batch_size)
        prev2_recv_reward = torch.zeros(env.batch_size)
        prev2_send_reward = torch.zeros(env.batch_size)

        for r in range(env.num_rounds):
            batch_round = torch.full((env.batch_size,), r).long()
            one_hot_round = F.one_hot(batch_round, env.num_rounds).float()

            send_state = torch.stack([target,
                                      prev_target, prev_message, prev_send_reward,
                                      prev2_target, prev2_message, prev2_send_reward],
                                     dim=1)
            send_state = torch.cat((send_state, one_hot_round), dim=1)
            message = sender.action(send_state)

            recv_state = torch.stack([message,
                                      prev_message, prev_action, prev_recv_reward,
                                      prev2_message, prev2_action, prev2_recv_reward],
                                     dim=1)
            # recv_state = torch.cat((recv_state, one_hot_round), dim=1)
            # if r > 0:
                # recver.memory.push(prev_recv_state, message, action,
                                   # send_reward, recv_reward, recv_state)
            action = recver.action(recv_state)

            prev2_target = prev_target
            prev_target = target
            prev2_message = prev_message
            prev2_action = prev_action
            prev2_recv_reward = prev_recv_reward
            prev2_send_reward = prev_send_reward
            prev_message = message
            prev_action = action
            prev_recv_reward = recv_reward
            prev_send_reward = send_reward
            prev_send_state = send_state
            prev_recv_state = recv_state

            target, (send_reward, recv_reward), done, _, = env.step(action)

            sender.rewards.append(send_reward)
            recver.rewards.append(recv_reward)

            # sender.memory.push(prev_send_state, message, action,
                               # send_reward, recv_reward, send_state)

            if render_freq and e % render_freq == 0:
                env.render(message=message[0].item())

        # recv_state = torch.stack([torch.zeros(env.batch_size),
                                  # prev_message, prev_action, prev_recv_reward,
                                  # prev2_message, prev2_action, prev2_recv_reward],
                                 # dim=1)
        # one_hot_round = torch.zeros(env.batch_size, env.num_rounds)
        # recv_state = torch.cat((recv_state, one_hot_round), dim=1)
        # recver.memory.push(prev_recv_state, message, action,
                           # send_reward, recv_reward, recv_state)

        log_now = log_freq and (e % log_freq == 0)
        sender.update(ep=e, log=log_now)
        recver.update(ep=e, log=log_now)

        if log_now:
            print(f'EPISODE {e}')
            print('REWD    {:2.2f}     {:2.2f}'.format(sender.last('ep_reward'), recver.last('ep_reward')))
            print('LOSS    {:2.2f}     {:2.2f}'.format(sender.last('loss'), recver.last('loss')))
            print('DIFF    {:2.2f}     {:2.2f}'.format(env.send_diffs[-1], env.recv_diffs[-1]))
            print('')

    sender.writer.close()
    recver.writer.close()
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
    plt.plot(x, running_mean(avg_rew, 100), label='avg reward')
    plt.plot(x, running_mean(srew, 100), label='sender')
    plt.plot(x, running_mean(rrew, 100), label='recver')
    nocomm_loss = torch.tensor(env.observation_space.n / 4)
    nocomm_rew = env._reward(nocomm_loss)
    oneshot_loss = (env.bias_space.range) / 4
    oneshot_rew = env._reward(oneshot_loss)
    perfect_rew = env._reward(oneshot_loss / env.num_rounds)
    plt.axhline(nocomm_rew, label='nocomm baseline')
    plt.axhline(oneshot_rew, label='one-shot baseline')
    plt.axhline(perfect_rew, label='perfect agent')
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/rewards.png')
    plt.show()

    # REWARD PER ROUND
    sround = np.array(sender.logger['round_reward'])
    rround = np.array(recver.logger['round_reward'])
    avg_round = (sround + rround) / 2
    for r in range(env.num_rounds):
        # plt.plot(x, running_mean(avg_round[:,r]), label='avg_round-{}'.format(r))
        # plt.plot(x, running_mean(sround[:,r]), label='sender-{}'.format(r))
        plt.plot(x, running_mean(rround[:,r]), label='recver-{}'.format(r))
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/round.png')
    plt.show()

    # WEIGHTS
    # weights = np.array(recver.logger['weights'])
    # biases = np.array(recver.logger['biases'])[:,np.newaxis]
    # num_weights = weights.shape[1]
    # for i in range(num_weights):
        # plt.plot(x, running_mean(weights[:,i]), label=f'weight {i}')
        # plt.plot(x, running_mean(biases[:,i]), label=f'bias {i}')
    # plt.title('Weights')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/weights.png')
    # plt.show()

    # # ABS DIFF AT ROUND 5
    # plt.plot(x, running_mean(env.send_diffs), label='sender')
    # plt.plot(x, running_mean(env.recv_diffs), label='recver')
    # plt.title('Absolute diff at Round 5')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/diff.png')
    # plt.show()

    # OUTPUT FOR 20
    # if '20' in sender.logger:
        # plt.plot(x, sender.logger['20'], label='sender')
    # plt.plot(x, recver.logger['20'], label='recver')
    # plt.title('Output for Round 1 Input=20')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/20.png')
    # plt.show()

    # # ENTROPY
    # if 'entropy' in recver.logger:
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
    parser.add_argument('gin_param', nargs='+')
    args = parser.parse_args()
    gin_files = ['configs/{}'.format(gin_file) for gin_file in args.gin_file]
    gin.parse_config_files_and_bindings(gin_files, args.gin_param)
    train()
