import json

import gin
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def log_grad_norm(log_list):
    def grad_norm_hook(module, grad_input, grad_output):
        __import__('pdb').set_trace()
        if not isinstance(grad_input, tuple):
            grad_input = [grad_input]
        log_list.append(total_norm(grad_input))

    return grad_norm_hook


def total_norm(iterable):
    return sum([x.norm(2).detach()**2 for x in iterable])**0.5


def discount_return(rewards, gamma):
    R = 0
    discounted = []
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)

    return discounted


def running_mean(x, N=100):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    afterN = (cumsum[N:] - cumsum[:-N]) / float(N)
    beforeN = cumsum[1:N] / np.arange(1, N)
    return np.concatenate([beforeN, afterN])


def soft_update(src, trg, tau):
    for trg_param, src_param in zip(trg.parameters(), src.parameters()):
        trg_param.data.copy_((1 - tau) * trg_param.data + tau * src_param.data)


def hard_update(source, target):
    for trg_param, src_param in zip(trg.parameters(), src.parameters()):
        trg_param.data.copy_(src_param.data)


def circle_diff(x, y, circ):
    # abs(x - y) on a circle of circumference circ
    # where x,y < circ
    diff = torch.abs(x - y)
    diff[diff > circ/2] = circ - diff[diff > circ/2]
    return diff


def load(sender, recver, loaddir):
    model_save = torch.load(f'{loaddir}/models.save')
    sender.load_state_dict(model_save['sender'])
    recver.load_state_dict(model_save['recver'])


def plot(logpath, env, savedir):
    sns.set()

    with open(logpath, 'r') as logfile:
        logs = pd.read_json(logfile)

    episode = logs['episode']
    sender = pd.DataFrame([s for s in logs['sender']], index=episode)
    recver = pd.DataFrame([s for s in logs['recver']], index=episode)

            # for line in logfile:
            # logline = json.loads(line)
            # episode.append(logline['episode'])
            # for key in sender.keys():
                # sender[key].append(logline['sender'][key])
            # for key in recver.keys():
                # recver[key].append(logline['recver'][key])

    # REWARDS

    sns.lineplot(data=sender, x=sender.index, y="reward", label="sender")
    sns.lineplot(data=recver, x=recver.index, y="reward", label="recver")
    sns.lineplot(x=sender.index, y=(sender['reward'] + recver['reward'])/2, label='avg')
#     plt.plot(x, running_mean(avg_rew, 100), label='avg reward')
#     plt.plot(x, running_mean(srew, 100), label='sender')
#     plt.plot(x, running_mean(rrew, 100), label='recver')
    nocomm_loss = torch.tensor(env.observation_space.n / 4)
    nocomm_rew = env._reward(nocomm_loss)
    oneshot_loss = (env.bias_space.range) / 4
    oneshot_rew = env._reward(oneshot_loss)
#     perfect_rew = env._reward(oneshot_loss / env.num_rounds)
    plt.axhline(nocomm_rew, label='nocomm baseline')
    plt.axhline(oneshot_rew, label='one-shot baseline')
#     plt.axhline(perfect_rew, label='perfect agent')
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/rewards.png')
    plt.show()
    plt.clf()

    # REWARD PER ROUND
#     sround = np.array(sender['round_reward'])
#     rround = np.array(recver['round_reward'])
#     avg_round = (sround + rround) / 2
#     for r in range(env.num_rounds):
#         # plt.plot(x, running_mean(avg_round[:,r]), label='avg_round-{}'.format(r))
#         # plt.plot(x, running_mean(sround[:,r]), label='sender-{}'.format(r))
#         plt.plot(x, running_mean(rround[:,r]), label='recver-{}'.format(r))
#     plt.axhline(oneshot_rew, label='one-shot baseline')
#     plt.legend()
#     if savedir:
#         plt.savefig(f'{savedir}/round.png')
#     plt.show()
#     plt.clf()

    # WEIGHTS
    # weights = np.array(recver['weights'])
    # biases = np.array(recver['biases'])[:,np.newaxis]
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

    # Sender and Recver Output Samples
    # for sample in ["0", "15", "30"]:
        # sns.lineplot(data=sender, x=sender.index, y=sample, label=sample)

    # plt.title('Sender output samples')
    # plt.ylabel('')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/send_samples.png')
    # plt.show()
    # plt.clf()

    # for sample in ["0", "15", "30"]:
        # sns.lineplot(data=recver, x=recver.index, y=sample, label=sample)
    # plt.title('Recver output samples')
    # plt.legend()
    # if savedir:
        # plt.savefig(f'{savedir}/recv_samples.png')
    # plt.show()
    # plt.clf()

    # # ENTROPY
    # if 'entropy' in recver:
        # plt.plot(x, recver['entropy'], label='entropy')
        # plt.legend()
        # if savedir:
            # plt.savefig(f'{savedir}/entropy.png')
        # plt.show()
