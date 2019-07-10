import gin
import torch
import matplotlib.pyplot as plt
import numpy as np


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
    # x - y on a circle
    diff = x - y
    diff[diff > circ/2] -= circ
    diff[diff < -circ/2] += circ
    return torch.abs(diff)


def save(sender, recver, env, savedir):
    with open(f'{savedir}/config.gin','w') as f:
        f.write(gin.operative_config_str())

    torch.save({
        'sender': sender.state_dict(),
        'recver': recver.state_dict(),
    }, f'{savedir}/models.save')


def plot(x, sender, recver, env, savedir):
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
    plt.clf()

    # REWARD PER ROUND
    sround = np.array(sender.logger['round_reward'])
    rround = np.array(recver.logger['round_reward'])
    avg_round = (sround + rround) / 2
    for r in range(env.num_rounds):
        # plt.plot(x, running_mean(avg_round[:,r]), label='avg_round-{}'.format(r))
        # plt.plot(x, running_mean(sround[:,r]), label='sender-{}'.format(r))
        plt.plot(x, running_mean(rround[:,r]), label='recver-{}'.format(r))
    plt.axhline(oneshot_rew, label='one-shot baseline')
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/round.png')
    plt.show()
    plt.clf()

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
    plt.plot(x, recver.logger['20'], label='recver')
    plt.axhline(oneshot_rew, label='one-shot baseline')
    plt.title('Output for Input=20')
    plt.legend()
    if savedir:
        plt.savefig(f'{savedir}/20.png')
    plt.show()

    # # ENTROPY
    # if 'entropy' in recver.logger:
        # plt.plot(x, recver.logger['entropy'], label='entropy')
        # plt.legend()
        # if savedir:
            # plt.savefig(f'{savedir}/entropy.png')
        # plt.show()
