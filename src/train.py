import argparse
import json
import math
import os

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.agents
from src.agents import mode
import src.maddpg
from src.game import ISR
from src.utils import save, plot


@gin.configurable
def train(Sender, Recver, episodes, vocab_size,
          render_freq, log_freq, print_freq,
          savedir, loaddir, device):
    env = ISR()
    sender = Sender(input_size=1,
                    output_size=vocab_size,
                    mode=mode.SENDER,
                    device=device)
    recver = Recver(input_size=vocab_size,
                    output_size=1,
                    mode=mode.RECVER,
                    device=device)

    from collections import OrderedDict
    testmodel = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, vocab_size),
        nn.Linear(vocab_size, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 1))

    if savedir is not None:
        savedir = os.path.join('results', savedir)
        os.makedirs(savedir, exist_ok=True)
        logpath = f'{savedir}/logs.json'
        logfile = open(logpath, 'w')
        logfile.write('[ \n')
    else:
        logfile = None

    if loaddir is not None:
        loaddir = os.path.join('results', loaddir)
        if os.path.exists(f'{loaddir}/models.save'):
            load(sender, recver, loaddir)

    for e in range(episodes):
        target = env.reset().to(device)
        send_rewards_list = []
        recv_rewards_list = []

        # message = torch.zeros(env.batch_size, device=device)
        # action = torch.zeros(env.batch_size, device=device)
        # recv_reward = torch.zeros(env.batch_size, device=device)
        # send_reward = torch.zeros(env.batch_size, device=device)
        # prev_recv_reward = torch.zeros(env.batch_size, device=device)
        # prev_send_reward = torch.zeros(env.batch_size, device=device)
        # prev_target = torch.zeros(env.batch_size, device=device)
        # prev_message = torch.zeros(env.batch_size, device=device)
        # prev_action = torch.zeros(env.batch_size, device=device)
        # prev2_target = torch.zeros(env.batch_size, device=device)
        # send_state = None
        # recv_state = None
        # testmodel[:5].load_state_dict(sender.policy.state_dict())
        # recv_state_dict = recver.policy.state_dict()
        # test_state_dict = OrderedDict()
        # for key, val in recv_state_dict.items():
            # key_num = int(key[0])
            # recv_key = f'{key_num + 5}' + key[1:]
            # test_state_dict[recv_key] = val
        # testmodel[5:].load_state_dict(test_state_dict)
        # testoptim = torch.optim.Adam(testmodel.parameters())
        # testrewards = []

        for r in range(env.num_rounds):
            # prev2_message = prev_message.detach()
            # prev2_action = prev_action.detach()
            # prev_message = message.detach()
            # prev_action = action.detach()
            # prev_send_state = send_state
            # prev_recv_state = recv_state

            # send_state = torch.stack([target, prev_target, prev_message, send_reward,
                                      # prev2_target, prev2_message, prev_send_reward],
                                     # dim=1)
            send_state = target.unsqueeze(1)
            message = sender.action(send_state)

            # recv_state = torch.stack([message, prev_message, prev_action, recv_reward,
                                      # prev2_message, prev2_action, prev_recv_reward],
                                     # dim=1)
            recv_state = message.unsqueeze(1)
            action = recver.action(recv_state)

            # if r > 0 and hasattr(sender, 'memory'):
                # sender.memory.push(prev_send_state.cpu(), prev_message.cpu(), prev_action.cpu(),
                                    # send_reward.cpu(), recv_reward.cpu(), send_state.cpu())
            # if r > 0 and hasattr(recver, 'memory'):
                # recver.memory.push(prev_recv_state.cpu(), prev_message.cpu(), prev_action.cpu(),
                                    # send_reward.cpu(), recv_reward.cpu(), recv_state.cpu())

            # prev2_target = prev_target
            # prev_target = target
            # prev_recv_reward = recv_reward.detach()
            # prev_send_reward = send_reward.detach()

            # testaction = testmodel(send_state).squeeze()
            # testrewards.append(env._reward(testaction, env.send_targets[env.round]))

            target, (send_reward, recv_reward), done, = env.step(message, action)
            target = target.to(device) if target is not None else None

            send_rewards_list.append(send_reward)
            recv_rewards_list.append(recv_reward)

            if render_freq and e % render_freq == 0:
                env.render()

        send_rewards = torch.stack(send_rewards_list, dim=1).to(device)
        recv_rewards = torch.stack(recv_rewards_list, dim=1).to(device)

        # sender MUST be update before recver
        send_loss, send_logs = sender.update(e, send_rewards, retain_graph=True)
        recv_loss, recv_logs = recver.update(e, recv_rewards)

        # testrewards = torch.stack(testrewards)
        # testloss = - testrewards.mean()
        # testoptim.zero_grad()
        # testloss.backward()
        # test_sgrad = testmodel[4]._parameters['weight'].grad
        # test_rgrad = testmodel[9]._parameters['weight'].grad
        # testoptim.step()
        # print(f'send grad correct {torch.isclose(test_sgrad, send_grad).byte().all()}')
        # print(f'recv grad correct {torch.isclose(test_rgrad, recv_grad).byte().all()}')


        if print_freq and (e % print_freq == 0):
            print(f'EPISODE {e}')
            print('REWD    {:2.2f}     {:2.2f}'.format(send_logs['reward'], recv_logs['reward']))
            print('LOSS    {:2.2f}     {:2.2f}'.format(send_logs['loss'], recv_logs['loss']))
            print('DIFF    {:2.2f}     {:2.2f}'.format(env.send_diffs[-1], env.recv_diffs[-1]))
            print('')

        if logfile and (e % log_freq == 0):
            dump = {'episode': e,
                    'sender': send_logs,
                    'recver': recv_logs}
            json.dump(dump, logfile, indent=2)
            logfile.write(',\n')

    print('Game Over')

    if savedir:
        dump = {'episode': e,
                'sender': send_logs,
                'recver': recv_logs}
        json.dump(dump, logfile, indent=2)
        logfile.write('\n]')
        logfile.close()
        save(sender, recver, env, savedir)

        plot(logpath, env, savedir)

    print(gin.operative_config_str())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', '-f', nargs='+', default=['default.gin'])
    parser.add_argument('--gin_param', '-p', nargs='+')
    args = parser.parse_args()
    gin_files = ['base.gin'] + args.gin_file
    gin_paths = [f'configs/{gin_file}' for gin_file in gin_files]

    # change device to torch.device
    gin.config.register_finalize_hook(
        lambda config: config[('', '__main__.train')].update({'device': torch.device(config[('', '__main__.train')]['device'])}))
    gin.parse_config_files_and_bindings(gin_paths, args.gin_param)
    train()

    # gin.clear_config()
    # gin.config._REGISTRY._selector_map.pop('__main__.train')
