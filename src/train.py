import argparse
import json
import os

import gin
import torch
import torch.nn as nn

from src.agents import mode
from src.game import ISR
from src.utils import load, plot
import src.lola

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
                    output_range=env.num_targets,
                    mode=mode.RECVER,
                    device=device)

    sender.other = recver
    sender.env = env

    if savedir is not None:
        savedir = os.path.join('results', savedir)
        os.makedirs(savedir, exist_ok=True)

        with open(f'{savedir}/config.gin', 'w') as f:
            f.write(gin.operative_config_str())

        logfile = open(f'{savedir}/logs.json', 'w')
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

        for _ in range(env.num_rounds):
            # prev2_message = prev_message.detach()
            # prev2_action = prev_action.detach()
            # prev_message = message.detach()
            # prev_action = action.detach()
            # prev_send_state = send_state
            # prev_recv_state = recv_state

            # send_state = torch.stack([target, prev_target, prev_message, send_reward,
                                      # prev2_target, prev2_message, prev_send_reward],
                                     # dim=1)
            message = sender(target.unsqueeze(1))

            # recv_state = torch.stack([message, prev_message, prev_action, recv_reward,
                                      # prev2_message, prev2_action, prev_recv_reward],
                                     # dim=1)
            action = recver(message).squeeze()

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

            target, (send_reward, recv_reward), done, = env.step(message, action)
            target = target.to(device) if target is not None else None

            send_rewards_list.append(send_reward)
            recv_rewards_list.append(recv_reward)

            if render_freq and e % render_freq == 0:
                env.render()

        send_rewards = torch.stack(send_rewards_list, dim=1).to(device)
        recv_rewards = torch.stack(recv_rewards_list, dim=1).to(device)

        send_loss, send_logs = sender.loss(send_rewards)
        recv_loss, recv_logs = recver.loss(recv_rewards)

        # sender MUST be update before recver
        sender.optimizer.zero_grad()
        send_loss.backward(retain_graph=True)
        sender.optimizer.step()

        recver.optimizer.zero_grad()
        recv_loss.backward()
        recver.optimizer.step()


        if print_freq and (e % print_freq == 0):
            print(f'EPISODE {e}')
            print('REWD    {:2.2f}     {:2.2f}'.format(send_logs['reward'], recv_logs['reward']))
            print('LOSS    {:2.2f}     {:2.2f}'.format(send_logs['loss'], recv_logs['loss']))
            # print('DIFF    {:2.2f}     {:2.2f}'.format(env.send_diffs[-1], env.recv_diffs[-1]))
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
        torch.save({'sender': sender.state_dict(),
                    'recver': recver.state_dict(),
                    }, f'{savedir}/models.save')

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
