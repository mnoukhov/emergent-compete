import argparse
import json
import os

import gin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.agents import mode
from src.data import Circle, CircleLoss
from src.utils import load, plot
import src.lola


@gin.configurable
def train(Sender, Recver, vocab_size,
          render_freq, log_freq, print_freq,
          savedir, loaddir, device):

    dataloader = Circle(device=device)
    sender = Sender(input_size=1,
                    output_size=vocab_size,
                    mode=mode.SENDER,
                    device=device)
    recver = Recver(input_size=vocab_size,
                    output_size=1,
                    output_range=dataloader.num_points,
                    mode=mode.RECVER,
                    device=device)
    circle_loss = CircleLoss(dataloader.num_points)

    # sender.other = recver

    # Saving
    if savedir is not None:
        savedir = os.path.join('results', savedir)
        os.makedirs(savedir, exist_ok=True)

        with open(f'{savedir}/config.gin', 'w') as f:
            f.write(gin.operative_config_str())

        logfile = open(f'{savedir}/logs.json', 'w')
        logfile.write('[ \n')
    else:
        logfile = None

    # Loading
    if loaddir is not None:
        loaddir = os.path.join('results', loaddir)
        if os.path.exists(f'{loaddir}/models.save'):
            load(sender, recver, loaddir)

    # Training
    for e, batch in enumerate(dataloader):
        send_target, recv_target = batch

        message = sender(send_target)
        action = recver(message)

        send_reward = -circle_loss(action, send_target)
        recv_reward = -circle_loss(action, recv_target)

        send_loss, send_logs = sender.loss(send_reward)
        recv_loss, recv_logs = recver.loss(recv_reward)

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
