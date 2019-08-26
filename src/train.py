import argparse
import json
import os
import random

import gin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.agents import mode
from src.game import Game, CircleL1, CircleL2
from src.lola import LOLA


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def _div_dict(d, n):
    result = dict(d)
    for k in result:
        result[k] /= n
    return result


@gin.configurable
def train(Sender, Recver, vocab_size, device,
          num_epochs, num_batches, batch_size,
          grounded=False, savedir=None, loaddir=None,
          random_seed=None):

    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(random_seed)

    game = Game(num_batches=num_batches,
                batch_size=batch_size,
                device=device)
    test_game = Game(num_batches=1,
                     batch_size=100,
                     device=device,
                     training=False)
    loss_fn = CircleL1(game.num_points)

    sender = Sender(input_size=1,
                    output_size=vocab_size,
                    mode=mode.SENDER).to(device)
    recver = Recver(input_size=vocab_size,
                    output_size=1,
                    mode=mode.RECVER).to(device)

    if issubclass(Sender, LOLA):
        sender.other = recver
        sender.loss_fn = loss_fn

    send_opt = Adam(sender.parameters(), lr=sender.lr)
    recv_opt = Adam(recver.parameters(), lr=recver.lr)

    # Saving
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        with open(f'{savedir}/config.gin', 'w') as f:
            f.write(gin.operative_config_str())

        logfile = open(f'{savedir}/logs.json', 'w')
        logfile.write('[ \n')
    else:
        print(gin.operative_config_str())
        logfile = None

    # Loading
    if loaddir is not None:
        loaddir = os.path.join('results', loaddir)
        if os.path.exists(f'{loaddir}/models.save'):
            model_save = torch.load(f'{loaddir}/models.save')
            sender.load_state_dict(model_save['sender'])
            recver.load_state_dict(model_save['recver'])

    best_epoch_error = None

    for e, epoch in enumerate(range(num_epochs)):
        epoch_send_logs = {}
        epoch_recv_logs = {}

        # Training
        sender.train()
        recver.train()
        for b, batch in enumerate(game):
            send_target, recv_target = batch

            message, send_logprobs, send_entropy = sender(send_target)
            action, recv_logprobs, recv_entropy = recver(message)

            if grounded:
                action = message + action

            send_error = loss_fn(action, send_target).mean(dim=1)
            recv_error = loss_fn(action, recv_target).mean(dim=1)

            send_loss, send_logs = sender.loss(send_error, send_logprobs, send_entropy)
            recv_loss, recv_logs = recver.loss(recv_error, recv_logprobs, recv_entropy)

            # sender must be updated before recver if using retain_graph
            send_opt.zero_grad()
            send_loss.backward(retain_graph=sender.retain_graph)
            send_opt.step()

            recv_opt.zero_grad()
            recv_loss.backward()
            recv_opt.step()

            epoch_send_logs = _add_dicts(epoch_send_logs, send_logs)
            epoch_recv_logs = _add_dicts(epoch_recv_logs, recv_logs)

        epoch_send_logs = _div_dict(epoch_send_logs, game.num_batches)
        epoch_recv_logs = _div_dict(epoch_recv_logs, game.num_batches)

        # Testing
        sender.eval()
        recver.eval()
        epoch_send_test_error = 0
        epoch_recv_test_error = 0
        for b, batch in enumerate(test_game):
            send_target, recv_target = batch

            message, send_logprobs, send_entropy = sender(send_target)
            action, recv_logprobs, recv_entropy = recver(message)

            if grounded:
                action = message + action

            send_test_error = loss_fn(action, send_target).mean(dim=1)
            recv_test_error = loss_fn(action, recv_target).mean(dim=1)

            epoch_send_test_error += send_test_error.mean().item()
            epoch_recv_test_error += recv_test_error.mean().item()

        epoch_send_logs['test_error'] = epoch_send_test_error / test_game.num_batches
        epoch_recv_logs['test_error'] = epoch_recv_test_error / test_game.num_batches

        print(f'EPOCH {e}')
        print(f'ERROR {epoch_send_logs["error"]:2.2f} {epoch_recv_logs["error"]:2.2f}')
        print(f'LOSS  {epoch_send_logs["loss"]:2.2f} {epoch_recv_logs["loss"]:2.2f}')
        print(f'TEST  {epoch_send_logs["test_error"]:2.2f} {epoch_recv_logs["test_error"]:2.2f}\n')

        epoch_total_error = epoch_send_logs["test_error"] + epoch_recv_logs["test_error"]
        if best_epoch_error is None or epoch_total_error < best_epoch_error:
            best_epoch_error = epoch_total_error

        if logfile:
            dump = {'epoch': e,
                    'sender': epoch_send_logs,
                    'recver': epoch_recv_logs}
            json.dump(dump, logfile, indent=2)
            logfile.write(',\n')

    print(f'Game Over: {best_epoch_error:2.2f}')

    if logfile:
        dump = {'epoch': e,
                'sender': send_logs,
                'recver': recv_logs}
        json.dump(dump, logfile, indent=2)
        logfile.write('\n]')
        logfile.close()
        torch.save({'sender': sender.state_dict(),
                    'recver': recver.state_dict(),
                    }, f'{savedir}/models.save')

    return best_epoch_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', '-f', nargs='+')
    parser.add_argument('--gin_param', '-p', nargs='+')
    args = parser.parse_args()

    # change device to torch.device
    gin.config.register_finalize_hook(
        lambda config: config[('', '__main__.train')].update({'device': torch.device(config[('', '__main__.train')]['device'])}))
    gin.parse_config_files_and_bindings(args.gin_file, args.gin_param)

    train()

    # gin.clear_config()
    # gin.config._REGISTRY._selector_map.pop('__main__.train')
