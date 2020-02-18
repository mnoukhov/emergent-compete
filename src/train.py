import argparse
import json
import os
import random

import gin
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.agents import mode, Reinforce
from src.game import Game, CircleL1


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
def train(Sender, Recver, vocab_size,
          num_epochs, num_batches, batch_size,
          savedir=None, loaddir=None,
          random_seed=None, Loss=None, device='cpu',
          last_epochs_metric=10):

    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
            torch.cuda.manual_seed(random_seed)

    game = Game(num_batches=num_batches,
                batch_size=batch_size,
                device=device)
    test_game = Game(num_batches=1,
                     batch_size=100,
                     device=device,
                     training=False)

    if Loss is None:
        loss_fn = CircleL1(game.num_points)
    else:
        loss_fn = Loss(game.num_points)

    sender = Sender(input_size=1,
                    output_size=vocab_size,
                    mode=mode.SENDER).to(device)
    recver = Recver(input_size=vocab_size,
                    output_size=1,
                    mode=mode.RECVER).to(device)

    send_opt = Adam(sender.parameters(), lr=sender.lr)
    recv_opt = Adam(recver.parameters(), lr=recver.lr)

    # Saving
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        with open(f'{savedir}/config.gin', 'w') as f:
            f.write(gin.operative_config_str())
            f.write(f"train.device = '{device}'")

        logfile = open(f'{savedir}/logs.json', 'w')
        logfile.write('[ \n')
    else:
        logfile = None

    # Loading
    if loaddir is not None:
        loaddir = os.path.join('results', loaddir)
        if os.path.exists(f'{loaddir}/models.save'):
            model_save = torch.load(f'{loaddir}/models.save')
            sender.load_state_dict(model_save['sender'])
            recver.load_state_dict(model_save['recver'])

    test_l1_errors = []
    test_l2_errors = []
    l1_loss_fn = CircleL1(game.num_points)
    l2_loss_fn = CircleL2(game.num_points)

    for epoch in range(num_epochs):
        epoch_send_logs = {}
        epoch_recv_logs = {}

        # Training
        sender.train()
        recver.train()
        for b, batch in enumerate(game):
            send_target, recv_target = batch

            start_rng_state = torch.get_rng_state()

            message, send_logprobs, send_entropy = sender(send_target)
            action, recv_logprobs, recv_entropy = recver(message.detach())
            send_error = loss_fn(action, send_target).squeeze()
            recv_error = loss_fn(action, recv_target).squeeze()

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
        epoch_send_test_l1_error = 0
        epoch_recv_test_l1_error = 0
        epoch_send_test_l2_error = 0
        epoch_recv_test_l2_error = 0
        with torch.no_grad():
            for b, batch in enumerate(test_game):
                send_target, recv_target = batch

                if isinstance(sender, Reinforce):
                    # get recver's action for any given message
                    all_messages = torch.arange(vocab_size)
                    action, recv_logprobs, recv_entropy = recver(all_messages)

                    # get sender's distribution of messages for the inputs
                    dist = sender.forward_dist(send_target)
                else:
                    raise Exception('TODO: continuous messages')


                # duplicate target for each possible message-action
                send_targets = send_target.repeat((1, vocab_size))
                recv_targets = recv_target.repeat((1, vocab_size))

                # get errors for each of those message-actions
                send_test_error = loss_fn(action, send_targets.T)
                recv_test_error = loss_fn(action, recv_targets.T)
                send_test_l1_error = l1_loss_fn(action, send_targets.T)
                recv_test_l1_error = l1_loss_fn(action, recv_targets.T)
                send_test_l2_error = l2_loss_fn(action, send_targets.T)
                recv_test_l2_error = l2_loss_fn(action, recv_targets.T)

                epoch_send_test_error += torch.einsum('bs,sb -> b', dist.probs, send_test_error).mean().item()
                epoch_recv_test_error += torch.einsum('bs,sb -> b', dist.probs, recv_test_error).mean().item()
                epoch_send_test_l1_error += torch.einsum('bs,sb -> b', dist.probs, send_test_l1_error).mean().item()
                epoch_recv_test_l1_error += torch.einsum('bs,sb -> b', dist.probs, recv_test_l1_error).mean().item()
                epoch_send_test_l2_error += torch.einsum('bs,sb -> b', dist.probs, send_test_l2_error).mean().item()
                epoch_recv_test_l2_error += torch.einsum('bs,sb -> b', dist.probs, recv_test_l2_error).mean().item()

        message, _, _ = sender(torch.tensor([[0.]]))
        action, _, _ = recver(message.detach())
        epoch_send_logs['action'] = message[0].item()
        epoch_recv_logs['action'] = action[0].item()
        epoch_send_logs['test_error'] = epoch_send_test_error / test_game.num_batches
        epoch_recv_logs['test_error'] = epoch_recv_test_error / test_game.num_batches
        epoch_send_logs['test_l1_error'] = epoch_send_test_l1_error / test_game.num_batches
        epoch_recv_logs['test_l1_error'] = epoch_recv_test_l1_error / test_game.num_batches
        epoch_send_logs['test_l2_error'] = epoch_send_test_l2_error / test_game.num_batches
        epoch_recv_logs['test_l2_error'] = epoch_recv_test_l2_error / test_game.num_batches

        print(f'EPOCH {epoch}')
        print(f'ERROR {epoch_send_logs["error"]:2.2f} {epoch_recv_logs["error"]:2.2f}')
        print(f'LOSS  {epoch_send_logs["loss"]:2.2f} {epoch_recv_logs["loss"]:2.2f}')
        print(f'TEST  {epoch_send_logs["test_error"]:2.2f} {epoch_recv_logs["test_error"]:2.2f}')
        print(f'L1    {epoch_send_logs["test_l1_error"]:2.2f} {epoch_recv_logs["test_l1_error"]:2.2f}\n')

        test_l1_errors.append(epoch_send_logs['test_l1_error'] + epoch_recv_logs['test_l1_error'])

        if logfile:
            if epoch > 0:
                logfile.write(',\n')
            dump = {'epoch': epoch,
                    'sender': epoch_send_logs,
                    'recver': epoch_recv_logs}
            json.dump(dump, logfile, indent=2)

    if logfile:
        logfile.write('\n]')
        logfile.close()
        torch.save({'sender': sender.state_dict(),
                    'recver': recver.state_dict(),
                    }, f'{savedir}/models.save')

    last_errors_avg = sum(test_l1_errors[-last_epochs_metric:]) / last_epochs_metric
    print(f'Game Over: {last_errors_avg:2.2f}')

    return last_errors_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', '-f', nargs='+')
    parser.add_argument('--gin_param', '-p', nargs='+')
    args = parser.parse_args()

    # change device to torch.device
    gin.config.register_finalize_hook(
        lambda config: config[('', '__main__.train')].update({'device': torch.device(config[('', '__main__.train')]['device'])}))
    gin.parse_config_files_and_bindings(args.gin_file, args.gin_param)

    print(gin.operative_config_str())
    train()

    # gin.clear_config()
    # gin.config._REGISTRY._selector_map.pop('__main__.train')
