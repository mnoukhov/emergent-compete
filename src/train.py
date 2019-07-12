import argparse
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
def train(Sender, Recver, episodes, vocab_size, render_freq, log_freq,
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

    if loaddir:
        model_save = torch.load(f'results/{loaddir}/models.save')
        sender.load_state_dict(model_save['sender'])
        recver.load_state_dict(model_save['recver'])

    for e in range(episodes):
        target = env.reset().to(device)
        send_rewards = []
        recv_rewards = []

        message = torch.zeros(env.batch_size, device=device)
        action = torch.zeros(env.batch_size, device=device)
        recv_reward = torch.zeros(env.batch_size, device=device)
        send_reward = torch.zeros(env.batch_size, device=device)
        prev_recv_reward = torch.zeros(env.batch_size, device=device)
        prev_send_reward = torch.zeros(env.batch_size, device=device)
        prev_target = torch.zeros(env.batch_size, device=device)
        prev_message = torch.zeros(env.batch_size, device=device)
        prev_action = torch.zeros(env.batch_size, device=device)
        prev2_target = torch.zeros(env.batch_size, device=device)
        send_state = None
        recv_state = None

        for r in range(env.num_rounds):
            prev2_message = prev_message.detach()
            prev2_action = prev_action.detach()
            prev_message = message.detach()
            prev_action = action.detach()
            prev_send_state = send_state
            prev_recv_state = recv_state

            # send_state = torch.stack([target, prev_target, prev_message, send_reward,
                                      # prev2_target, prev2_message, prev_send_reward],
                                     # dim=1)
            send_state = target.unsqueeze(1)
            message = sender.action(send_state)

            # recv_state = torch.stack([message, prev_message, prev_action, recv_reward,
                                      # prev2_message, prev2_action, prev_recv_reward],
                                     # dim=1)
            recv_state = message
            action = recver.action(recv_state)

            if r > 0 and hasattr(sender, 'memory'):
                sender.memory.push(prev_send_state.cpu(), prev_message.cpu(), prev_action.cpu(),
                                    send_reward.cpu(), recv_reward.cpu(), send_state.cpu())
            if r > 0 and hasattr(recver, 'memory'):
                recver.memory.push(prev_recv_state.cpu(), prev_message.cpu(), prev_action.cpu(),
                                    send_reward.cpu(), recv_reward.cpu(), recv_state.cpu())

            prev2_target = prev_target
            prev_target = target
            prev_recv_reward = recv_reward.detach()
            prev_send_reward = send_reward.detach()

            target, (send_reward, recv_reward), done, = env.step(action)
            target = target.to(device) if target is not None else None
            send_reward = send_reward.to(device)
            recv_reward = recv_reward.to(device)

            send_rewards.append(send_reward)
            recv_rewards.append(recv_reward)

            if render_freq and e % render_freq == 0:
                env.render(message=message[0].item())

        log_now = log_freq and (e % log_freq == 0)
        recver.update(e, recv_rewards, log_now, retain_graph=True)
        sender.update(e, send_rewards, log_now)

        if log_now:
            print(f'EPISODE {e}')
            print('REWD    {:2.2f}     {:2.2f}'.format(sender.last('ep_reward'), recver.last('ep_reward')))
            print('LOSS    {:2.2f}     {:2.2f}'.format(sender.last('loss'), recver.last('loss')))
            print('DIFF    {:2.2f}     {:2.2f}'.format(env.send_diffs[-1], env.recv_diffs[-1]))
            print('')

    # sender.writer.close()
    # recver.writer.close()
    print('Game Over')
    x = list(range(episodes))

    if savedir is not None:
        savedir = os.path.join('results', savedir)
        os.makedirs(savedir, exist_ok=True)

        plot(x, sender, recver, env, savedir)
        save(sender, recver, env, savedir)

    print(gin.operative_config_str())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', nargs='+', default=['default.gin'])
    parser.add_argument('gin_param', nargs='+')
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
