#!/usr/bin/env python
import argparse
import os

import gin
from orion.client import report_results
import torch

from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, nargs='+')
    parser.add_argument('--gin_param', '-p', nargs='+')
    parser.add_argument('--savedir')
    parser.add_argument('--objective', choices=['mean', 'max'], default='max')
    args = parser.parse_args()

    # change device to torch.device
    gin.config.register_finalize_hook(
        lambda config: config[('', 'src.train.train')].update({'device': torch.device(config[('', 'src.train.train')]['device'])}))

    gin.parse_config_files_and_bindings(args.config, args.gin_param)

    rewards = []
    for random_seed in range(5):
        if args.savedir:
            os.makedirs(args.savedir, exist_ok=True)

            seed_savedir = f'{args.savedir}/{random_seed}'
        else:
            seed_savedir = None

        best_reward = train(savedir=seed_savedir,
                            random_seed=random_seed)
        rewards.append(best_reward)

    if args.objective == 'mean':
        objective = -sum(rewards) / len(rewards)
    elif args.objective == 'max':
        objective = -max(rewards)
    else:
        raise NotImplementedError(f'no objective: {args.objective}')

    print(f'{args.objective} best objective over seeds {objective:2.2f}')

    report_results([dict(
        name='best_neg_reward',
        type='objective',
        value=objective)])
