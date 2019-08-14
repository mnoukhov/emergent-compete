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
    args = parser.parse_args()

    # change device to torch.device
    gin.config.register_finalize_hook(
        lambda config: config[('', 'src.train.train')].update({'device': torch.device(config[('', 'src.train.train')]['device'])}))

    gin.parse_config_files_and_bindings(args.config, args.gin_param)


    mean_best_reward = 0

    for random_seed in range(5):
        if args.savedir:
            os.makedirs(args.savedir, exist_ok=True)

            seed_savedir = f'{args.savedir}/{random_seed}'
        else:
            seed_savedir = None

        best_reward = train(savedir=seed_savedir,
                            random_seed=random_seed)
        mean_best_reward += best_reward

    mean_best_reward /= 5

    print('mean best reward over seeds {mean_best_reward}')
    report_results([dict(
        name='best_neg_reward',
        type='objective',
        value=-mean_best_reward)])
