#!/usr/bin/env python
import argparse
import os

import gin
from orion.client import report_results
import torch

from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--gin_file', nargs='+')
    parser.add_argument('--gin_param', '-p', nargs='+')
    parser.add_argument('--savedir')
    parser.add_argument('--objective', choices=['mean', 'min'], default='mean')
    args = parser.parse_args()

    # change device to torch.device
    gin.config.register_finalize_hook(
        lambda config: config[('', 'src.train.train')].update({'device': torch.device(config[('', 'src.train.train')]['device'])}))

    gin.parse_config_files_and_bindings(args.config, args.gin_param)

    errors = []
    for random_seed in range(5):
        if args.savedir:
            os.makedirs(args.savedir, exist_ok=True)

            seed_savedir = f'{args.savedir}/{random_seed}'
        else:
            seed_savedir = None

        best_error = train(savedir=seed_savedir,
                           random_seed=random_seed)
        errors.append(best_error)

    if args.objective == 'mean':
        objective = sum(errors) / len(errors)
    elif args.objective == 'min':
        objective = min(errors)
    else:
        raise NotImplementedError(f'no objective: {args.objective}')

    print(f'{args.objective} error over seeds: {objective:2.2f}')

    report_results([dict(
        name=f'{objective}_error_over_seeds',
        type='objective',
        value=objective)])
