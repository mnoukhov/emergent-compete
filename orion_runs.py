#!/usr/bin/env python
import argparse
import os

import gin
from orion.client import report_results
import torch

from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--gin_file', nargs='+')
    parser.add_argument('--gin_param', '-p', nargs='+')
    parser.add_argument('--savedir')
    parser.add_argument('--loaddir')
    parser.add_argument('--aggregate_seeds', choices=['mean', 'min'], default='mean')
    args = parser.parse_args()

    # change device to torch.device
    # gin.config.register_finalize_hook(
        # lambda config: config[('', 'src.train.train')].update({'device': torch.device(config[('', 'src.train.train')].get('device','cpu'))}))

    gin.parse_config_files_and_bindings(args.config, args.gin_param)
    print(gin.operative_config_str())

    errors = []
    for random_seed in range(5):
        if args.savedir:
            os.makedirs(args.savedir, exist_ok=True)

            seed_savedir = f'{args.savedir}/{random_seed}'
        else:
            seed_savedir = None

        best_error = train(savedir=seed_savedir,
                           random_seed=random_seed,
                           loaddir=args.loaddir)
        errors.append(best_error)

    if args.aggregate_seeds == 'mean':
        objective = sum(errors) / len(errors)
    elif args.aggregate_seeds == 'min':
        objective = min(errors)

    print(f'{args.aggregate_seeds} error over seeds: {objective:2.2f}')

    report_results([dict(
        name=f'{args.aggregate_seeds}_error_over_seeds',
        type='objective',
        value=objective)])
