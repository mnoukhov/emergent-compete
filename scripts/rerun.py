#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import gin
import torch

from src.train import train

def rerun(results_dir):
    results_path = Path(results_dir)
    output_path = Path(f'{results_dir}-extended')
    output_path.mkdir(exist_ok=True)

    biases = []
    errors = []
    ids = []

    for bias_path in results_path.iterdir():
        if not bias_path.is_dir():
            continue
        config_path = next(bias_path.glob('**/*.gin'))

        gin.config.register_finalize_hook(
            lambda config: config[('', 'src.train.train')].update({'device': torch.device(config[('', 'src.train.train')].get('device','cpu'))}))
        gin.parse_config_file(str(config_path))

        bias_index = bias_path.name.find('bias') + 4
        under_index = bias_path.name.find('_')
        bias = int(bias_path.name[bias_index:under_index])
        id_ = bias_path.name[under_index+1:]

        biases.append(bias)
        ids.append(id_)

        bias_error = 0
        bias_output_path = output_path / bias_path.name
        bias_output_path.mkdir(exist_ok=True)
        for random_seed in range(5):
            seed_output_path =  bias_output_path / f'{random_seed}'
            seed_output_path.mkdir(exist_ok=True)
            seed_error = train(savedir=seed_output_path,
                               random_seed=random_seed,
                               num_epochs=30,
                               last_epochs_metric=30)
            bias_error += seed_error

        errors.append(bias_error / 5)

    with open(output_path / 'results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['bias', 'error', 'id'])
        for bias, error, id_ in sorted(zip(biases, errors, ids)):
            writer.writerow([bias, error, id_])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()

    rerun(args.results_dir)

