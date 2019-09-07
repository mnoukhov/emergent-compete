import argparse
import csv
from pathlib import Path
import shutil

import pandas as pd


def metric(seeds_dir):
    # average of last 10 epochs
    results_path = Path(seeds_dir)

    run_logs = []
    for path in results_path.glob('*/logs.json'):
#         print(path)
        with open(path, 'r') as logfile:
            run_logs.append(pd.read_json(logfile))

    logs = pd.concat(run_logs, ignore_index=True)
    epoch = logs['epoch']
    sender = pd.DataFrame(logs['sender'].tolist()).join(logs['epoch'])
    recver = pd.DataFrame(logs['recver'].tolist()).join(logs['epoch'])
    total_error = sender['test_error'] + recver['test_error']
    return total_error.to_frame().groupby(epoch).mean()[-10:].mean()['test_error']


def metric_over_runs(all_results_dir, verbose=True):
    all_results_path = Path(all_results_dir)

    min_score = None
    min_index = None
    for result_dir in all_results_path.iterdir():
        if result_dir.is_dir():
            if verbose:
                print(result_dir.name)
            score = metric(result_dir)
            if min_score is None or score < min_score:
                min_score = score
                min_index = result_dir.name

    return min_score, min_index


def generate_results_csv(experiment_name, cluster_dir, output_dir='.'):
    output_path = Path(output_dir)
    cluster_results_path = Path(cluster_dir)

    biases = []
    best_errors = []
    ids = []
    run_paths = []

    for exp_results_path in cluster_results_path.glob(f'{experiment_name}*'):
        exp_full_name = exp_results_path.name
        bias_index = exp_results.find('bias') + 4
        name_index = len(exp_full_name) + 1
        print(f'running on {exp_full_name}')
        error, run_name = metric_over_runs(exp_results_path, verbose=False)

        biases.append(int(exp_full_name[bias_index:]))
        best_errors.append(error)
        ids.append(run_name[name_index:])
        run_paths.append(exp_results_path / run_name)

    with open(output_path / 'results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['bias', 'error', 'id'])
        for bias, error, id_ in zip(biases, best_errors, ids):
            writer.writerow([bias, error, id_])

    return run_paths


def generate_results_folder(experiment_name, cluster_dir, output_dir):
    output_path = Path(output_dir)
    results_folder_path = output_path.mkdir(experiment_name, exists_ok=True)

    print('generating results csv')
    best_run_paths = generate_results_csv(experiment_name, cluster_dir, results_folder_path)
    print('copying files')
    for run_path in best_run_paths:
        shutil.copy2(str(run_path), results_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # generate results folder
    gen_parser = subparser.add_parser('generate')
    gen_parser.set_defaults(command='generate')
    gen_parser.add_argument('experiment_name')
    gen_parser.add_argument('results_dir)
    gen_parser.add_argument('output_dir')

    check_parser = subparsers.add_parser('check')
    gen_parser.set_defaults(command='check')
    parser.add_argument('results_dir')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_results_folder(args.experiment_name, args.results_dir, args.outputt_dir)
    elif args.command == 'check':
        print(metric_over_runs(args.results_dir))
