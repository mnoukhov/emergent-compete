import argparse
import csv
from pathlib import Path
import shutil

import pandas as pd
import gin

# from src.agents import Gaussian


def metric(seeds_dir, error_name='l1', verbose=False):
    # average of last 10 epochs
    results_path = Path(seeds_dir)

    # if verbose:
        # print(results_path.name)

    run_logs = []
    for path in results_path.glob('*/logs.json'):
        # if verbose:
            # print(path)
        with open(path, 'r') as logfile:
            try:
                run_logs.append(pd.read_json(logfile))
            except ValueError as e:
                raise ValueError(f'cant read json {path}: {e}')

    if not run_logs:
        return None

    gin.parse_config_file(results_path / '0/config.gin', skip_unknown=True)
    # if gin.config.query_parameter('Gaussian.dim') < 32:
        # return None

    logs = pd.concat(run_logs, ignore_index=True)
    epoch = logs['epoch']
    sender = pd.DataFrame(logs['sender'].tolist()).join(logs['epoch'])
    recver = pd.DataFrame(logs['recver'].tolist()).join(logs['epoch'])
    if error_name == 'l1' and 'test_l1_error' in sender:
        error_metric = 'test_l1_error'
    elif error_name == 'l2' and 'test_l2_error' in sender:
        error_metric = 'test_l2_error'
    elif error_name == 'train':
        error_metric = 'test_error'
    else:
        raise Exception(f'error name {error_name} either not found or not valid')

    last_10 = logs['epoch'] >= 20
    if error_name == 'l1' and (sender[last_10][error_metric].mean() > 9 and recver[last_10][error_metric].mean() > 9):
        return None
    # elif error_name == 'l2' and (sender[last_10][error_metric].mean() > 120 and recver[last_10][error_metric].mean() > 120):
        # __import__('pdb').set_trace()
        # return None
    else:
        total_error = sender[error_metric] + recver[error_metric]
        if total_error.isnull().values.any():
            return None
        else:
            score = total_error.to_frame().groupby(epoch).mean()[-10:].mean()[error_metric]
            return score


def metric_over_runs(all_results_dir, error_name, verbose=True):
    all_results_path = Path(all_results_dir)

    empty =  []
    errors = []
    min_score = None
    min_index = None
    min_l1 = None
    num_runs = 0
    for result_dir in all_results_path.iterdir():
        if result_dir.is_dir():
            try:
                score = metric(result_dir, error_name, verbose)
                if error_name != 'l1':
                    l1 = metric(result_dir, 'l1', verbose)
                else:
                    l1 = score
            except ValueError:
                errors.append(result_dir.name)
            else:
                if score is None:
                    empty.append(result_dir.name)
                elif min_score is None or score < min_score:
                    min_score = score
                    min_index = result_dir.name
                    min_l1 = l1
                    num_runs += 1
                else:
                    num_runs += 1

    if verbose:
        print(f'Empty dirs {empty}')
        print(f'Error dirs {errors}')
        print(f'Number of runs {num_runs}')

    return min_score, min_index, min_l1


def generate_results_csv(experiment_name, cluster_dir, output_dir='.', error_name='l1', verbose=False):
    output_path = Path(output_dir)
    cluster_results_path = Path(cluster_dir)
    if not cluster_results_path.exists():
        raise Exception(f'results path {cluster_results_path} does not exist')

    biases = []
    best_errors = []
    best_l1s = []
    ids = []
    run_paths = []

    for exp_results_path in cluster_results_path.glob(f'{experiment_name}-bias*'):
        exp_full_name = exp_results_path.name
        bias_index = exp_full_name.find('bias') + 4
        name_index = len(exp_full_name) + 1
        print(f'running on {exp_full_name}')
        error, run_name, l1 = metric_over_runs(exp_results_path, error_name=error_name, verbose=verbose)
        if error is None:
            print(f'no results in {exp_full_name}')
        else:
            biases.append(int(exp_full_name[bias_index:]))
            best_errors.append(error)
            best_l1s.append(l1)
            ids.append(run_name[name_index:])
            run_paths.append(exp_results_path / run_name)

    if not run_paths:
        raise Exception(f'could not find any experiment {experiment_name} in {cluster_results_path}')

    with open(output_path / 'results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['bias', 'error', 'id', 'l1'])
        for bias, error, id_, l1 in sorted(zip(biases, best_errors, ids, best_l1s)):
            writer.writerow([bias, error, id_, l1])

    return run_paths


def generate_results_folder(experiment_name, cluster_dir, output_path, error_name, verbose=False):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    print('generating results csv')
    best_run_paths = generate_results_csv(experiment_name, cluster_dir, output_path, error_name, verbose)

    print('copying files')
    for run_path in best_run_paths:
        dest = output_path / run_path.name
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(run_path), str(dest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # generate results folder
    gen_parser = subparsers.add_parser('generate')
    gen_parser.set_defaults(command='generate')
    gen_parser.add_argument('--experiment-name', '--exp-name', required=True)
    gen_parser.add_argument('--results-dir', default='/scratch/')
    gen_parser.add_argument('--output-dir', default=None)
    gen_parser.add_argument('--error', default='l1')
    gen_parser.add_argument('--verbose', action='store_true')

    # get error metric over hyperparams
    check_parser = subparsers.add_parser('check')
    check_parser.set_defaults(command='check')
    check_parser.add_argument('dir')
    check_parser.add_argument('--error', default='l1')

    args = parser.parse_args()

    if args.command == 'generate':
        if args.output_dir is None:
            output_path = f'/home/mnoukhov/emergent-compete/results/{args.experiment_name}'
        else:
            output_path = args.output_dir
        generate_results_folder(args.experiment_name, args.results_dir, output_path, args.error, args.verbose)
    elif args.command == 'check':
        print(metric_over_runs(args.dir, args.error))
