import argparse
from pathlib import Path

import pandas as pd


def alternative_score(seeds_dir):
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

def check_all_runs(all_results_dir):
    all_results_path = Path(all_results_dir)

    min_score = None
    min_index = None
    for result_dir in all_results_path.iterdir():
        if result_dir.is_dir():
            print(result_dir.name)
            score = alternative_score(result_dir)
            if min_score is None or score < min_score:
                min_score = score
                min_index = result_dir.name

    return min_score, min_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')

    args = parser.parse_args()

    print(check_all_runs(args.dir))
