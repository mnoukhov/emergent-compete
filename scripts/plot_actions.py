from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def get_logs(seeds_dir):
    results_path = Path(seeds_dir)

    if verbose:
        print(results_path.name)

    run_logs = []
    for path in results_path.glob('*/logs.json'):
        if verbose:
            print(path)
        with open(path, 'r') as logfile:
            try:
                run_logs.append(pd.read_json(logfile))
            except ValueError as e:
                raise ValueError(f'cant read json {path}: {e}')

    if not run_logs:
        return None

    logs = pd.concat(run_logs, ignore_index=True)
    epoch = logs['epoch']
    sender = pd.DataFrame(logs['sender'].tolist()).join(logs['epoch'])
    recver = pd.DataFrame(logs['recver'].tolist()).join(logs['epoch'])

    return epoch, sender, recver


def get_actions(seeds_dir):
    epoch, sender, recver = get_logs(seeds_dir)

if __name__ == '__main__':
    with sns.plotting_context('paper'):
        path = '/home/mnoukhov/emergent-selfish/results/continuous/gauss-deter-bias9'
        sender_l1, recver_l1 = all_metrics(path)
        sns.set(font_scale=1.2)
        sns.scatterplot(sender_l1,recver_l1, label='continuous')
