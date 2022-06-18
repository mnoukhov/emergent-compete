import tqdm
import argparse
import os

import gin
from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr
from haven import haven_utils as hu

from train import train
from haven_config.exp_configs import EXP_GROUPS
from haven_config.job_config import JOB_CONFIG


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    exp_args = argparse.Namespace(**exp_dict)
    gin.parse_config_files_and_bindings(exp_args.gin_config, exp_args.gin_param)
    print(gin.operative_config_str())

    errors = []
    for random_seed in range(5):
        if exp_args.savedir:
            os.makedirs(exp_args.savedir, exist_ok=True)

            seed_savedir = f"{exp_args.savedir}/{random_seed}"
        else:
            seed_savedir = None

        best_error = train(
            savedir=seed_savedir, random_seed=random_seed, loaddir=exp_args.loaddir
        )
        errors.append(best_error)

    print(f"mean error over seeds: {sum(errors) / len(errors):2.2f}")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument(
        "-j", "--job_scheduler", default=None, help="Choose Job Scheduler."
    )
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )

    args, others = parser.parse_known_args()

    # # Define a list of experiments
    # if args.exp_group == "syn":
    #     exp_list = []
    #     for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    #         exp_list += [{"lr": lr, "dataset": "syn", "model": "linear"}]

    # Choose Job Scheduler
    if args.job_scheduler == "toolkit":
        job_config = JOB_CONFIG
    else:
        job_config = None

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="haven_config/results.ipynb",
        python_binary_path=args.python_binary,
        args=args,
    )
