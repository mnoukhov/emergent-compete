#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --array=1-5
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/noukhovm/scratch/emergent-selfish/hyperparam-search.%A.%a.out
#SBATCH --error=/home/noukhovm/scratch/emergent-selfish/hyperparam-search.%A.%a.err
#SBATCH --job-name=emergent-hyperparam
#SBATCH --mem=4GB
#SBATCH --time=2:59:00

source scripts/beluga.sh

exp_name="test"

orion hunt -n $exp_name	\
	--working_dir /home/noukovm/scratch/emergent-selfish/$exp_name \
	--max_trials 10
	./scripts/hyperparam_search.py --config configs/test.gin --savedir {trial.working_dir}
