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
	--working_dir $SLURM_TMP_DIR/$exp_name \
	--max_trials 10
	src/orion_runs.py --config configs/cat-deter-search.gin --savedir {trial.working_dir}

cp -r $SLURM_TMP_DIR/$exp_name $SCRATCH/emergent-selfish/$exp_name

