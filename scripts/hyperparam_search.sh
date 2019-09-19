#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --array=1-3
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/noukhovm/scratch/slurm-logs/hyperparam-search.%A.%a.out
#SBATCH --error=/home/noukhovm/scratch/slurm-logs/hyperparam-search.%A.%a.err
#SBATCH --job-name=emergent-hyperparam
#SBATCH --mem=4GB
#SBATCH --time=8:59:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mnoukhov@gmail.com

module load python/3.7
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install -e .

bias=9
experiment_name="gauss-deter-grounded-bias$bias-redo"
config="gauss-deter-search.gin"
params="Game.bias=$bias train.grounded=True"

export PYTHONUNBUFFERED=1

orion hunt -n $experiment_name	\
    --working-dir $SLURM_TMPDIR/$experiment_name \
    --max-trials 75 \
    src/orion_runs.py --config configs/$config \
    --savedir {trial.working_dir} \
    --gin_param $params

mkdir -p $SCRATCH/emergent-selfish/$experiment_name
cp -r $SLURM_TMPDIR/$experiment_name/* $SCRATCH/emergent-selfish/$experiment_name/
rm -rf $SLURM_TMPDIR/env
