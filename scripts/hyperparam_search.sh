#!/bin/bash

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

LOGDIR=$SCRATCH

max_trials=100
bias=0
experiment_name="cat-deter-bias$bias"
config="cat-deter-search.gin"
params="Game.bias=$bias"

orion --debug hunt -n $experiment_name	\
    --working-dir $LOGDIR/$experiment_name \
    --max-trials $max_trials \
    orion_runs.py --config configs/$config \
    --savedir {trial.working_dir} \
    --gin_param $params
