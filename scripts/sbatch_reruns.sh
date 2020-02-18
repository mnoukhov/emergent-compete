#!/usr/bin/env bash

for BIAS in 0 3 6 9 12 15
do
    sbatch scripts/rerun_one_slurm.sh $BIAS
done
