#!/usr/bin/env bash

for BIAS in 0 3 6 9 12 15
do
    sbatch scripts/hyperparam_search.sh $BIAS
done
