#!/usr/bin/env bash

for BIAS in 0 3 6 9 12 15
do
    config="cat-deter-search.gin"
    experiment_name="cat-deter-bias$BIAS"
    params="Game.bias=$BIAS"

    for process in $(seq 1 5)
    do
        borgy submit \
            --name $experiment_name \
            --mem 4 \
            -e HOME=$HOME \
            -i volatile-images.borgy.elementai.net/mnoukhov/emergent:latest \
            -v $HOME/emergent-compete:/workspace:ro \
            -v /mnt/scratch/mnoukhov/emergent:/scratch:rw \
            -- bash -c "./scripts/hyperparam_search.sh $experiment_name $config $params"
    done
done
