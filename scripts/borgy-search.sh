#!/usr/bin/env bash

for BIAS in 0 3 6 9 12 15
do
    experiment_name="gauss-deter-bias$BIAS"
    config="gauss-deter-search.gin"
    params="Game.bias=$BIAS"

    borgy submit \
        --name $experiment_name \
        --mem 4 \
        -e HOME=$HOME \
        -i volatile-images.borgy.elementai.net/mnoukhov/emergent:latest \
        -v $HOME:$HOME:rw \
        -v /mnt/scratch/mnoukhov/emergent:/scratch:rw \
        -- bash -c "/home/mnoukhov/emergent-selfish/scripts/hyperparam_search.sh $BIAS $experiment_name $config $params"
done
