#!/usr/bin/env bash

if [ -z "$WANDB_API_KEY" ]
then
    echo "var WANDB_API_KEY not provided, make sure you're not running with --wandb"
fi

borgy submit -I \
    --mem 4 \
    -e HOME=$HOME \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -i volatile-images.borgy.elementai.net/mnoukhov/emergent:latest \
    -v $HOME:$HOME:rw \
    -v /mnt/scratch/mnoukhov/emergent:/scratch:rw \
    -- bash -c "while true; do sleep 60; done"
