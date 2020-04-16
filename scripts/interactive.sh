#!/bin/bash

borgy submit -I \
    --name "interactive" \
    --mem 4 \
    -e HOME=$HOME \
    -i volatile-images.borgy.elementai.net/mnoukhov/emergent:latest \
    -v $HOME:$HOME:rw \
    -v /mnt/scratch/mnoukhov/emergent:/scratch:rw \
    -- bash -c "while true; do sleep 60; done"
