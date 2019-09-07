#!/bin/bash

EXP_NAME=$1

ssh graham << EOF
    source ~/env.sh
    cd emergent-selfish
    python scripts/new_metric_check.py generate --results_dir /scratch/noukhovm/emergent-selfish --experiment_name $EXP_NAME --output_dir results/
EOF
rsync -chavP graham:/home/noukhovm/emergent-selfish/results/$EXP_NAME /home/mnoukhov/emergent-selfish/results/$EXP_NAME
