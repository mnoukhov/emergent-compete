#!/bin/bash
#SBATCH --job-name=deter-deter
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --mem=4Gb

export PYTHONPATH='/network/home/noukhovm/emergent-selfish'
source activate selfish

bias=9
vocab=2
savedir="deter-deter-bias${bias}-vocab${vocab}/${SLURM_ARRAY_TASK_ID}"

python src/train.py -p train.savedir="'${savedir}'" ISR.min_bias=${bias} train.vocab_size=${vocab} -f deter-deter.gin
