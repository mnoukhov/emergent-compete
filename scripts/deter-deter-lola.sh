#!/bin/bash
#SBATCH --job-name=lola
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=4Gb

export PYTHONPATH='/network/home/noukhovm/emergent-selfish'
source activate selfish

bias=1
lola=0
vocab=1
savedir="deter-deter-lola/lola${lola}-vocab${vocab}-bias${bias}/${SLURM_ARRAY_TASK_ID}"

python src/train.py -p train.savedir="'${savedir}'" train.vocab_size=${vocab} ISR.min_bias=${bias} DeterExactLOLA.order=${lola} -f deter-deter-lola.gin

