#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --output=/home/noukhovm/scratch/slurm-logs/rerun.%A.%a.out
#SBATCH --error=/home/noukhovm/scratch/slurm-logs/rerun.%A.%a.err
#SBATCH --job-name=rerun
#SBATCH --mem=4GB
#SBATCH --time=2:59:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mnoukhov@gmail.com

module load python/3.7
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install -e .

export PYTHONUNBUFFERED=1

EXP_NAME="cat-deter"

python scripts/rerun.py $HOME/emergent-selfish/results/$EXP_NAME
