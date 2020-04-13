bias=$1
experiment_name=$2
config=$3
params=${@:4}

export PYTHONUNBUFFERED=1

cd /home/mnoukhov/emergent-selfish
pip install --user -e .

orion --debug hunt -n $experiment_name	\
    --working-dir /tmp/$experiment_name \
    --max-trials 100 \
    src/orion_runs.py --config configs/$config \
    --savedir {trial.working_dir} \
    --gin_param $params

mkdir -p /scratch/$experiment_name
cp -r /tmp/$experiment_name/* /scratch/$experiment_name/
