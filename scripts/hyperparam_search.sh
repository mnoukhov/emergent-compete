max_trials=$1
experiment_name=$2
config=$3
params=${@:4}

export PYTHONUNBUFFERED=1

orion --debug hunt -n $experiment_name	\
    --working-dir /tmp/$experiment_name \
    --max-trials $max_trials \
    orion_runs.py --config configs/$config \
    --savedir {trial.working_dir} \
    --gin_param $params

mkdir -p /scratch/$experiment_name
cp -r /tmp/$experiment_name/* /scratch/$experiment_name/
