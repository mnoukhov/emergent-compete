experiment_name=$1
config=$2
params=${@:3}

export PYTHONUNBUFFERED=1

orion --debug hunt -n $experiment_name	\
    --working-dir /tmp/$experiment_name \
    --max-trials 20 \
    orion_runs.py --config configs/$config \
    --savedir {trial.working_dir} \
    --gin_param $params

mkdir -p /scratch/$experiment_name
cp -r /tmp/$experiment_name/* /scratch/$experiment_name/
