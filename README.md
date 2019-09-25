# Emergent Selfish Communication
emerging communication with selfish agents

## Setup
install requirements with either
- `pip install -r requirements.txt` or
- `conda env create -f environment.yml`

to do hyperparameter searches set up an `orion` database with
```
orion setup-db

```

and select `pickleddb` and a location to save it to as your `host`

## Run
We use `gin-config` so all configurations can be easily viewed in `configs/`

`python src/train.py --gin_file /path/to/your/config --gin_param overriding_param=value`

## Hyperparameter Search

We use `Orion` which blends nicely with `gin` to do hyperparameter optimization.

The search space for an experiment has the name `-search.gin` and can be run with

```
orion --debug hunt -n some-name \
      --working-dir /path/to/savedir \
      --max-trials number-of-hyperparams-tried \
      src/orion_runs.py \
      --config /path/to/config-search.gin \
      --gin_param overrriding_param=value
```


