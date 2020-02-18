# Emergent Selfish Communication
emerging communication with selfish agents

## Setup
### Repo
Clone this repo and `cd` into the directory. I recommend installing in a `venv` (or `conda`, `virtualenv` ...)

```
python -m venv env
source env/bin/activate
```

I recommend treating the repo as a package and you can then install all dependecies with `pip`

```
pip install -e .
```

If you want to run the extra dependecies for the notebooks, add `notebook` to the end of the previous command.

### Hyperparameter Search (optional)
To facilitate hyperparameter searches, I use `orion`. If you want to set up a database you can do so with
```
orion setup-db

```

and select `pickleddb` and a location to save it to as your `host` e.g. `~/orion.db`

## Run
### Training
Train the models with `src/train.py`. We use `gin-config` so all configurations can be easily viewed in `configs/`. You can override any params of the configs when running by adding them in the `--gin_param` arg

`python src/train.py --gin_file /path/to/your/config --gin_param overriding_param=value`

The best hyperparameters for any given experiment are given in the `.gin` config. See the `README.md` in `configs/` for more info.

### Hyperparameter Search

`orion` works nicely with `gin-config` to do hyperparameter optimization. To define the search space of our parameter we specify a distribution for the parameter and `orion`'s random search will draw a value for the parameter at each hyperparmeter seed.

Our search spaces are under `configs/` and have the `-search.gin` ending. You can run the hyperparameter search taking the mean over 5 seeds with

```
orion hunt -n some-name \
      --working-dir /path/to/savedir \
      --max-trials number-of-hyperparams-tried \
      src/orion_runs.py \
      --config /path/to/config-search.gin \
      --gin_param overrriding_param=value
```

If you don't care about using `orion`'s cli to check the best run, you can run the above as `orion --debug hunt ...` to eliminate the bottleneck of writing to the db.
