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

## Reproducing Graphs
### Best Results (Figure 2,3)
To reproduce the graphs for the best hyperparameters of any particular experiment, you need to run 5 random seeds (using `src/orion_runs.py`) with the best hyperparameters found for that setup and then plot the results with the jupyter notebook `notebooks/Best Results Plot.ipynb`

E.g if you wanted to reproduce the run for the game with discrete messages (`cat-deter`) and using a bias of `90` degrees (`bias9`) saving to some `$SAVEDIR`

```
src/orion_runs.py  --config ./configs/cat-deter-bias9.gin --savedir $SAVEDIR
```

This will create folders `0-5` corresponding to each random seed in `$SAVEDIR`. Then call `plot($SAVEDIR)` from `Best Resuts Plot` to reproduce the plot for that run.

### Best Results Per Bias (Figure 2,3)

To reproduce the graph plotting the best result per bias, you need to run the five seeds for each bias and then save them in folders with the bias specified as `bias$BIAS`.
E.g if you wanted to reproduce the results for discrete messages (`cat-deter`), then make your `$SAVEDIR` a template such as `./cat-deter-results/cat-deter-bias$BIAS`

```
src/orion_runs.py  --config ./configs/cat-deter-bias0.gin --savedir ./cat-deter-results/cat-deter-bias0
src/orion_runs.py  --config ./configs/cat-deter-bias3.gin --savedir ./cat-deter-results/cat-deter-bias3
```

Next, in `Best Results Plot.ipynb` you can run `plot_hyperparam_results("./cat-deter-results/")` which will plot the best hyperparameters for each run as well as the graph of best result per bias

### All Hyperparameter Run (Figure 4)

This plots the sender vs receiver error for all hyperparameter searches so first you need to run 100 hyperparameter searches for the given hyperparameter search space.
E.g. if you wanted to get all 100 hyperparameter runs (`--max-trials 100`) for discrete messages (`configs/cat-deter-search.gin`) with bias 90 degrees (`Game.bias=9`) and the directory you're saving all runs in is `results/`

```
orion --debug hunt -n cat-deter-bias9 \
      --working-dir results/ \
      --max-trials 100 \
      src/orion_runs.py \
      --config configs/cat-deter-search.gin \
      --gin_param Game.bias=9
```

To recreate the continuous messages vs discrete messages plots you need to run this for all biases `0,3,6,9,12,15` and for both configs `cat-deter-search.gin` and `gauss-deter-search.gin`.
If you save your results in folders with template names (e.g. `cat-deter-bias$BIAS`) then you can use the notebook `Hyperparam Search Plots.ipynb` to create the Figure 4 plot.
For individual plots, you can call `all_metrics("../results/")` and make a scatterplot with the resulting values
