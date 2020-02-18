# Gin config files for runs and hyperparameter searches

## Files
- `base.gin` is the base config file, all config files inherit but may override these configurations
- `cat-deter*` is for REINFORCE agent (with Categorical output) and deterministic agents sending discrete messages
- `gauss-deter*` is for a REINFORCE agent (with Gaussian output) and deterministic agents sending continuous messages
- `*-search.gin` is the hyperparameter search parameters for the given config
- `*-bias*.gin` are the best hyperparameters for that setup run with that bias


## Running
Reproduce the best run for any hyperparameters by using that `.gin` file as the config.
E.g if you wanted to reproduce the run for the game with discrete messages (`cat-deter`) and using a bias of `90` degrees (`bias9`) then run

```
src/orion_runs.py  --config ./configs/cat-deter-bias9.gin
```
