# Gin config files for runs and hyperparameter searches

## Files
- `base.gin` is the base config file, all config files inherit but may override these configurations
- `cat-deter` is for REINFORCE agent (with Categorical output) and deterministic agents sending discrete messages
- `gauss-deter` is for a REINFORCE agent (with Gaussian output) and deterministic agents sending continuous messages
- `*-search.gin` is the hyperparameter search parameters for the given config
