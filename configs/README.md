# Gin config files for runs and hyperparameter searches

## Files
- `base.gin` is the base config file, all config files inherit but may override these configurations
- `cat-deter` is for REINFORCE agent (with Categorical output) and deterministic agents playing the discrete game
- `cat-recverlola` is for REINFORCE agent (with Categorical output) and a LOLA-trained deterministic agents playing the discrete game
- `senderlola-deter` is for a LOLA-trained DiCE agent (with Categorical output) and a deterministic agents playing the discrete game
- `senderlola-recverlola` is for a LOLA-trained DiCE agent (with Categorical output) and a LOLA-trained deterministic agents playing the discrete game
- `gauss-deter` is for a REINFORCE agent (with Gaussian output) and deterministic agents playing the continuous game
- `gauss-recverlola` is for a REINFORCE agent (with Gaussian output) and a LOLA-trained deterministic agents playing the continuous game
- `gausslola-deter` is for a LOLA-trained DiCE agent (with Gaussian output) and deterministic agents playing the continuous game
- `gausslola-recverlola` is for a LOLA-trained DiCE agent (with Gaussian output) and a LOLA-trained deterministic agents playing the continuous game
- `*-search.gin` is the hyperparameter search parameters for the given config
