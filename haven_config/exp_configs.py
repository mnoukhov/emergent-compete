from collections import ChainMap
from datetime import datetime

import numpy as np
from scipy.stats import loguniform

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# fix the different random points used
np.random.seed(0)

BASE_SAVEDIR = '/mnt/public/results/emergent-compete'

HYPERPARAM_GROUPS = {
    f"hyperparam_bias{bias}": [
        {
            "gin_config": ["configs/cat-deter-drift-search.gin"],
            "gin_param": [
                f"Game.bias = {bias}",
                f"Reinforce.lr = {loguniform.rvs(1e-4, 1e-2)}",
                f"Deterministic.lr = {loguniform.rvs(1e-4, 1e-2)}",
                f"Reinforce.ent_reg = {loguniform.rvs(1e-4, 1)}",
            ],
            "savedir": f"/mnt/public/results/emergent-compete/cat-deter-drift-search-bias{bias}/cat-deter-drift-bias{bias}-{i}",
            "loaddir": "/mnt/public/results/emergent-compete/cat-deter-bias0",
        }
        for i in range(100)
    ]
    for bias in [0, 3, 6, 9, 12, 15, 18]
}

ALL_HPS = {
    "all_hyperparams": [
        config 
        for hyperparam_bias_configs in HYPERPARAM_GROUPS.values()
        for config in hyperparam_bias_configs
    ]
}

ALL_HPS_LOAD9 = {
    # update config and return dict
    "all_hyperparams_load9": [
        {
            **config, 
            **{
                'gin_config': ['configs/cat-deter-drift-search9.gin'],
                'savedir': config['savedir'].replace('search', 'search9'),
                'loaddir': "/mnt/public/results/emergent-compete/cat-deter-bias9/4/",
            }
        }
        for config in ALL_HPS['all_hyperparams']
    ]
}

TEST_GROUPS = {
    "test": [
        {
            "gin_config": ["configs/cat-deter-bias9.gin"],
            "gin_param": [""],
            "savedir": "/mnt/public/results/emergent-compete/cat-deter-bias9",
            "loaddir": None,
        }
    ],
    "test_hyperparam": HYPERPARAM_GROUPS["hyperparam_bias3"][:2],
    "test_loading": [
        {
            "gin_config": ["configs/cat-deter-bias0.gin"],
            "gin_param": ["train.measure_drift = True"],
            "savedir": "/mnt/public/results/emergent-compete/test_loading",
            "loaddir": "/mnt/public/results/emergent-compete/cat-deter-bias0",
        }
    ],
    "test_load9": ALL_HPS_LOAD9["all_hyperparams_load9"][:2]
}

EXP_GROUPS = ChainMap(HYPERPARAM_GROUPS, TEST_GROUPS, ALL_HPS, ALL_HPS_LOAD9)
