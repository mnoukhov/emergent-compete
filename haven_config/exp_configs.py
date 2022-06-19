from collections import ChainMap
from datetime import datetime

from scipy.stats import loguniform

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
            "savedir": f"/mnt/public/results/emergent-compete/cat-deter-drift-search-bias{bias}/cat-deter-drift-bias{bias}-{i}-{timestamp}",
            "loaddir": "results/cat-deter-bias0/",
        }
        for i in range(100)
    ]
    for bias in [0, 3, 6, 9, 12, 15, 18]
}

TEST_GROUPS = {
    "test": [
        {
            "gin_config": ["configs/cat-deter-drift-bias3.gin"],
            "gin_param": [""],
            "savedir": "/mnt/public/results/emergent-compete/test",
            "loaddir": "results/cat-deter-bias0/",
        }
    ],
    "test_hyperparam": HYPERPARAM_GROUPS["hyperparam_bias3"][:2],
}

EXP_GROUPS = ChainMap(HYPERPARAM_GROUPS, TEST_GROUPS)
