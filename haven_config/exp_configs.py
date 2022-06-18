from collections import ChainMap
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

TEST_GROUPS = {
    "test": [
        {
            "gin_config": ["configs/cat-deter-drift-bias3.gin"],
            "gin_param": [""],
            "savedir": "results/test/",
            "loaddir": "results/cat-deter-bias0/",
        }
    ]
}

HYPERPARAM_GROUPS = {
    f"hyperparam{bias}": [
        {
            "gin_config": ["configs/cat-deter-drift-search.gin"],
            "gin_param": [f"Game.bias = {bias}"],
            "savedir": f"results/cat-deter-drift-search-bias{bias}-{timestamp}/cat-deter-drift-bias{bias}-{i}",
            "loaddir": "results/cat-deter-bias0/",
        }
        for i in range(1)
    ]
    for bias in [0, 3, 6, 9, 12, 15, 18]
}

EXP_GROUPS = ChainMap(HYPERPARAM_GROUPS, TEST_GROUPS)
