import os


JOB_CONFIG = {
    "account_id": os.environ["EAI_ACCOUNT_ID"],
    "image": "registry.console.elementai.com/snow.colab/cuda",
    "data": [
        "snow.mnoukhov.home:/mnt/home",
        "snow.colab.public:/mnt/public",
    ],
    "restartable": True,
    "resources": {
        "cpu": 4,
        "mem": 8,
        # 'gpu': 0,
    },
    "interactive": False,
    "bid": 9999,
}
