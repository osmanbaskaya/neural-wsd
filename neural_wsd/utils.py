import random

import numpy as np
import toolz
import torch
import yaml

SEED = 42


def configure_logger():
    import logging.config

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def merge_params(params, higher_priority_params):

    if params is None:
        params = {}
    if higher_priority_params is None:
        higher_priority_params = {}

    return toolz.merge(params, higher_priority_params)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)
