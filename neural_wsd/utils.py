import logging
import random
import subprocess
from functools import reduce
from pprint import pprint
from time import sleep

import numpy as np
import toolz
import torch
import yaml

SEED = 42
LOGGER = logging.getLogger(__name__)


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


def print_gpu_info(seconds_to_sleep=1):
    try:
        sleep(seconds_to_sleep)
        process = subprocess.Popen("/usr/bin/nvidia-smi", stdout=subprocess.PIPE)
        output, error = process.communicate()
        # Returning only the memory.
        output = output.decode().split("\n")[8].split("|")[2].strip()
        pprint(output)
    except FileNotFoundError:
        pass


def total_num_of_params(named_params):
    total = 0
    for n, p in named_params:
        total += reduce(np.multiply, p.shape)
    return total
