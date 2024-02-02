import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm.autonotebook import tqdm
import sklearn.neighbors as skln
import gc
import pickle
import os
import pandas as pd
import scipy

from lshexperiment import *

DATA_DIR = 'data'
FIG_DIR = "figs"
DEFAULT_ITER_COUNT = 10


if DATA_DIR is not None and not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if FIG_DIR is not None and not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

def save_fig(name, fig=plt):
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches='tight')
    
# rng = np.random.default_rng(seed=67)
plt.rcParams["figure.figsize"] = (8,3)
plt.rcParams['figure.dpi'] = 300

def run_mnist(iter_num=DEFAULT_ITER_COUNT, part=0, iter_batch=0):
    env = Environment()

    # d_mnist_eff = 300

    point_params = {
        'n': 10000,
        'd': 784,
        'point_type': "mnist_binary",
        'seed_offset': 0
    }

    lsh_params = {
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * 0.15),
        'r2': int(point_params['d'] * 0.3)
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': iter_num,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    if part == 0 or part == 1:
        grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR)
    if part == 0 or part == 2:
        t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
        new_exp_param = exp_params.copy()
        res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR)

def run_msweb(iter_num=DEFAULT_ITER_COUNT, part=0, iter_batch=0):
    env = Environment()


    point_params = {
        'n': 10000,
        'd': 294,
        'point_type': "msweb",
        'seed_offset': 0
    }

    lsh_params = {
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * 0.15),
        'r2': int(point_params['d'] * 0.3)
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': iter_num,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    if part == 0 or part == 1:
        grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR)

        msweb_r_data = prepare_plot_data(res, grid)

    if part == 0 or part == 2:
        t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
        new_exp_param = exp_params.copy()
        res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR)

import sys
from multiprocessing import Pool

def run_mnist_comp(x):
    t =  run_mnist(iter_batch=x)


def run_msweb_comp(x):
    return run_msweb(iter_batch=x)

if __name__ == "__main__":

    assert len(sys.argv) >= 2
    task = list(range(100))
    with Pool(int(sys.argv[1])) as p:
        print("MNIST")
        p.map(run_mnist_comp, task)
        print("MSWeb")
        p.map(run_msweb_comp, task)
        
        
    # if sys.argv[1] == 'mush':
    #     print("Mushroom")
    #     run_mushroom()
    # if sys.argv[1] == 'sparse':
    #     print("sparse")
    #     run_sparse()
    # if sys.argv[1] == 'zero':
    #     print("Zero")
    #     run_zero()
    # if sys.argv[1] == 'random':
    #     print("Random")
    #     run_random()
    # if sys.argv[1] == 'sparseall':
    #     print("Sparse all")
    #     run_sparsity_all(sys.argv[2], int(sys.argv[3]))
    # if sys.argv[1] == 'isolated':
    #     print("Isolated")
    #     run_isolated()
    # if sys.argv[1] == 'infliction':
    #     print("infliction")
    #     run_infliction()
    # if sys.argv[1] == 'querynum':
    #     print("querynum")
    #     run_querynum(point_type='random')
    