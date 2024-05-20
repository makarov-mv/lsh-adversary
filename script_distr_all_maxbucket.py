import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm.autonotebook import tqdm
import sklearn.neighbors as skln
import os
import pandas as pd
import scipy

from lshadversary.lshexperiment import *

DATA_DIR = 'data'
FIG_DIR = "figs"
DEFAULT_ITER_COUNT = 10


if DATA_DIR is not None and not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if FIG_DIR is not None and not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

def save_fig(name, fig=plt):
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches='tight')
    
plt.rcParams["figure.figsize"] = (8,3)
plt.rcParams['figure.dpi'] = 300

R1_FRAC_POST_EXP = 0.15
R2_FRAC_POST_EXP = 0.3

def run_infliction(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/2,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    # help_list = np.arange(1, 14, 3)
    help_list = np.array([1 ,2, 3, 4, 5, 6, 7, 10, 13])
    delta_list = np.exp2(-help_list)

    my_rho = np.log(1 - lsh_params['r1']/point_params['d']) / np.log(1 - lsh_params['r2']/point_params['d'])
    my_L = int(np.ceil(np.power(point_params['n'], my_rho)))
    L_list = help_list * my_L

    grid = np.arange(0, lsh_params['r2'], 1)

    all_res = []

    for dl in delta_list:
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
        all_res.append(res)

def run_ensemble(iter_batch):
    env = Environment()


    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        'ensemble_size': 1,
        'delta': 1/2,
        'seed_offset': 4,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    size_list=np.array([2, 4])


    grid = np.arange(0, lsh_params['r2'], 1)

    all_res = []

    for sz in size_list:
        new_lsh_params = lsh_params.copy()
        new_lsh_params['ensemble_size'] = sz
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
        all_res.append(res)

def run_dp(iter_batch, query_samples):
    env = Environment()


    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        'ensemble_size': 4,
        'delta': 1/2,
        'query_samples': query_samples,
        'privacy_epsilon': 1/2,
        'seed_offset': 4,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    size_list=np.array([4, 5, 6, 10])

    grid = np.arange(0, lsh_params['r2'], 1)

    all_res = []

    for sz in size_list:
        if sz < query_samples:
            continue
        new_lsh_params = lsh_params.copy()
        new_lsh_params['ensemble_size'] = sz
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
        all_res.append(res)

def run_ade(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        "use_ade": True,
        'delta': 1/2,
        'seed_offset': 4,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    help_list = np.array([1, 2, 4])
    delta_list = np.exp2(-help_list)

    grid = np.arange(0, lsh_params['r2'], 1)

    all_res_ade = []

    for dl in delta_list:
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
        all_res_ade.append(res)

def run_basic(iter_batch):
    env = Environment()

    point_params = {
        'n': 1000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': point_params['d'] // 10,
        'r2': point_params['d'] // 5
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_points': True,
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': None,
    }

    grid = np.geomspace(100, 10000, num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'n', env, point_params, lsh_params, exp_params, target='points', data_dir=DATA_DIR, disable_tqdm=True)

    grid = np.linspace(70, 200, num=20, dtype=int)
    res = run_advanced_grid_experiment([grid], ['d'], ['points'], env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)

    r1_grid = np.linspace(1, 70, num=40, dtype=int)
    r2_grid = r1_grid * 2

    res = run_advanced_grid_experiment([r1_grid, r2_grid], ['r1', 'r2'], ['lsh', 'lsh'], env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)


    r1_grid = np.linspace(1, 70, num=40, dtype=int)
    r2_grid = r1_grid * 2
    new_point_params = point_params.copy()
    new_point_params['point_type'] = 'zero'
    new_lsh_params = lsh_params.copy()

    res = run_advanced_grid_experiment([r1_grid, r2_grid], ['r1', 'r2'], ['lsh', 'lsh'], env, new_point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True),

    lambda_grid = np.arange(1, 14)
    new_point_params = point_params.copy()
    new_lsh_params = lsh_params.copy()

    res = run_advanced_grid_experiment([np.power(2.0, -lambda_grid)], ['delta'], ['lsh'], 
                                    env, new_point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
    
    c_grid = np.linspace(1.1, 2.5, num=40)
    r2_grid = (lsh_params['r1'] * c_grid).astype(int)
    new_point_params = point_params.copy()
    new_lsh_params = lsh_params.copy()

    res = run_advanced_grid_experiment([r2_grid], ['r2'], ['lsh'], 
                                    env, new_point_params, new_lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
    
    t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
    new_exp_param = exp_params.copy()
    run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, disable_tqdm=True)

def run_2d(iter_batch):
    env = Environment()
    point_params = {
        'n': 1000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_points': True,
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': None,
    }

    d_grid_size = 70
    r_grid_size = 70
    d_grid_1d = np.linspace(50, 300, num=d_grid_size, dtype=int)
    frac_grid = np.linspace(0, 0.3, num=r_grid_size)
    d_grid, frac_grid = np.meshgrid(d_grid_1d, frac_grid)

    r1_grid = np.ceil(d_grid * frac_grid)
    r1_grid = np.maximum(r1_grid, 1).astype(int)

    r2_grid = r1_grid * 2
    r2_grid = np.minimum(r2_grid, d_grid - 1)

    new_exp_params = exp_params.copy()

    res = run_advanced_grid_experiment([d_grid.flatten(), r1_grid.flatten(), r2_grid.flatten()], ['d', 'r1', 'r2'], ['points', 'lsh', 'lsh'], env, point_params, lsh_params, new_exp_params, data_dir=DATA_DIR, disable_tqdm=True)


    lambda_grid_size = 50
    r_grid_size = 50
    lambda_grid_1d = np.linspace(1, 14, num=lambda_grid_size)
    r1_grid = np.linspace(1, 70, num=r_grid_size).astype(int)

    lambda_grid, r1_grid = np.meshgrid(lambda_grid_1d, r1_grid)

    delta_grid = np.power(2.0, -lambda_grid)

    r2_grid = r1_grid * 2

    new_exp_params = exp_params.copy()

    res = run_advanced_grid_experiment([delta_grid.flatten(), r1_grid.flatten(), r2_grid.flatten()], ['delta', 'r1', 'r2'], ['lsh', 'lsh', 'lsh'], env, point_params, lsh_params, new_exp_params, data_dir=DATA_DIR, disable_tqdm=True)

def run_mnist(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 784,
        'point_type': "mnist_binary",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    
    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)
    
    t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
    new_exp_param = exp_params.copy()
    run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, disable_tqdm=True)

def run_msweb(iter_batch):
    env = Environment()


    point_params = {
        'n': 10000,
        'd': 294,
        'point_type': "msweb",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    
    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)

    t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
    new_exp_param = exp_params.copy()
    run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, disable_tqdm=True)

def run_mushroom(iter_batch):
    env = Environment()

    point_params = {
        'n': 8124,
        'd': 116,
        'point_type': "mushroom",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }
    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)


    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    new_exp_param = exp_params.copy()
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, disable_tqdm=True)

def run_sparse(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'sample_probability': 1/15,
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)


    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    new_exp_param = exp_params.copy()
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, disable_tqdm=True)
    

def run_zero(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "zero",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)


    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    new_exp_param = exp_params.copy()
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, disable_tqdm=True)
    

def run_random(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/16,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'change_points': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, disable_tqdm=True)

    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params,exp_params, data_dir=DATA_DIR, disable_tqdm=True)

def run_querynum(iter_batch):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': 'random',
        'seed_offset': 0
    }

    lsh_params = {
        'buckets': 'max',
        'delta': 1/2,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'change_points': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'max_queries': 100000,
        'max_resamples': 50000,
    }

    delta_list = np.logspace(1, 10, 10, endpoint=True, base=1/2)
    my_rho = np.log(1 - lsh_params['r1']/point_params['d']) / np.log(1 - lsh_params['r2']/point_params['d'])
    my_L = int(np.ceil(np.power(point_params['n'], my_rho)))
    L_list = np.arange(1, 9) * my_L

    delta_adapt_res = []

    for dl in tqdm(delta_list, leave=False, miniters=1, disable=True):
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        cur_res = run_experiments(env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR)
        delta_adapt_res.append(cur_res)

    delta_list = np.logspace(1, 8, 8, endpoint=True, base=1/2)
    delta_rand_res = []

    for dl in tqdm(delta_list, leave=False, miniters=1, disable=True):
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        new_exp_params = exp_params.copy()
        new_exp_params['alg_type'] = 'random'
        cur_res = run_experiments(env, point_params, new_lsh_params, new_exp_params, data_dir=DATA_DIR)
        delta_rand_res.append(cur_res)

def run_querynum_generic(iter_batch, point_params=None):
    env = Environment()

    lsh_params = {
        'buckets': 'max',
        'delta': 1/2,
        'seed_offset': 0,
        'r1': int(point_params['d'] * R1_FRAC_POST_EXP),
        'r2': int(point_params['d'] * R2_FRAC_POST_EXP),
    }

    exp_params = {
        'alg_type': 'random',
        'change_lsh': True,
        'change_points': True,
        'iter_num': DEFAULT_ITER_COUNT,
        'iter_batch': iter_batch,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'max_queries': 80000,
        'max_resamples': 40000,
    }

    delta_list = np.logspace(1, 8, 8, endpoint=True, base=1/2)
    delta_rand_res = []

    for dl in tqdm(delta_list, leave=False, miniters=1, disable=True):
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        new_exp_params = exp_params.copy()
        new_exp_params['alg_type'] = 'random'
        cur_res = run_experiments(env, point_params, new_lsh_params, new_exp_params, data_dir=DATA_DIR)
        delta_rand_res.append(cur_res)


def run_querynum_mushroom(iter_batch):
    point_params = {
        'n': 8124,
        'd': 116,
        'point_type': "mushroom",
        'seed_offset': 0
    }
    return run_querynum_generic(iter_batch, point_params)

def run_querynum_mnist(iter_batch):
    point_params = {
        'n': 10000,
        'd': 784,
        'point_type': "mnist_binary",
        'seed_offset': 0
    }
    return run_querynum_generic(iter_batch, point_params)
        
def run_querynum_msweb(iter_batch):
    point_params = {
        'n': 10000,
        'd': 294,
        'point_type': "msweb",
        'seed_offset': 0
    }
    return run_querynum_generic(iter_batch, point_params)

def run_querynum_sparse(iter_batch):
    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'sample_probability': 1/15,
        'seed_offset': 0
    }
    return run_querynum_generic(iter_batch, point_params)

def run_querynum_zero(iter_batch):
    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "zero",
        'seed_offset': 0
    }
    return run_querynum_generic(iter_batch, point_params)

def run_dp3(iter_batch):
    return run_dp(iter_batch, 3)

def run_dp4(iter_batch):
    return run_dp(iter_batch, 4)

def run_dp5(iter_batch):
    return run_dp(iter_batch, 5)

def run_dp6(iter_batch):
    return run_dp(iter_batch, 6)

import sys
from multiprocessing import Pool

def run_infliction_comp(x):
    return run_infliction(iter_batch=x)

def run_easy_sets(x):
    run_random(x)
    run_zero(x)
    run_sparse(x)
    run_mushroom(x)


if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser(description='Run experiments with LSH adversary.')

    parser.add_argument("--target", help="Which experiment to run", required=True)
    parser.add_argument("--threads", type=int, help="Number of threads to use",required=True)
    parser.add_argument("--batch_interval", nargs=2, type=int, help="The semi-interval of batches to execute. The whole interval is [0, 100)", required=True)

    args = parser.parse_args()

    # task_size = 100

    target = args.target

    if target == 'infliction':
        func = run_infliction_comp
    elif target == 'basic':
        func = run_basic
    elif target == '2d':
        func = run_2d
        task_size = 10 # TODO: fix!
    elif target == 'easysets':
        func = run_easy_sets
    elif target == 'mnist':
        func = run_mnist
    elif target == 'msweb':
        func = run_msweb
    elif target == 'querynum':
        func = run_querynum
    elif target == 'querynum_mnist':
        func = run_querynum_mnist
    elif target == 'querynum_msweb':
        func = run_querynum_msweb
    elif target == 'querynum_mushroom':
        func = run_querynum_mushroom
    elif target == 'querynum_zero':
        func = run_querynum_zero
    elif target == 'querynum_sparse':
        func = run_querynum_sparse
    elif target == 'ensemble':
        func = run_ensemble
    elif target == 'dp3':
        func = run_dp3
    elif target == 'dp4':
        func = run_dp4
    elif target == 'dp5':
        func = run_dp5
    elif target == 'dp6':
        func = run_dp6        
    elif target == 'ade':
        func = run_ade
    else:
        raise RuntimeError("Unkown experiment name")
 
    print(f"Running {args.target} for batches {args.batch_interval} with {args.threads} threads.")
    task = list(range(*args.batch_interval))
    with Pool(int(args.threads)) as p:
        for r in tqdm(p.imap(func, task), total=len(task), miniters=1, disable=False, file=sys.stdout):
            print("Job completed")
        
    