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

if DATA_DIR is not None and not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if FIG_DIR is not None and not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

def save_fig(name, fig=plt):
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches='tight')
    
# rng = np.random.default_rng(seed=67)
plt.rcParams["figure.figsize"] = (8,3)
plt.rcParams['figure.dpi'] = 300

def run_mnist(iter_num=1000, part=0):
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
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }
    if part == 0 or part == 1:
        grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
        run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, random_order=True)
    if part == 0 or part == 2:
        t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
        new_exp_param = exp_params.copy()
        run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, random_order=True)

def run_msweb(iter_num=1000, part=0):
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
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    if part == 0 or part == 1:
        grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
        run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR, random_order=True)

        msweb_r_data = prepare_plot_data(res, grid)

    if part == 0 or part == 2:
        t_grid = np.linspace(0, lsh_params['r1'] + 1, num=30, dtype=int)
        new_exp_param = exp_params.copy()
        run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR, random_order=True)

def run_mushroom(iter_num=1000):
    env = Environment()

    point_params = {
        'n': 8124,
        'd': 116,
        'point_type': "mushroom",
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
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR)

    mushroom_r_data = prepare_plot_data(res, grid)

    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    new_exp_param = exp_params.copy()
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR)
    return res, (point_params, lsh_params, exp_params)

def run_sparse(iter_num=1000):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'sample_probability': 1/15,
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
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR)

    sparse_r_data = prepare_plot_data(res, grid)

    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    new_exp_param = exp_params.copy()
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR)
    return res, (point_params, lsh_params, exp_params)

def run_zero(iter_num=1000):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "zero",
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
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR)

    zero_r_data = prepare_plot_data(res, grid)

    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    new_exp_param = exp_params.copy()
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params, new_exp_param, DATA_DIR)
    return res, (point_params, lsh_params, exp_params)

def run_random(iter_num=1000):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
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
        'change_points': True,
        'iter_num': iter_num,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    grid = np.linspace(1, lsh_params['r2'], num=30, dtype=int)
    res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, lsh_params, exp_params, data_dir=DATA_DIR)

    rand_r_data = prepare_plot_data(res, grid)

    t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
    res = run_basic_grid_experiment(t_grid, 't', env, point_params, lsh_params,exp_params, data_dir=DATA_DIR)
    return res, (point_params, lsh_params, exp_params)

def run_sparsity_all(type, num, iter_num=1000):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'sample_probability': 1/500,
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
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'origin': 'random',
    }
    if type=="t":
        p_grid = [1/2, 1/10, 25/300, 1/100, 1/10000]
        sample_prob_res = []

        for sample_prob in p_grid:
            cur_point_params = point_params.copy()
            cur_point_params['sample_probability'] = sample_prob
            t_grid = np.arange(0, lsh_params['r1'] + 1, 1)
            new_exp_param = exp_params.copy()
            res = run_basic_grid_experiment(t_grid, 't', env, cur_point_params, lsh_params, new_exp_param, DATA_DIR)
            sample_prob_res.append(res)
        return sample_prob_res, (point_params, lsh_params, exp_params)
    elif type == 'r':
        p_grid = [1/2, 1/10, 25/300, 1/100, 1/10000]

        grid = np.arange(0, lsh_params['r2'], 1)

        all_res = []

        # for sample_prob in p_grid:
        sample_prob = p_grid[num]
        cur_point_params = point_params.copy()
        cur_point_params['sample_probability'] = sample_prob
        res = run_basic_grid_experiment(grid, 'target_distance', env, cur_point_params, lsh_params, exp_params, data_dir=DATA_DIR)
        all_res.append(res)
        return all_res, (point_params, lsh_params, exp_params)
    else:
        assert False

def run_isolated(iter_num=1000):
    env = Environment()
    
    point_params_list = [
        {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'seed_offset': 0
        },
        {
        'n': 10000,
        'd': 300,
        'point_type': "zero",
        'seed_offset': 0
        },
        {
        'n': 10000,
        'd': 300,
        'point_type': "random",
        'sample_probability': 1/15,
        'seed_offset': 0
        },
        {
        'n': 10000,
        'd': 784,
        'point_type': "mnist_binary",
        'seed_offset': 0
        },
        {
        'n': 8124,
        'd': 116,
        'point_type': "mushroom",
        'seed_offset': 0
        },
        {
        'n': 10000,
        'd': 294,
        'point_type': "msweb",
        'seed_offset': 0
        }
    ]

    dataset_names = ['Random', "Zero", "Sparse random", "MNIST", "Mushroom", "MS Web"]

    lsh_params = {
        'delta': 1/16,
        'seed_offset': 0,
    #     'r1': int(point_params['d'] * 0.15),
    #     'r2': int(point_params['d'] * 0.3)
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': iter_num,
        'seed_offset': 0,
        # 'target_distance': lsh_params['r1'],
        'origin': 'random',
    }

    grid = ['random', 'isolated']
    all_res_prob = []
    all_res_queries = []


    for point_params in point_params_list:
        new_lsh_params = lsh_params.copy()
        new_lsh_params['r1'] = int(point_params['d'] * 0.15)
        new_lsh_params['r2'] = int(point_params['d'] * 0.3)
        new_exp_params = exp_params.copy()
        new_exp_params['target_distance'] = new_lsh_params['r1']
        
        res = run_basic_grid_experiment(grid, 'origin', env, point_params, new_lsh_params, new_exp_params, DATA_DIR)
        prob, queries = process_results(res)
        all_res_prob.append(prob)
        all_res_queries.append(queries)

    # all_res = list(zip(*all_res))

def run_infliction(iter_num=1000):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': "zero",
        'seed_offset': 0
    }

    lsh_params = {
        'delta': 1/2,
        'seed_offset': 0,
        'r1': point_params['d'] // 10,
        'r2': point_params['d'] // 5
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': iter_num,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
    }

    help_list = np.arange(1, 14, 3)
    delta_list = np.exp2(-help_list)

    my_rho = np.log(1 - lsh_params['r1']/point_params['d']) / np.log(1 - lsh_params['r2']/point_params['d'])
    my_L = int(np.ceil(np.power(point_params['n'], my_rho)))
    L_list = help_list * my_L

    grid = np.arange(0, lsh_params['r2'], 1)

    all_res = []

    for dl in delta_list:
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        res = run_basic_grid_experiment(grid, 'target_distance', env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR)
        all_res.append(res)

def run_querynum(iter_num=1000, point_type='random'):
    env = Environment()

    point_params = {
        'n': 10000,
        'd': 300,
        'point_type': point_type,
        'seed_offset': 0
    }

    lsh_params = {
        'delta': 1/2,
        'seed_offset': 0,
        'r1': point_params['d'] // 10,
        'r2': point_params['d'] // 5
    }

    exp_params = {
        'alg_type': 'adaptive',
        'change_lsh': True,
        'iter_num': iter_num,
        'seed_offset': 0,
        'target_distance': lsh_params['r1'],
        'max_queries': 80000,
        'max_resamples': 40000,
    }

    delta_list = np.logspace(1, 8, 8, endpoint=True, base=1/2)
    my_rho = np.log(1 - lsh_params['r1']/point_params['d']) / np.log(1 - lsh_params['r2']/point_params['d'])
    my_L = int(np.ceil(np.power(point_params['n'], my_rho)))
    L_list = np.arange(1, 9) * my_L

    delta_adapt_res = []

    # for cur_delta in tqdm(delta_list):
    #     iters_per_point = 100
    #     cur_res = []
    #     for i in range(iters_per_point):
    #         lsh = HammingLSH(points, r1, r2, delta=cur_delta)
    #         cur_res.append(run_exp_fast(points[0], r1, nn_checker, lsh, max_resamples=30))
    #     delta_res.append(cur_res)

    for dl in tqdm(delta_list):
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        cur_res = run_experiments(env, point_params, new_lsh_params, exp_params, data_dir=DATA_DIR)
        delta_adapt_res.append(cur_res)

    delta_list = np.logspace(1, 8, 8, endpoint=True, base=1/2)
    delta_rand_res = []

    for dl in tqdm(delta_list):
        new_lsh_params = lsh_params.copy()
        new_lsh_params['delta'] = dl
        new_exp_params = exp_params.copy()
        new_exp_params['alg_type'] = 'random'
        cur_res = run_experiments(env, point_params, new_lsh_params, new_exp_params, data_dir=DATA_DIR)
        delta_rand_res.append(cur_res)

    return delta_rand_res
    


import sys

if __name__ == "__main__":
    assert len(sys.argv) >= 2
    if sys.argv[1] == 'mnist':
        print("MNIST")
        run_mnist(part=int(sys.argv[2]))
    if sys.argv[1] == 'msweb':
        print("MSWeb")
        run_msweb(part=int(sys.argv[2]))
    if sys.argv[1] == 'mush':
        print("Mushroom")
        run_mushroom()
    if sys.argv[1] == 'sparse':
        print("sparse")
        run_sparse()
    if sys.argv[1] == 'zero':
        print("Zero")
        run_zero()
    if sys.argv[1] == 'random':
        print("Random")
        run_random()
    if sys.argv[1] == 'sparseall':
        print("Sparse all")
        run_sparsity_all(sys.argv[2], int(sys.argv[3]))
    if sys.argv[1] == 'isolated':
        print("Isolated")
        run_isolated()
    if sys.argv[1] == 'infliction':
        print("infliction")
        run_infliction()
    if sys.argv[1] == 'querynum':
        print("querynum")
        run_querynum(point_type='random')
    