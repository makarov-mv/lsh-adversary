import numpy as np
from matplotlib import pyplot as plt
import sklearn.neighbors as skln
import pickle
import os
from tqdm.autonotebook import tqdm
import hashlib
import scipy.stats
from sklearn.datasets import fetch_openml
import pandas as pd
import scipy

from lshadversary.prepdatasets import *
from lshadversary.lsh import *

CODE_VER = 2

def hash_repr(*params):
    m = hashlib.sha256()
    for p in params:
        m.update(str(sorted(p.items())).encode("utf-8"))
    return m

def int_repr(*params):
    return int.from_bytes(hash_repr(*params).digest(), 'little')

def l1_norm(a):
    return np.linalg.norm(a, 1)

def l1_norm_int(a):
    return int(np.rint(np.abs(a).sum()))

def get_most_isolated_point(points):
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points, metric='cityblock'))
    return (distances + np.eye(distances.shape[0]) * distances.max()).min(axis=0).argmax()

class ZeroChecker:
    def __init__(self, d):
        self._d = d
    
    def query(self, q, return_distance=True):
        assert q.shape[1] == self._d
        assert q.shape[0] == 1
        if return_distance:
            return np.zeros(self._d, dtype=int), [l1_norm_int(q[0])]
        else:
            return np.zeros(self._d, dtype=int)

class Environment:
    def __init__(self):
        self.points = None
        self.nn_checker = None
        self.lsh = None
        self._point_params = None
        self._lsh_params = None
        self._most_isolated_point = None # keep it lazily computed

    def get_most_isolated_point(self):
        if self._most_isolated_point is None:
            self._most_isolated_point = get_most_isolated_point(self.points)
        return self._most_isolated_point
        
    def prepare(self, point_params, lsh_params, rng):
        need_to_change_points = point_params != self._point_params
        need_to_change_lsh = need_to_change_points or self._lsh_params != lsh_params
        if need_to_change_points:
            self._most_isolated_point = None
            self._point_params = point_params.copy()
            if self._point_params['point_type'] == "zero":
                self.points = np.zeros((point_params['n'], point_params['d']), dtype=int)
                self.nn_checker = ZeroChecker(point_params['d'])
            elif self._point_params['point_type'] == "random":
                prob = self._point_params.get('sample_probability', 1/2)
                self.points = rng.binomial(1, prob, size=(point_params['n'], point_params['d']))
                self.nn_checker = skln.KDTree(self.points, metric='l1')
            elif self._point_params['point_type'] == "mnist_binary":
                # base dim 784
                assert point_params['d'] == 784
                assert point_params['n'] <= 70000
                
                mnist_points = get_mnist_binary()
                mnist_points = distort_dataset(mnist_points, point_params['d'])

                self.points = mnist_points[:point_params['n']]
                self.nn_checker = skln.KDTree(self.points, metric='l1')
            elif self._point_params['point_type'] == "msweb":
                # base dim 294
                assert point_params['d'] == 294
                assert point_params['n'] <= 32711

                msweb_points = get_msweb()
                msweb_points = distort_dataset(msweb_points, point_params['d'])

                self.points = msweb_points[:point_params['n']]
                self.nn_checker = skln.KDTree(self.points, metric='l1')  
            elif self._point_params['point_type'] == "mushroom":
                # base dim 116
                assert point_params['d'] == 116
                assert point_params['n'] <= 8124

                mush_points = get_mushroom()
                mush_points = distort_dataset(mush_points, point_params['d'])
                self.points = mush_points[:point_params['n']]
                self.nn_checker = skln.KDTree(self.points, metric='l1')
            else:
                raise ValueError
        if need_to_change_lsh:
            self._lsh_params = lsh_params.copy()
            if lsh_params.get("use_ade", False):
                self.lsh = ADENN(self.points, random_gen=rng, **lsh_params)
            else:
                if "ensemble_size" in lsh_params.keys():
                    assert 'buckets' not in lsh_params.keys()
                    if "query_samples" not in lsh_params.keys():
                        self.lsh = HammingLSHEnsemble(self.points, random_gen=rng, **lsh_params)
                    else:
                        self.lsh = HammingLSHDP(self.points, random_gen=rng, **lsh_params)
                else:
                    if lsh_params.get('buckets', 1) == 'max':
                        self.lsh = HammingLSHLargeBucket(self.points, random_gen=rng, **lsh_params)
                    elif lsh_params.get('buckets', 1) == 1:
                        self.lsh = HammingLSH(self.points, random_gen=rng, **lsh_params)
                    else:
                        raise ValueError
            
class OutOfQueriesError(Exception):
    pass

def flip_bits(p, mask):
    p[mask] = 1 - p[mask]

def run_adaptive_alg(z, nn_checker, lsh, target_distance, t=None, max_resamples=1, max_queries=None, rng=None, **kwargs):
    assert rng is not None
    if target_distance is None:
        target_distance = lsh._r1

    if t is None:
        t = target_distance
    found_error = False
    error_query = None
    total_queries = 0

    def make_query(q):
        nonlocal total_queries
        if max_queries is not None and total_queries >= max_queries:
            raise OutOfQueriesError
        total_queries += 1
        res2 = lsh.query(q)
        return res2
    
    def check_query(q):
        res_dist, res1 = nn_checker.query(q[None, :], return_distance=True)
        res2 = make_query(q)
        if res_dist[0] <= target_distance and res2 is None or res2 is not None and l1_norm(res2 - q) > lsh._r2:
            nonlocal found_error, error_query
            found_error = True
            error_query = q
        return res2
    
    dist = np.max((0, target_distance - t))
    num_resample = 0
    bound_iterations = (max_queries is None)
    inner_iterations = 0
    max_inner_iterations = 3 * (target_distance - dist + 1)
    try:
        while not found_error and num_resample < max_resamples:
            num_resample += 1
            q = z.copy()
            flip_bits(q, rng.choice(lsh._d, dist, replace=False))

            while l1_norm_int(q - z) < target_distance:
                if check_query(q) is None:
                    break
                q1 = np.copy(q)
                next_bit = None
                found_bit = False

                failed = True
                while not bound_iterations or inner_iterations < max_inner_iterations:
                    inner_iterations += 1
                    qr = np.copy(q1)
                    flip_bits(qr, rng.choice(np.argwhere(q1 == z), lsh._r2 - l1_norm_int(q1 - z), replace=False).flatten())
                    if make_query(qr) is None:
                        failed = False
                        break

                if failed:
                    break

                ql = q1

                while l1_norm_int(qr - ql) > 1:
                    qm = np.copy(ql)
                    flip_bits(qm, rng.choice(np.argwhere(qr != ql), l1_norm_int(qr - ql) // 2, replace=False).flatten())
                    if make_query(qm) is None:
                        qr = qm
                    else:
                        ql = qm

                flip_bits(q, np.argwhere(qr != ql)[0])
                if found_error:
                    break
    except OutOfQueriesError:
        pass
    if found_error:
        if kwargs.get("control_accuracy", False):
            resample_failures = 0
            for i in range(kwargs['control_resamples']):
                p = lsh.query(error_query)
                if p is None:
                    resample_failures += 1
            return found_error, total_queries, l1_norm(error_query), error_query, resample_failures
        else:
            return found_error, total_queries, l1_norm(error_query), error_query
    else:
        if kwargs.get("control_accuracy", False):
            return found_error, total_queries, None, None, 0
        else:
            return found_error, total_queries, None, None

def run_random_alg(z, nn_checker, lsh, target_distance, max_queries=None, rng=None, **kwargs):
    assert rng is not None
    assert max_queries is not None
    
    if target_distance is None:
        target_distance = lsh._r1

    found_error = False
    error_query = None
    total_queries = 0

    def make_query(q):
        nonlocal total_queries
        total_queries += 1
        res2 = lsh.query(q)
        return res2
    
    def check_query(q):
        res_dist, res1 = nn_checker.query(q[None, :], return_distance=True)
        res2 = make_query(q)
        if res_dist[0] <= target_distance and res2 is None or res2 is not None and l1_norm(res2 - q) > lsh._r2:
            nonlocal found_error, error_query
            found_error = True
            error_query = q
        return res2
    
    while not found_error and total_queries < max_queries:
        q = z.copy()
        flip_bits(q, rng.choice(lsh._d, target_distance, replace=False))
        check_query(q)
    
    if found_error:
        return found_error, total_queries, l1_norm(error_query), error_query
    else:
        return found_error, total_queries, None, None

def run_experiments_batches(environment, point_params, lsh_params, experiment_params, data_dir, batches):
    res = []
    for batch_num in np.arange(batches):
        new_exp_param = experiment_params.copy()
        new_point_params = point_params.copy()
        new_lsh_params = lsh_params.copy()
        
        new_exp_param['iter_batch'] = batch_num
    
            
        cur_res = run_experiments(environment, new_point_params, new_lsh_params, new_exp_param, data_dir=data_dir)
        res.extend(cur_res)
        
    return res


def run_experiments(environment, point_params, lsh_params, experiment_params, data_dir):
    if data_dir is not None:
        name = hash_repr(experiment_params, point_params, lsh_params).hexdigest()
        if CODE_VER > 0:
            name = name + "_ver_" + str(CODE_VER)
        name = name + ".pickle"
        name = os.path.join(data_dir, name)
        if os.path.exists(name):
            with open(name, 'rb') as handle:
                res = pickle.load(handle)
            return res
    
    seed = int_repr(experiment_params, point_params, lsh_params)
    rng = np.random.default_rng(seed)
    
    assert not (experiment_params.get('origin', 'random') == 'isolated' and experiment_params.get('change_points', False))

    if experiment_params['alg_type'] == "adaptive":
        exp_func = run_adaptive_alg
    elif experiment_params['alg_type'] == "random":
        exp_func = run_random_alg
    cur_res = []
    new_point_params = point_params.copy()
    new_lsh_params = lsh_params.copy()
    for i in range(experiment_params['iter_num']):
        if experiment_params.get('change_points', True):
            new_point_params['cur_iter'] = i
        if experiment_params.get('change_lsh', True):
            new_lsh_params['cur_iter'] = i
        environment.prepare(new_point_params, new_lsh_params, rng)
        match experiment_params.get('origin', 'first'):
            case 'first':
                origin_ind = 0
            case 'random':
                origin_ind = rng.choice(len(environment.points))
            case 'isolated':
                origin_ind = environment.get_most_isolated_point()
            case _:
                assert False
        cur_res.append(exp_func(environment.points[origin_ind], environment.nn_checker, environment.lsh, rng=rng, **experiment_params))
    
    if data_dir is not None:
        try:
            with open(name, 'wb') as handle:
                pickle.dump(cur_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except KeyboardInterrupt as e:
            os.remove(name)
            raise(e)
    
    return cur_res

def process_results(res):
    success_prob = []
    err_succ_prob = []
    mean_queries = np.zeros(len(res))
    err_queries = np.zeros(len(res))
    for i, v in enumerate(res):
        successes = np.array([int(e[0]) for e in v])
        success_prob.append(successes.mean())
        err_succ_prob.append(3 * scipy.stats.sem(successes)) # 99.5% confidence
        
        queries = np.array([e[1] for e in v])
        mean_queries[i] = queries.mean()
        err_queries[i] = 3 * scipy.stats.sem(queries) # 99.5% confidence
        
    return (success_prob, err_succ_prob), (mean_queries, err_queries)

def process_results_with_resamples(res, aggregation, **kwargs):
    means = np.zeros(len(res))
    errs = np.zeros(len(res))
    for i, v in enumerate(res):
        if aggregation == "query":
            values = np.array([e[1] for e in v])
        elif aggregation == "success":
            values = np.array([int(e[0]) for e in v])
        elif aggregation == "average": # mean fraction of resample failures
            values = np.array([e[4] / kwargs['resamples'] for e in v])
        elif aggregation == 'threshold':
            minfail = kwargs['resamples'] * kwargs['threshold']
            values = np.array([0 if e[4] < minfail else 1 for e in v])
        means[i] = values.mean()
        errs[i] = 3 * scipy.stats.sem(values) # 99.5% confidence
        
    return means, errs


def prepare_plot_data(res, grid, batch_count=10):
    prob_data, query_data = process_results(res, batch_count)
    return grid, prob_data, query_data

def process_results_with_grid(res, dist_grid, lsh_params, batch_count=10):
    prob_data, query_data = process_results(res, batch_count)
    dist_grid = lsh_params['r1'] - dist_grid
    return dist_grid, prob_data, query_data

def run_basic_grid_experiment(grid, param_name, environment, point_params, lsh_params, exp_params, data_dir, target="experiment", disable_tqdm=False, batches=None):
    if batches is None:
        batch_count = 1
    else:
        batch_count = batches
    all_res = []
    for batch_num in tqdm(np.arange(batch_count), miniters=1,disable=disable_tqdm or batches is None):
        res = []
        new_exp_param = exp_params.copy()
        new_point_params = point_params.copy()
        new_lsh_params = lsh_params.copy()
        
        if batches is not None:
            new_exp_param['iter_batch'] = batch_num
        
        for val in tqdm(grid, miniters=1, disable=disable_tqdm, leave=batches is None):
            if target == "experiment":
                new_exp_param[param_name] = val
            elif target == "points":
                new_point_params[param_name] = val
            elif target == "lsh":
                new_lsh_params[param_name] = val
            else:
                assert False
            
            cur_res = run_experiments(environment, new_point_params, new_lsh_params, new_exp_param, data_dir=data_dir)
            res.append(cur_res)
        
        all_res.append(res)
    
    final_res = []
    for i in range(len(all_res[0])):
        final_res.append([])
        for j in range(len(all_res)):
            final_res[i].extend(all_res[j][i])
    return final_res

def run_advanced_grid_experiment(grids, param_names, targets, environment, point_params, lsh_params, exp_params, data_dir, disable_tqdm=False, batches=None):
    if batches is None:
        batch_count = 1
    else:
        batch_count = batches
    all_res = []
    for batch_num in tqdm(np.arange(batch_count), miniters=1,disable=disable_tqdm or batches is None):
        res = []
        for vals in tqdm(list(zip(*grids)), miniters=1, disable=disable_tqdm, leave=batches is None):
            new_exp_param = exp_params.copy()
            new_point_params = point_params.copy()
            new_lsh_params = lsh_params.copy()
            if batches is not None:
                new_exp_param['iter_batch'] = batch_num

            for target, val, param_name in zip(targets, vals, param_names):
                if target == "experiment":
                    new_exp_param[param_name] = val
                elif target == "points":
                    new_point_params[param_name] = val
                elif target == "lsh":
                    new_lsh_params[param_name] = val
                else:
                    assert False
            cur_res = run_experiments(environment, new_point_params, new_lsh_params, new_exp_param, data_dir=data_dir)
            res.append(cur_res)
        all_res.append(res)
    final_res = []
    for i in range(len(all_res[0])):
        final_res.append([])
        for j in range(len(all_res)):
            final_res[i].extend(all_res[j][i])
    return final_res
