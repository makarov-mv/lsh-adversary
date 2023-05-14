import numpy as np
from matplotlib import pyplot as plt
import sklearn.neighbors as skln
import pickle
import os
from tqdm.notebook import tqdm

def l1_norm(a):
    return np.linalg.norm(a, 1)

def l1_norm_int(a):
    return int(np.rint(np.abs(a).sum()))

class HammingLSH:
    def __init__(self, points, r1, r2, delta=None, l=None, random_gen=None, **kwargs):
        assert l is None or delta is None
        assert np.issubdtype(points.dtype, np.signedinteger)
        assert ((points == 0) | (points == 1)).all()
        assert random_gen is not None
        
        self._d = points.shape[1]
        self._n = points.shape[0]
        self._dtype = points.dtype
        self._r1 = r1
        self._r2 = r2
        self._k = int(np.floor(-np.log(self._n) / np.log(1 - r2 / self._d)))
        self._rho = np.log(1 - r1/self._d) / np.log(1 - r2/self._d)
        if l is None:
            self._l = int(np.ceil(np.power(self._n, self._rho) * np.log(1/delta)))
        else:
            self._l = l
            
        self._populate(points, random_gen)
        
    def _populate(self, points, random_gen):
        self._buckets = [dict() for i in range(self._l)]
        self._g_support = [random_gen.choice(self._d, self._k, replace=True) for i in range(self._l)]
        for p in points:
            for i in range(self._l):
                bucket = p[self._g_support[i]]
                if not bucket.tobytes() in self._buckets[i]:
                    self._buckets[i][bucket.tobytes()] = p
    
    def _validate_point(self, p):
        assert p.dtype == self._dtype
        assert ((p == 0) | (p == 1)).all()
        assert len(p) == self._d
        
    def query(self, q):
        self._validate_point(q)
        for i in range(self._l):
            bucket = q[self._g_support[i]]
            p = self._buckets[i].get(bucket.tobytes())
            if p is not None and np.linalg.norm(p - q, 1) <= self._r2:
                return p
        return None
    
    def cnt_mutual_buckets(self, q, z):
        self._validate_point(q)
        self._validate_point(z)
        ans = 0
        for i in range(self._l):
            q_buck = q[self._g_support[i]]
            z_buck = z[self._g_support[i]]
            if np.array_equal(q_buck, z_buck):
                ans += 1
        return ans
    
    def cnt_good_buckets(self, z):
        self._validate_point(z)
        ans = 0
        for i in range(self._l):
            z_buck = z[self._g_support[i]]
            if np.array_equal(z, z_buck):
                ans += 1
        return ans

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
#         self.n = None
#         self.d = None
        self.points = None
        self.nn_checker = None
        self.lsh = None
#         self._point_seed = None
#         self._lsh_seed = None
#         self._point_rng = None
#         self._lsh_rng = None
#         self._point_type = None
        self._point_params = None
        self._lsh_params = None
        
    def prepare(self, point_params, lsh_params):
        need_to_change_points = point_params != self._point_params
        need_to_change_lsh = need_to_change_points or self._lsh_params != lsh_params
        if need_to_change_points:
            self._point_params = point_params
            if self._point_params['point_type'] == "zero":
                self.points = np.zeros((point_params['n'], point_params['d']), dtype=int)
                self.nn_checker = ZeroChecker(point_params['d'])
            elif self._point_params['point_type'] == "random":
                seed = np.abs(hash(frozenset(point_params.items())))
                rng = np.random.default_rng(seed=seed)
                self.points = rng.binomial(1, 0.5, size=(point_params['n'], point_params['d']))
                self.nn_checker = skln.KDTree(self.points, metric='l1')
            else:
                raise ValueError
        if need_to_change_lsh:
            self._lsh_params = lsh_params
            seed = np.abs(hash(frozenset(lsh_params.items())))
            rng = np.random.default_rng(seed=seed)
            self.lsh = HammingLSH(self.points, random_gen=rng, **lsh_params)

class OutOfQueriesError(Exception):
    pass

def flip_bits(p, mask):
    p[mask] = 1 - p[mask]

def run_adaptive_alg(z, nn_checker, lsh, target_distance, t=None, max_resamples=1, max_queries=None, rng=None, **kwargs):
    assert rng is not None
    
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
            #         print("Found bit at dist", l1_norm(qr))
                if found_error:
                    break
    except OutOfQueriesError:
        pass
    if found_error:
        return found_error, total_queries, l1_norm(error_query), error_query
    else:
        return found_error, total_queries, None, None

def run_random_alg(z, nn_checker, lsh, target_distance, max_queries=None, rng=None, **kwargs):
    assert rng is not None
    assert max_queries is not None
    
    found_error = False
    error_query = None
    total_queries = 0

    def make_query(q):
        nonlocal total_queries
#         if max_queries is not None and total_queries >= max_queries:
#             raise OutOfQueriesError
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

def run_experiments(environment, point_params, lsh_params, experiment_params, data_dir):
    if data_dir is not None:
        name = str((sorted(experiment_params.items()), sorted(point_params.items()), sorted(lsh_params.items())))
        name = str(hash(name))
        name = name + ".pickle"
        name = os.path.join(data_dir, name)
        if os.path.exists(name):
            with open(name, 'rb') as handle:
                res = pickle.load(handle)
            return res
    
    seed = hash((frozenset(experiment_params.items()), frozenset(point_params.items()), frozenset(lsh_params.items())))
    seed = np.abs(seed)
    rng = np.random.default_rng(seed)
    
    
    if experiment_params['alg_type'] == "adaptive":
        exp_func = run_adaptive_alg
    elif experiment_params['alg_type'] == "random":
        exp_func = run_random_alg
    environment.prepare(point_params, lsh_params)
    cur_res = []
    for i in range(experiment_params['iter_num']):
        cur_res.append(exp_func(environment.points[0], environment.nn_checker, environment.lsh, rng=rng, **experiment_params))
    
    if data_dir is not None:
        with open(name, 'wb') as handle:
            pickle.dump(cur_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return cur_res

def process_results(res, batch_count=10, count_failures=False):
    success_prob = []
    err_succ_prob = []
    mean_queries = np.zeros(len(res))
    err_queries = np.zeros(len(res))
    for i, v in enumerate(res):
        successes = np.array([int(e[0]) for e in v]).reshape(batch_count, -1)
        success_prob.append(successes.mean())
        err_succ_prob.append(successes.mean(axis=1).std())
        
        if count_failures:
            queries = [e[1] for e in v]
        else:
            queries = [e[1] for e in v if e[0]]
        if len(queries) == 0:
            mean_queries[i] = np.nan
            err_queries[i] = np.nan
        else:
            mean_queries[i] = np.mean(queries)
            err_queries[i] = np.std(queries)
    return (success_prob, err_succ_prob), (mean_queries, err_queries)

def run_basic_grid_experiment(grid, exp_param_name, environment, point_params, lsh_params, exp_params, data_dir, disable_tqdm=False):
    res = []
    new_exp_param = exp_params.copy()    
    
    for val in tqdm(grid, disable=disable_tqdm):
        new_exp_param[exp_param_name] = val
        cur_res = run_experiments(environment, point_params, lsh_params, new_exp_param, data_dir=data_dir)
        res.append(cur_res)
    return res
