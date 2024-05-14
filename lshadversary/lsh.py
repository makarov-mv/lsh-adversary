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
            self._l = int(np.ceil(np.power(self._n, self._rho) * np.log2(1/delta)))
        else:
            self._l = l
        self._kwargs = kwargs.copy()
        self._populate(points, random_gen)
        
    def _populate(self, points, random_gen):
        self._buckets = [dict() for i in range(self._l)]
        if self._kwargs.get('unique_support', False):
            self._g_support = [random_gen.choice(self._d, np.min((self._d,self._k)), replace=False) for i in range(self._l)]
        else:
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
    
class HammingLSHLargeBucket:
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
            self._l = int(np.ceil(np.power(self._n, self._rho) * np.log2(1/delta)))
        else:
            self._l = l
        self._kwargs = kwargs.copy()
        self._populate(points, random_gen)
        
    def _populate(self, points, random_gen):
        self._buckets = [dict() for i in range(self._l)]
        if self._kwargs.get('unique_support', False):
            self._g_support = [random_gen.choice(self._d, np.min((self._d,self._k)), replace=False) for i in range(self._l)]
        else:
            self._g_support = [random_gen.choice(self._d, self._k, replace=True) for i in range(self._l)]
        for p in points:
            for i in range(self._l):
                bucket = p[self._g_support[i]]
                if not bucket.tobytes() in self._buckets[i]:
                    self._buckets[i][bucket.tobytes()] = [p]
                else:
                    self._buckets[i][bucket.tobytes()].append(p)
    
    def _validate_point(self, p):
        assert p.dtype == self._dtype
        assert ((p == 0) | (p == 1)).all()
        assert len(p) == self._d
        
    def query(self, q):
        self._validate_point(q)
        for i in range(self._l):
            bucket = q[self._g_support[i]]
            
            for p in self._buckets[i].get(bucket.tobytes(), []):
                if np.linalg.norm(p - q, 1) <= self._r2:
                    return p
        return None

class HammingLSHEnsemble:
    def __init__(self, points, ensemble_size=None, random_gen=None, **kwargs):  
        assert ensemble_size is not None
        assert random_gen is not None
        
        self._d = points.shape[1]
        self._n = points.shape[0]
        self._dtype = points.dtype
        self._r1 = kwargs["r1"]
        self._r2 = kwargs["r2"]

        self._num_copies = ensemble_size
        self._random_gen = random_gen
        self._kwargs = kwargs.copy()
        self._populate(points, random_gen)
        
    def _populate(self, points, random_gen):
        self._copies = []
        for i in range(self._num_copies):
            self._copies.append(HammingLSHLargeBucket(points=points, random_gen=random_gen, **self._kwargs))
    
    # def _validate_point(self, p):
    #     assert p.dtype == self._dtype
    #     assert ((p == 0) | (p == 1)).all()
    #     assert len(p) == self._d
        
    def query(self, q):
        # res = []
        # for lsh in self._copies:
        #     a = lsh.query(q)
        #     if a is not None:
        #         res.append(a)
        # if len(res) == 0:
        #     return None
        # else:
        #     return self._random_gen.choice(res)

        return self._copies[self._random_gen.choice(self._num_copies)].query(q)
    
class HammingLSHDP:
    def __init__(self, points, ensemble_size=None, random_gen=None, query_samples=None, privacy_epsilon=None, **kwargs):  
        # ensemble_size is approx query_samples * sqrt(T * log(n)) * privacy_epsilon
        assert ensemble_size is not None
        assert random_gen is not None
        assert privacy_epsilon is not None
        assert query_samples is not None
        
        self._d = points.shape[1]
        self._n = points.shape[0]
        self._dtype = points.dtype
        self._r1 = kwargs["r1"]
        self._r2 = kwargs["r2"]

        self._num_copies = ensemble_size
        self._query_samples = query_samples
        self._random_gen = random_gen
        self._privacy_epsilon = privacy_epsilon
        self._kwargs = kwargs.copy()

        self._populate(points, random_gen)
        
    def _populate(self, points, random_gen):
        self._copies = []
        for i in range(self._num_copies):
            self._copies.append(HammingLSHLargeBucket(points=points, random_gen=random_gen, **self._kwargs))
    
    # def _validate_point(self, p):
    #     assert p.dtype == self._dtype
    #     assert ((p == 0) | (p == 1)).all()
    #     assert len(p) == self._d
        
    def query(self, q):
        # res = []
        # for lsh in self._copies:
        #     a = lsh.query(q)
        #     if a is not None:
        #         res.append(a)
        # if len(res) == 0:
        #     return None
        # else:
        #     return self._random_gen.choice(res)
        success = []
        fail = 0
        for i in self._random_gen.choice(self._num_copies, size=self._query_samples, replace=False):
            query_result = self._copies[i].query(q)
            if query_result is None:
                fail += 1
            else:
                success.append(query_result)

        offset = np.rint(self._random_gen.laplace(scale=1/self._privacy_epsilon)).astype(int)
        if offset < 0:
            offset = 0
        offset -= fail
        if offset < 0 or len(success) == 0:
            return None
        else:
            return success[np.min((offset, len(success) - 1))]
