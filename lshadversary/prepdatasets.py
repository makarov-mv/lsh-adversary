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


def get_msweb():
    # get the data here
    # https://data.world/uci/anonymous-microsoft-web-data
    msweb = pd.read_csv("datasets/anonymous+microsoft+web+data/anonymous-msweb.data", low_memory=False, header=None, names=['A', 'B', 'C', 'D', 'E', 'F'])
    ids = msweb['B'][msweb['A'] == 'C']
    attr = msweb[msweb['A'] == 'A']

    attr_map = dict()
    for i in range(len(attr)):
        attr_map[attr['B'].iloc[i]] = i

    msweb_points = np.zeros((len(ids), len(attr_map)), dtype=int)
    cur_pos = -1
    for ir, row in msweb.iterrows():
        if row['A'] == 'C':
            cur_pos += 1
        elif row['A'] == 'V':
            msweb_points[cur_pos, attr_map[row['B']]] = 1
    return msweb_points

def get_mnist_binary():
    mnist_data = fetch_openml('mnist_784', parser='auto')
    mnist_points = (mnist_data['data'].to_numpy() > 0).astype(int)
    return mnist_points

def get_mushroom():
    mushroom_data = fetch_openml(data_id=24, parser='auto')
    return pd.get_dummies(mushroom_data.data).to_numpy(dtype=int)

def distort_dataset(points, target_dimension):
    if points.shape[1] > target_dimension:
        seed = 123
        rng = np.random.default_rng(seed=seed)
        mask = rng.choice(points.shape[1], size=target_dimension, replace=False)
        return points[:, mask]
    if points.shape[1] < target_dimension:
        points = np.tile(points, -(-target_dimension // points.shape[1]))[:, :target_dimension]
        return points
    return points
