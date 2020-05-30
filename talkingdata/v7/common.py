import numpy as np
import pandas as pd
import pickle
import gc, os, time, sys

def get_pickled_columns(filename):
    with open(filename, "rb") as fp:
        col = pickle.load(fp)
    return col

def under_sample(X, y, pos_idx=None, neg_sample_rate=0.05, random_state=42, keep_order=False, which_batch=0):
    if neg_sample_rate>0.99:
        if pos_idx is None:
            return X, Y
        else:
            return X.iloc[pos_idx], y.iloc[pos_idx]
    else:
        print("Under-sampling, neg_sample_rate is {}...".format(neg_sample_rate))
        np.random.seed(random_state)
        if pos_idx is not None:
            y = y.iloc[pos_idx].astype(bool)
        else:
            y = y.astype(bool)
        idx_pos = np.array(y[y].index)
        idx_neg = np.array(y[~y].index)
        np.random.shuffle(idx_neg)
        batch_len = int(neg_sample_rate*len(idx_neg))
        idx_neg = idx_neg[which_batch*batch_len:min(len(idx_neg),(which_batch+1)*batch_len)]
        idx = np.concatenate([idx_pos, idx_neg])
        if keep_order:
            idx = np.sort(idx)
        else:
            np.random.shuffle(idx)
        return X.loc[idx], y.loc[idx]

class Timer(object):
    def __init__(self):
        import time
        self.t_begin = time.time()
    def reset(self):
        self.t_begin = time.time()
    def get_eclipse(self, reset=True):
        t = time.time()-self.t_begin
        if t>3:
            print("Timer message: {} seconds passed!".format(int(t)))
        else:
            print("Timer message: {:2f} seconds passed!".format(t))
        if reset:
            self.reset()
    def eclipse(self, reset=True):
        self.get_eclipse(reset)

def get_quantile_encoding(data, n_cuts=20):
    if type(data) is not list:
        data = [data]
    quantiles = np.percentile(data[0], np.linspace(0, 100, n_cuts+1)[1:-1])
    ans = []
    for tmp in data:
        encodes = np.zeros(len(tmp), dtype=np.uint8)
        for q in quantiles:
            encodes += tmp>q
        ans.append(encodes)
    return ans

def normalize(data):
    if type(data) is not list:
        data = [data]
    base = data[0]
    median = np.float32(base.median())
    std = np.float32(base.std())
    ans = []
    for tmp in data:
        t = (tmp.astype(np.float32)-median)/std
        t = np.nan_to_num(t)
        ans.append(t)
    return ans





