import gc
import os
import sys
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import multiprocessing
import lightgbm as lgb
from scipy import stats
from functools import partial
from dateutil.parser import parse
from lightgbm import LGBMClassifier
from collections import defaultdict
from sklearn.metrics import f1_score
from datetime import date, timedelta
from contextlib import contextmanager
from sklearn.metrics import recall_score
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score
from joblib import dump, load, Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  StratifiedKFold
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA




def xgb_cv(params, train_feat, test_feat, predictors, label='label', cv=5,stratified=True):
    print('开始CV 5折训练...')
    t0 = time.time()
    train_preds = np.zeros((len(train_feat), train_feat[label].nunique()))
    test_preds = np.zeros((len(test_feat), train_feat[label].nunique()))
    xgb_test = xgb.DMatrix(test_feat[predictors])
    models = []
    if stratified:
        folds = StratifiedKFold(n_splits= cv, shuffle=True, random_state=66)
    else:
        folds = KFold(n_splits= cv, shuffle=True, random_state=66)
    for i, (train_index, test_index) in enumerate(folds.split(train_feat,train_feat[label])):
        xgb_train = xgb.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
        xgb_eval = xgb.DMatrix(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

        print('开始第{}轮训练...'.format(i))
        params = {'objective': 'multi:softprob',
                 'eta': 0.1,
                 'max_depth': 6,
                 'silent': 1,
                 'num_class': 11,
                 'eval_metric': "mlogloss",
                 'min_child_weight': 3,
                 'subsample': 0.7,
                 'colsample_bytree': 0.7,
                 'seed': 66
                 } if params is None else params
        watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]

        clf = xgb.train(params,
                        xgb_train,
                        num_boost_round=3000,
                        evals=watchlist,
                        verbose_eval=50,
                        early_stopping_rounds=50)

        train_preds[test_index] += clf.predict(xgb_eval)
        test_preds += clf.predict(xgb_test)
        models.append(clf)
    pickle.dump(models,open('xgb_{}.model'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),'+wb'))
    print('用时{}秒'.format(time.time()-t0))
    return train_preds,test_preds/5

def lgb_cv(params, train_feat, test_feat, predictors, label='label', cv=5,stratified=True):
    '''
    适用于二分类多分类需要修改
    :param params:
    :param train_feat:
    :param test_feat:
    :param predictors:
    :param label:
    :param cv:
    :param stratified:
    :return:
    '''
    print('开始CV 5折训练...')
    t0 = time.time()
    train_preds = np.zeros(len(train_feat))
    test_preds = np.zeros(len(test_feat))
    models = []
    if stratified:
        folds = StratifiedKFold(n_splits= cv, shuffle=True, random_state=66)
    else:
        folds = KFold(n_splits= cv, shuffle=True, random_state=66)
    for i, (train_index, test_index) in enumerate(folds.split(train_feat,train_feat[label])):
        lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
        lgb_eval = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

        print('开始第{}轮训练...'.format(i))
        params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'binary_logloss',
                    'num_class':11,
                    'max_depth': 8,
                    'num_leaves': 150,
                    'learning_rate': 0.05,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.95,
                    'bagging_freq': 5,
                    'verbose': 0,
                    'seed': 66,
                } if params is None else params

        clf = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        verbose_eval=50,
                        early_stopping_rounds=100)

        train_preds[test_index] += clf.predict(train_feat[predictors].iloc[test_index])
        test_preds += clf.predict(test_feat[predictors])
        models.append(clf)
    pickle.dump(models, open('xgb_{}.model'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), '+wb'))
    print('用时{}秒'.format(time.time() - t0))
    return train_preds, test_preds / cv




