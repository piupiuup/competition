# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import roc_auc_score
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
import random
from sklearn.cross_validation import KFold

# 获取特征重要性
from xgboost.sklearn import XGBRegressor
def get_feat_imp(train,ID=['ID'],target='y'):

    predictors = [x for x in train.columns if x not in ['ID','y']]
    model = XGBRegressor( max_depth=4, learning_rate=0.0045, n_estimators=700,
                          silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                          max_delta_step=0, subsample=0.93, seed=27)
    model.fit(train[predictors],train[target])
    feat_imp = pd.Series(model.booster().get_fscore(),index=predictors).sort_values(ascending=False)
    return feat_imp

# 特征是否单调
def if_monotonicity(data,feat):
    result = data.groupby(feat,as_index=False)['y'].agg({'y_mean':'mean','n_count':'count'})
    return result

# 特征重编码
def recode(train,test,feat):
    feat_rank = train.groupby(feat,as_index=False)['y'].agg({'y_mean':'mean','n_count':'count'})
    feat_rank.sort_values('y_mean',inplace=True)
    feat_rank['rank'] = list(range(1,feat_rank.shape[0]+1))
    result1 = train.copy()
    result2 = test.copy()
    result1[feat] = result1[feat].map(dict(zip(feat_rank[feat].values,feat_rank['rank'].values)))
    result2[feat] = result2[feat].map(dict(zip(feat_rank[feat].values,feat_rank['rank'].values)))
    return result1,result2

# 重采样
def over_sample(train,test,feat):
    predictors = [x for x in train.columns if x not in ['ID', 'y']]
    groups = list(train[feat].unique())
    result = None
    for name in groups:
        train_temp = pd.concat([train,train[train[feat]==name]])
        test_temp = test[test[feat]==name]
        model = XGBRegressor(max_depth=4, learning_rate=0.0045, n_estimators=1250,
                         silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                         max_delta_step=0, subsample=0.93, seed=27)
        model.fit(train_temp[predictors], train_temp['y'])
        pred = model.predict(test_temp[predictors])
        if result is None:
            result = pd.DataFrame({'ID':test_temp['ID'].values,'y':pred})
        else:
            result = pd.concat([result,pd.DataFrame({'ID':test_temp['ID'].values,'y':pred})])
    result.sort_values('ID',inplace=True)

    return result