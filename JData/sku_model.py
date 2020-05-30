# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import roc_auc_score
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import random
import time


# 获取n个样本（多退少补）
def get_n_sample(data, n):
    sample_positive = data[data['label']==1]
    sample_negative = data[data['label']!=1]
    index = list(np.random.permutation(sample_negative.shape[0]))
    if sample_negative.shape[0] >= n:
        sample_negetive = sample_negative.iloc[index[:n]]
    else:
        for i in range(4):
            index.extend(index)
        sample_negetive = sample_negative.iloc[index[:n]]
    sample = pd.concat([sample_positive,sample_negetive])

    return sample

# 调整样本个数
def sample(data,n=13):
    grouped = data.groupby('user_id')
    result = None
    for user_id,group in grouped:
        result_sub = get_n_sample(group, n)
        if result is None:
            result = result_sub
        else:
            result = pd.concat([result,result_sub])

    return result

def right_sku_test(train,test):

    t0 = time.time()
    target = 'label'
    predictors = [x for x in train.columns if x not in ['user_id', 'sku_id', 'label']]
    train_X = train[predictors]
    train_Y = train[target]
    test_X = test[predictors]

    params = {
        'objective': 'binary:logistic',
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'min_child_weight': 2,
        'max_depth': 3,
        'subsample': 0.6,
        'alpha': 10,
        'gamma': 30,
        'lambda':50,
        'silent': 1,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'auc',
        'scale_pos_weight': 8,
        'seed': 201703,
        'missing':-1
    }

    mean_auc = []
    pred_test = np.zeros(len(test_X))
    kf = KFold(len(train_Y), n_folds = 5, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):

        x_train = train_X.iloc[train_index]
        x_test = train_X.iloc[test_index]
        y_train = train_Y.iloc[train_index]
        y_test = train_Y.iloc[test_index]

        ## build xgb
        xgtrain = xgb.DMatrix(x_train, y_train)
        xgtest = xgb.DMatrix(x_test, y_test)
        watchlist = [(xgtrain,'train'), (xgtest, 'val')]
        gbdt = xgb.train(params, xgtrain, 5000, evals = watchlist, verbose_eval = 10, early_stopping_rounds = 100)

        pred = gbdt.predict(xgb.DMatrix(x_test))
        score = metrics.roc_auc_score(y_test, np.array(pred))
        mean_auc.append(score)
        print ('{0}: AUC:{1}\n\n'.format(i+1, score))

        # predict test
        pred_test += gbdt.predict(xgb.DMatrix(test_X))

    df_submit = pd.DataFrame(data = {'user_id': test['user_id'], 'sku_id':test['sku_id'], 'label':pred_test/5})
    df_submit.sort_values('label',ascending=False,inplace=True)

    print (u'AUC-均值：%s'%(np.array(mean_auc).mean()))
    print ('Done in %.1fs!' % (time.time()-t0))

    return df_submit
