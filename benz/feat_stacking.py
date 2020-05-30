# coding=utf-8

import numpy as np
import pandas as pd
import time
from sklearn.cross_validation import KFold
from sklearn import metrics
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression

# 对样本采样
def subfeat_stacking(train1,train2,test,sub=0.75,repeat=20):
    predictors = [x for x in train1.columns if x not in ['ID', 'y']]
    y_train2 = np.zeros((train2.shape[0], repeat))
    y_test = np.zeros((test.shape[0], repeat))
    for i in range(repeat):
        import random
        random.seed(i)
        random.shuffle(predictors)
        predictors_sub = predictors[:int(len(predictors)*sub)]
        model = XGBRegressor(max_depth=4, learning_rate=0.0045, n_estimators=1250,
                             silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                             max_delta_step=0, subsample=0.93, seed=27)
        model.fit(train1[predictors_sub], train1['y'])
        y_train2[:, i] = model.predict(train2[predictors_sub])
        y_test[:, i] = model.predict(test[predictors_sub])
    return y_train2,y_test

def feat_stacking(train,test,cv=5,sample=20):

    t0 = time.time()
    S_train = np.zeros((train.shape[0], sample))
    S_test = np.zeros((test.shape[0], sample))
    kf = KFold(train.shape[0], n_folds = cv, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):
        train1 = train.iloc[train_index]
        train2 = train.iloc[test_index]

        y_train2,y_test = subfeat_stacking(train1,train2,test,sub=0.8,repeat=20)
        S_train[test_index,:] = y_train2
        S_test[:,:] += y_test
        print(i)
    S_test = S_test/5

    model2 = LinearRegression()
    model2.fit(S_train,train['y'].values)
    y_pred = model2.predict(S_test)
    output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
    print ('Done in %.1fs!' % (time.time()-t0))

    return output