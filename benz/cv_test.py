# coding=utf-8

import numpy as np
import pandas as pd
import time
from sklearn.cross_validation import KFold
from sklearn import metrics
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor


def cv_test(train,cv=5):

    t0 = time.time()
    target = 'y'
    predictors = [x for x in train.columns if x not in ['ID', 'y']]
    train_X = train[predictors]
    train_Y = train[target]


    mean_r2 = []
    kf = KFold(len(train_Y), n_folds = cv, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):

        x_train = train_X.iloc[train_index]
        x_test = train_X.iloc[test_index]
        y_train = train_Y.iloc[train_index]
        y_test = train_Y.iloc[test_index]


        lgb_model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=675,
                              max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5,
                              min_child_samples=10, subsample=0.995, subsample_freq=1, colsample_bytree=1, reg_alpha=0,
                              reg_lambda=0, seed=0, nthread=-1, silent=True)
        xgb_model = XGBRegressor(max_depth=4, learning_rate=0.0045, n_estimators=1250,
                             silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                             max_delta_step=0, subsample=0.93, seed=27)
        xgb_model.fit(x_train, y_train)

        pred = xgb_model.predict(x_test)
        from sklearn.metrics import r2_score
        score = r2_score(y_test, pred)
        mean_r2.append(score)
        print ('{0}: r2:{1}\n\n'.format(i+1, score))



    print (u'r2-均值：%s'%(np.array(mean_r2).mean()))
    print ('Done in %.1fs!' % (time.time()-t0))

    return None


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

# 过采样
def cv_test_oversample(train,cv=5):

    t0 = time.time()
    mean_r2 = []
    kf = KFold(len(train), n_folds = cv, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):

        train1 = train.iloc[train_index]
        train2 = train.iloc[test_index]


        result = over_sample(train1,train2,'X3')

        from sklearn.metrics import r2_score
        score = r2_score(train2['y'], result['y'])
        mean_r2.append(score)
        print ('{0}: r2:{1}\n\n'.format(i+1, score))



    print (u'r2-均值：%s'%(np.array(mean_r2).mean()))
    print ('Done in %.1fs!' % (time.time()-t0))

    return None

# 对样本采样
def subfeat_predict(train,test,sub=0.75,repeat=20):
    predictors = [x for x in train.columns if x not in ['ID', 'y']]
    pred = np.zeros(test.shape[0])
    for i in range(repeat):
        import random
        random.seed(i)
        random.shuffle(predictors)
        predictors_sub = predictors[:int(len(predictors)*sub)]
        model = XGBRegressor(max_depth=4, learning_rate=0.0045, n_estimators=1250,
                             silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                             max_delta_step=0, subsample=0.93, seed=27)
        model.fit(train[predictors_sub], train['y'])
        pred_sub = model.predict(test[predictors_sub])
        pred += pred_sub
    pred = pred/repeat
    return pred