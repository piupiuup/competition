# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score

#过采样加 新二分类
def predict_6(train, test, silent=1):
    # Oversample
    Oversampling1000 = train.loc[train.subsidy == 1000]
    Oversampling1500 = train.loc[train.subsidy == 1500]
    Oversampling2000 = train.loc[train.subsidy == 2000]
    for i in range(5):
        train = train.append(Oversampling1000)
    for j in range(8):
        train = train.append(Oversampling1500)
    for k in range(10):
        train = train.append(Oversampling2000)

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    train_1['subsidy'] = train_1['subsidy'].map({0: 1, 1000: 0, 1500: 0, 2000: 0})

    train_2 = train.copy()
    train_2['subsidy'] = train_2['subsidy'].map({0: 0, 1000: 1, 1500: 0, 2000: 0})

    train_3 = train.copy()
    train_3['subsidy'] = train_3['subsidy'].map({0: 0, 1000: 0, 1500: 1, 2000: 0})

    train_4 = train.copy()
    train_4['subsidy'] = train_4['subsidy'].map({0: 0, 1000: 0, 1500: 0, 2000: 1})

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    # model of selecting id which equal 0
    alg1 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])

    # model of select id which equal 1000
    alg2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1)
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test[predictors])

    # model of select id which equal 1500
    alg3 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1)
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test[predictors])

    # model of select id which equal 2000
    alg4 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=3, min_samples_leaf=6,min_samples_split=150, subsample=1, )
    alg4.fit(train_4[predictors], train_4[target])
    result4 = alg4.predict_proba(test[predictors])

    # 将id对应的助学金合并起来
    result =  pd.DataFrame({'studentid': test['studentid'], 0: result1[:,1], 1000:result2[:,1], 1500:result3[:,1], 2000:result4[:,1]})
    result['subsidy'] = result[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1)

    if silent==0: print result['subsidy'].value_counts()

    return result[['studentid','subsidy']]