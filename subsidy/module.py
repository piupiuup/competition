# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score

'''
简单声明：本代码由本队共同商讨完成
本次算法 主要采用了三次二分类：
（1）第一层主要用来区分 是否获奖。              {0:(0),    1:(1000,1500,2000)}
（2）第二层主要用来区分 获得助学金是否大于1000  {0:(1000), 1:(1500,2000)}
（3）第三层主要用来区分 获得助学金是否          {0:(1500), 1:(2000)}
实现本算法最重要的地方在于如何确定人数比例。由于比较难于简单的表述，会在之后答辩ppt中做详细的介绍。
比赛过程中我们试过很多其他算法，更多内容我们都会在ppt中来介绍。
'''
def predict(train, test, silent=1):

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    train_1['subsidy'] = train_1['subsidy'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_2 = train.copy()
    train_2 = train_2[train['subsidy'] > 0]
    train_2['subsidy'] = train_2['subsidy'].map({1000: 0, 1500: 1, 2000: 1})

    train_3 = train.copy()
    train_3 = train_3[train['subsidy'] > 1000]
    train_3['subsidy'] = train_3['subsidy'].map({1500: 0, 2000: 1})


    # model of selecting id which big 0
    alg1 = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,
                          n_estimators=160, nthread=-1, subsample=0.6)
    #初赛最高分用的是GBDT单模型，但XGB效果基本也差不多；复赛单模型最高分用的是XGB。
    #GradientBoostingClassifier(learning_rate=0.05,n_estimators=300,max_depth=5,min_samples_leaf=40,subsample=0.7,random_state=30)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test['studentid'].values]
    test_2 = test[test_2_id]
    alg2 = XGBClassifier(learning_rate=0.05,n_estimators=150)
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test['studentid'].values]
    test_3 = test[test_3_id]
    alg3 = XGBClassifier(learning_rate=0.05,n_estimators=150)
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test_3[predictors])
    result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]})
    result3.sort_values(1,inplace=True,ascending=False)
    n2000 = int(len(test)*0.03)
    test_id_2000 = list(result3.head(n2000)['studentid'].values)

    # 将id对应的助学金合并起来
    subsidy = []
    for x in test['studentid']:
        if x in test_id_1000:
            if x in test_id_1500:
                if x in test_id_2000:
                    subsidy.append(2000)
                else:
                    subsidy.append(1500)
            else:
                subsidy.append(1000)
        else:
            subsidy.append(0)

    result = pd.DataFrame({'studentid': test['studentid'], 'subsidy': subsidy})

    if silent == 0: print result['subsidy'].value_counts()

    return result
