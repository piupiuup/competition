# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score


def predict_55(train, test, silent=1):

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
    alg1 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])

    # model of select id which big 1000
    alg2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1)
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test[predictors])

    # model of select id which big 1000
    alg3 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1)
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test[predictors])

    result = pd.DataFrame({'studentid':test['studentid'], '>0':result1[:, 1],'>1000':result2[:, 1],'>1500':result3[:, 1]})

    return result

def mapping(data):
    n = len(data)

    # 筛选出大于0的id
    data1 = data.copy()
    data1.sort_values(['>0'], inplace=True, ascending=False)
    test_id_1000 = list(data1['studentid'].head(int(n*0.23)).values)

    # 筛选出大于1000的id
    data2 = data1[[idx in test_id_1000 for idx in list(data1['studentid'].values)]].copy()
    data2.sort_values(['>1000'], inplace=True, ascending=False)
    test_id_1500 = list(data2['studentid'].head(int(n*0.09)).values)

    # 筛选出大于1500的id
    data3 = data2[[idx in test_id_1500 for idx in list(data2['studentid'].values)]].copy()
    data3.sort_values(['>1500'], inplace=True, ascending=False)
    test_id_2000 = list(data3['studentid'].head(int(n*0.03)).values)

    # 将id对应的助学金合并起来
    subsidy = []
    for x in data['studentid']:
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

    result =  pd.DataFrame({'studentid': list(data['studentid'].values), 'subsidy': subsidy})
    print result['subsidy'].value_counts()

    return  result

def predict_55(train, test, silent=1,alg=XGBClassifier(learning_rate=0.05,n_estimators=150)):

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

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    # model of selecting id which big 0
    alg1 = alg
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])

    # model of select id which big 1000
    alg2 = alg
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test[predictors])

    # model of select id which big 1000
    alg3 = alg
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test[predictors])

    result = pd.DataFrame({'studentid':test['studentid'], '>0':result1[:, 1],'>1000':result2[:, 1],'>1500':result3[:, 1]})

    return result