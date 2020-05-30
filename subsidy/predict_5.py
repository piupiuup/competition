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

#过采样加逐级二分类，失败。。。。。(已修正，非常棒)
def predict_5(train, test, silent=1):

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
    alg1 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)
    if silent == 0: print '非零的个数为：', n1000

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test['studentid'].values]
    test_2 = test[test_2_id]
    alg2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)
    if silent == 0: print '大于1000的个数为：', n1500

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test['studentid'].values]
    test_3 = test[test_3_id]
    alg3 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test_3[predictors])
    result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]})
    result3.sort_values(1,inplace=True,ascending=False)
    n2000 = int(len(test)*0.03)
    test_id_2000 = list(result3.head(n2000)['studentid'].values)
    if silent == 0: print '2000的个数：', n2000

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

    return pd.DataFrame({'studentid': test['studentid'], 'subsidy': subsidy})

    if silent==0: print result.groupby(result['subsidy']).size()

    return result






def predict_51(train, test, silent=1,alg=GradientBoostingClassifier(learning_rate=0.05,n_estimators=150)):
#GradientBoostingClassifier(learning_rate=0.05,n_estimators=500,max_depth=5,min_samples_leaf=30,subsample=0.6)
#GradientBoostingClassifier(learning_rate=0.05,n_estimators=300,max_depth=5,min_samples_leaf=40,subsample=0.7,random_state=30)
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
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)
    if silent == 0: print '非零的个数为：', n1000

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test['studentid'].values]
    test_2 = test[test_2_id]
    alg2 = alg
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)
    if silent == 0: print '大于1000的个数为：', n1500

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test['studentid'].values]
    test_3 = test[test_3_id]
    alg3 = alg
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test_3[predictors])
    result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]})
    result3.sort_values(1,inplace=True,ascending=False)
    n2000 = int(len(test)*0.03)
    test_id_2000 = list(result3.head(n2000)['studentid'].values)
    if silent == 0: print '2000的个数：', n2000

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

    return pd.DataFrame({'studentid': test['studentid'], 'subsidy': subsidy})

    if silent==0: print result.groupby(result['subsidy']).size()

    return result

def predict_52(train, test, silent=1,alg=XGBClassifier(learning_rate=0.05,n_estimators=150)):

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
    #alg1 = GradientBoostingRegressor(loss='ls')
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test['studentid'].values]
    test_2 = test[test_2_id]
    alg2 = alg
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test['studentid'].values]
    test_3 = test[test_3_id]
    alg3 = alg
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






#调参(2月7日提交)
def predict_53(train, test, silent=1):

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
    alg1 = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,  n_estimators=160, nthread=-1, subsample=0.6)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)
    if silent == 0: print '非零的个数为：', n1000

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test['studentid'].values]
    test_2 = test[test_2_id]
    alg2 = XGBClassifier(learning_rate=0.05,n_estimators=100,max_depth=2,min_child_weight=4,subsample=0.75)
    alg2.fit(train_2[predictors[:80]], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors[:80]])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)
    if silent == 0: print '大于1000的个数为：', n1500

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test['studentid'].values]
    test_3 = test[test_3_id]
    alg3 = XGBClassifier(learning_rate=0.05,n_estimators=100,max_depth=2,min_child_weight=4,subsample=0.75)
    alg3.fit(train_3[predictors[:80]], train_3[target])
    result3 = alg3.predict_proba(test_3[predictors[:80]])
    result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]})
    result3.sort_values(1,inplace=True,ascending=False)
    n2000 = int(len(test)*0.03)
    test_id_2000 = list(result3.head(n2000)['studentid'].values)
    if silent == 0: print '2000的个数：', n2000

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

    if silent==0: print result['subsidy'].value_counts()

    return result


#变形4
def predict_54(train, test, silent=1,alg=XGBClassifier(learning_rate=0.05,n_estimators=150)):

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    train_1['subsidy'] = train_1['subsidy'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_2 = train.copy()
    train_2['subsidy'] = train_2['subsidy'].map({0: 0, 1000: 0, 1500: 1, 2000: 1})

    train_3 = train.copy()
    train_3['subsidy'] = train_3['subsidy'].map({0: 0, 1000: 0, 1500: 0, 2000: 1})

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    # model of selecting id which big 0
    alg1 = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,  n_estimators=160, nthread=-1, subsample=0.6)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)

    # model of select id which big 1000
    alg2 = alg
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test[predictors])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)

    # model of select id which big 1000
    alg3 = alg
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test[predictors])
    result3 = pd.DataFrame({'studentid': test['studentid'].values, 1: result3[:, 1]})
    result3.sort_values(1,inplace=True,ascending=False)
    n2000 = int(len(test)*0.03)
    test_id_2000 = list(result3.head(n2000)['studentid'].values)

    # 将id对应的助学金合并起来
    subsidy = []
    for x in test['studentid']:
        if x in test_id_2000:
            subsidy.append(2000)
        elif x in test_id_1500:
            subsidy.append(1500)
        elif x in test_id_1000:
            subsidy.append(1000)
        else:
            subsidy.append(0)

    result =  pd.DataFrame({'studentid': test['studentid'], 'subsidy': subsidy})

    if silent==0: print result['subsidy'].value_counts()

    return result

#2月8日 调参
def predict_56(train, test, silent=1):

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
    alg1 = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,  n_estimators=160, nthread=-1, subsample=0.6)
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test[predictors])
    result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
    result1.sort_values(1,inplace=True,ascending=False)
    n1000 = int(len(test)*0.23)
    test_id_1000 = list(result1.head(n1000)['studentid'].values)
    if silent == 0: print '非零的个数为：', n1000

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test['studentid'].values]
    test_2 = test[test_2_id]
    alg2 = XGBClassifier(learning_rate=0.05,n_estimators=180,max_depth=2,min_child_weight=14,subsample=0.85)
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors])
    result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
    result2.sort_values(1,inplace=True,ascending=False)
    n1500 = int(len(test)*0.09)
    test_id_1500 = list(result2.head(n1500)['studentid'].values)
    if silent == 0: print '大于1000的个数为：', n1500

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test['studentid'].values]
    test_3 = test[test_3_id]
    alg3 = XGBClassifier(learning_rate=0.05,n_estimators=180,max_depth=2,min_child_weight=13,subsample=0.85)
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test_3[predictors])
    result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]})
    result3.sort_values(1,inplace=True,ascending=False)
    n2000 = int(len(test)*0.03)
    test_id_2000 = list(result3.head(n2000)['studentid'].values)
    if silent == 0: print '2000的个数：', n2000

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

    if silent==0: print result['subsidy'].value_counts()

    return result