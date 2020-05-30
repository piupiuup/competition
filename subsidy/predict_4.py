# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score

# 破特曼分享的模型
def predict_4(train, test, silent=1):
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

    # model
    clf = GradientBoostingClassifier(n_estimators=200)
    # clf = RandomForestClassifier(n_estimators=500,random_state=2016)
    clf = clf.fit(train[predictors], train[target])
    result = clf.predict(test[predictors])
    result = pd.DataFrame({'studentid': test['studentid'].values, 'subsidy': result})

    if silent==0: print result.groupby(result['subsidy']).size()

    return result


def predict_41(train, test, silent=1):
    #输入概率矩阵，设置输出数量，输出获奖结果
    def mapping2(data, p0=0.77, p1000=0.14, p1500=0.06, p2000=0.03):

        sum_of_p = float(p0 + p1000 + p1500 + p2000)
        sum_of_n = len(data)
        p0 = p0 / sum_of_p
        p1000 = p1000 / sum_of_p
        p1500 = p1500 / sum_of_p
        p2000 = p2000 / sum_of_p

        count1000 = int(sum_of_n * p1000)
        count1500 = int(sum_of_n * p1500)
        count2000 = int(sum_of_n * p2000)
        count_n = pd.Series({1000: count1000, 1500: count1500, 2000: count2000})

        value_count = data[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1).value_counts()
        # print value_count
        coefficient = pd.Series({1000: 0.2, 1500: 0.2, 2000: 0.2})
        while True:
            value_count_temp = value_count
            for i in [1000, 1500, 2000]:
                if value_count[i] > count_n[i]:
                    if coefficient[i] > 0:
                        coefficient[i] = coefficient[i] * (-0.5)
                elif value_count[i] < count_n[i]:
                    if coefficient[i] < 0:
                        coefficient[i] = coefficient[i] * (-0.5)
                else:
                    continue
                data[i] = data[i] * (1 + coefficient[i])

            # 选择概率最大值作为分类
            value_count = data[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1).value_counts()
            # print value_count
            if value_count.equals(value_count_temp):
                break

        result = data[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1)
        result = pd.DataFrame({'studentid':result.index,'subsidy':result.values})

        return result

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

    # model
    clf = GradientBoostingClassifier(n_estimators=200)
    # clf = RandomForestClassifier(n_estimators=500,random_state=2016)
    clf = clf.fit(train[predictors], train[target])
    result = clf.predict_proba(test[predictors])
    result = pd.DataFrame(result, index=test['studentid'].values, columns=[0, 1000, 1500, 2000])
    result = mapping2(result)

    if silent == 0: print result.groupby(result['subsidy']).size()

    return result