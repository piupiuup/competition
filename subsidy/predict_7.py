# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score

#恢复0.026最初的代码  线下评分十分低。。
def predict_7(train, test, silent=1):

    len_of_test = len(test)
    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    train_1['money'] = train_1['money'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})
    train_1 = train_1[train_1['studentid'] < 22073]
    test_1_1 = test[test['studentid'] < 22073]

    # model
    clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf.fit(train_1[predictors], train_1[target])
    result1 = clf.predict_proba(test_1_1[predictors])
    result1 = pd.DataFrame({'studentid':test_1_1['studentid'].values,'proba_of_1':result1[:,1]}).sort_values(by=['proba_of_1'],ascending=False)

    select_id = [id in result1.head(int(len_of_test*0.28))['studentid'].values for id in test['studentid'].values]
    clf2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf2.fit(train[train['money'] != 0][predictors], train[train['money'] != 0][target])
    result2 = clf2.predict(test[select_id][predictors])
    result2 = pd.DataFrame({'studentid': test[select_id]['studentid'].values, 'subsidy': result2})

    remain_id = [id not in result1.head(int(len_of_test*0.28))['studentid'].values for id in test['studentid'].values]
    result3 = pd.DataFrame({'studentid':test[remain_id]['studentid'],'subsidy':0})
    result = result2.append(result3).sort_values('studentid')
    if silent==0: print result.groupby(result['subsidy']).size()

    return result