# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score


def predict_1(train, test, silent=1):
    train_1 = train.copy()
    test_1 = test.copy()
    train_1['subsidy'] = train_1['subsidy'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_1 = train_1.fillna(-1)
    test_1 = test_1.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train_1.columns if x not in [target]]

    # model
    clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf.fit(train_1[predictors], train_1[target])
    result1 = clf.predict_proba(test_1[predictors])
    result1 = pd.DataFrame({'studentid':test_1['studentid'].values,'proba_of_1':result1[:,1]}).sort(['proba_of_1'],ascending=False)

    select_id = [id in result1.head(2800)['studentid'].values for id in test_1['studentid']]
    clf2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf2.fit(train_1[train['subsidy'] != 0][predictors], train[train_1['subsidy'] != 0][target])
    result2 = clf2.predict(test_1[select_id][predictors])
    result2 = pd.DataFrame({'studentid': test_1[select_id]['studentid'].values, 'subsidy': result2})

    result = pd.merge(result1[['studentid']],result2,how='left',on='studentid').fillna(0)
    if silent==0: print result.groupby(result['subsidy']).size()

    return result