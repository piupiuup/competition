# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold



#助学金第一层stacking
#返回是否获奖的概率
def first_stacking(train,test):
    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    train_1['subsidy'] = train_1['subsidy'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    def StackModels(train, test, y, clfs, n_folds):  # train data (pd data frame), test data (pd date frame), Target data,

        train.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        skf = KFold(len(y), n_folds, shuffle=True)
        blend_train = np.zeros((train.shape[0], len(clfs)))  # Number of training data x Number of classifiers
        blend_test = np.zeros((test.shape[0], len(clfs)))  # Number of testing data x Number of classifiers
        for j, clf in enumerate(clfs):
            print ('Training classifier [%s]' % (j))
            for i, (tr_index, cv_index) in enumerate(skf):

                print ('stacking Fold [%s] of train data' % (i))

                # This is the training and validation set (train on 2 folders, predict on a 3d folder)
                X_train = train.iloc[tr_index, :]
                Y_train = y[tr_index]
                X_cv = train.iloc[cv_index, :]

                clf.fit(X_train, Y_train)
                pred = clf.predict_proba(X_cv)
                blend_train[cv_index, j ] = pred[:,1]

            clf.fit(train, y)
            pred = clf.predict_proba(test)

            blend_test[:, j ] = pred[:,1]

        return blend_train, blend_test


    def Trees_stacking(train, test, y):
        clf1 = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,
                              min_child_weight=4,  n_estimators=160, nthread=-1, subsample=0.6)

        clf2 = RandomForestClassifier(n_estimators=250, criterion='entropy', max_depth=15, min_samples_split=2,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.6,
                                      max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1,
                                      random_state=1301, verbose=0)

        clf3 = ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_depth=15,
                                    min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0, max_features=0.5,
                                    max_leaf_nodes=None, bootstrap=False, oob_score=False,
                                    n_jobs=-1, random_state=1301, verbose=0)

        clf4 = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=1,
                                                         min_weight_fraction_leaf=0.0, max_features=0.4,
                                                         random_state=1301),
                                  n_estimators=300, learning_rate=0.07, random_state=1301)

        clf5 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=3, min_samples_leaf=6,
                                   min_samples_split=150, subsample=1, random_state=30)

        clfs = [clf1, clf2, clf3, clf4, clf5]
        train_probs, test_probs = StackModels(train, test, y, clfs, 3)  # n_folds=3

        return train_probs, test_probs


    #进入主程序，哈哈
    train_raw = train_1[predictors]
    test_raw = test[predictors]
    y = train_1[target]
    # print(train_raw.shape)
    # print(test_raw.shape)

    meta_trees_train, meta_trees_test = Trees_stacking(train_raw, test_raw, y)
    clf_xgb = XGBClassifier()
    clf_xgb.fit(meta_trees_train, y)
    preds_xgb = clf_xgb.predict_proba(meta_trees_test)

    return preds_xgb