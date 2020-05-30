import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

train = user2
test = user1
target = 'label'
predictors = [x for x in train.columns if x not in ['user_id','sku_id','label']]

params = {
        'objective': 'binary:logistic',
        'eta': 0.045,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 8,
        'subsample': 0.886,
        'alpha': 10,
        'gamma': 30,
        'lambda':50,
        'silent': 1,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'auc',
        'scale_pos_weight': 21,
        'seed': 201703,
        'missing':-1
    }
xgtrain = xgb.DMatrix(train[predictors], train[target])
xgtest = xgb.DMatrix(test[predictors], test[target])
watchlist = [(xgtrain,'train'), (xgtest, 'val')]
gbdt = xgb.train(params, xgtrain, 5000, evals = watchlist, verbose_eval = 10, early_stopping_rounds = 100)


def get_feat_imp(train,ID=['user_id','sku_id'],target='label'):

    ID_and_target = ID.copy()
    ID_and_target.append(target)
    predictors = [x for x in train.columns if x not in ID_and_target]
    model = XGBClassifier( learning_rate =0.06,n_estimators=140, max_depth=4,
                           min_child_weight=1, gamma=0,subsample=0.9, colsample_bytree=0.8,
                           scale_pos_weight=1, seed=27)
    model.fit(train[predictors],train[target])
    feat_imp = pd.Series(model.feature_importances_,index=predictors).sort_values(ascending=False)
    feat_imp.head(20).plot(kind='bar', title='Feature Importances',figsize=(15,5))
    plt.ylabel('Feature Importance Score')

    return feat_imp
