import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

target = 'label'
predictors = [x for x in train.columns if x not in ['user_id','sku_id','label']]
param_test1 = {
 'max_depth':[100,150,200,300,500]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.05, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])

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

    mean_auc = []
    F11 = []
    pred_test = np.zeros(len(test_X))
    kf = KFold(len(train_Y), n_folds = 5, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):

        x_train = train_X.iloc[train_index]
        x_test = train_X.iloc[test_index]
        y_train = train_Y.iloc[train_index]
        y_test = train_Y.iloc[test_index]

        ## build xgb
        xgtrain = xgb.DMatrix(x_train, y_train)
        xgtest = xgb.DMatrix(x_test, y_test)
        watchlist = [(xgtrain,'train'), (xgtest, 'val')]
        gbdt = xgb.train(params, xgtrain, 5000, evals = watchlist, verbose_eval = 50, early_stopping_rounds = 100)
        # predict test
        pred_test += gbdt.predict(xgb.DMatrix(test_X))
        # predict pred of train
        pred = gbdt.predict(xgb.DMatrix(x_test))

        ## cal auc value
        score = metrics.roc_auc_score(y_test, np.array(pred))
        mean_auc.append(score)
        print ('{0}: AUC:{1}\n\n'.format(i+1, mean_auc[-1]))

        # cal F1
        F = cal_f(pred, y_test, 'F11')
        F11.append(F[0])
        # sub important feature
        stat_f(gbdt, i, x_train)

    df_submit = pd.DataFrame(data = {'user_id': test['user_id']})
    df_submit['probability'] = pd.DataFrame(pred_test)
    df_submit['probability'] = df_submit['probability']/1.0/5

    df_submit2 = sdta(df_submit,0.695)
    df_submit2.to_csv(r'./result/过渡文件夹/MM1.csv', index = None)