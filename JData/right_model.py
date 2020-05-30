# coding=utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.cross_validation import KFold
from sklearn import metrics
def right_test(train,test):

    t0 = time.time()
    target = 'label'
    predictors = [x for x in train.columns if x not in ['user_id', 'sku_id', 'label']]
    train_X = train[predictors]
    train_Y = train[target]
    test_X = test[predictors]

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
        gbdt = xgb.train(params, xgtrain, 5000, evals = watchlist, verbose_eval = 10, early_stopping_rounds = 100)

        pred = gbdt.predict(xgb.DMatrix(x_test))
        score = metrics.roc_auc_score(y_test, np.array(pred))
        mean_auc.append(score)
        print ('{0}: AUC:{1}\n\n'.format(i+1, score))

        # predict test
        pred_test += gbdt.predict(xgb.DMatrix(test_X))

    if ('sku_id' in test.columns):
        df_submit = pd.DataFrame(data={'user_id': test['user_id'],'sku_id': test['sku_id'], 'probability': pred_test / 5})
    else:
        df_submit = pd.DataFrame(data = {'user_id': test['user_id'],'probability':pred_test/5})
    df_submit.sort_values('probability',ascending=False,inplace=True)

    print (u'AUC-均值：%s'%(np.array(mean_auc).mean()))
    print ('Done in %.1fs!' % (time.time()-t0))

    return df_submit

def under_sample(data,n=5):
    data_positive = data[data['label'] == 1]
    data_negative = data[data['label'] != 1]
    kf = KFold(len(data_negative), n_folds=n, shuffle=True, random_state=520)
    data_list = []
    for i, (train_index, test_index) in enumerate(kf):
        data_negative_sub = data_negative.iloc[train_index]
        data_sub = pd.concat([data_negative_sub, data_positive])
        data_list.append(data_sub)
    return data_list

def right_test_sample(train,test,n=5):
    train_list = under_sample(train,n)
    result = None
    for train_sub in train_list:
        result_sub = right_test(train_sub,test)
        result_sub.set_index('user_id',inplace=True)
        if result is None:
            result = result_sub
        else:
            result['probability'] = result_sub['probability'] + result['probability']
    result['probability'] = result['probability']/n
    result.sort_values('probability',ascending=False,inplace=True)
    result.reset_index(inplace=True)

    return result


