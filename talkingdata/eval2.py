from talkingdata.feat1 import *
from tool.tool import *
import gc
import time
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss

# channel 全天流量是否异常,工作时流量比上夜间流量

inplace = False
hours = [4,5,9,10,13,14]
data_path = 'C:/Users/cui/Desktop/python/talkingdata/data/'


train = pd.read_hdf(data_path + 'train.csv').drop('attributed_time',axis=1)
train = pre_treatment(train,'eval_all')


train_feat = pd.DataFrame()
for date in ['2017-11-07','2017-11-08']:
    train_feat_sub = make_feat(train[(train['date']==date) & (train['hour'].isin(hours))].copy(),
                               train[train['date']!=date].copy(),
                               train,'{}_all'.format(date),inplace)
    train_feat_sub = train_feat_sub[train_feat_sub['is_attributed'] == 1].append(
        train_feat_sub[train_feat_sub['is_attributed'] == 0].sample(frac=0.1, random_state=66))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

test_feat = make_feat(train[(train['date']=='2017-11-09') & (train['hour'].isin(hours))].copy(),
                      train[train['date']!='2017-11-09'].copy(),
                      train,'2017-11-09_all',inplace)

test_feat = test_feat.sample(frac=0.3,random_state=66)
gc.collect()

predictors = [c for c in train_feat.columns if c not in ['app_count','channel_count',
 'click_time','date','ip','is_attributed']]
# predictors = ['app', 'device', 'os', 'channel', 'hour','ip_count','device_count','os_count',
#               'ip_group_rank1','ip_group_rank2','ip_group_rank2']

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 32,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.is_attributed)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.is_attributed,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_test,
                verbose_eval = 20,
                early_stopping_rounds=50)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
print('线下auc得分为： {}'.format(roc_auc_score(test_feat.is_attributed,preds)))
print('线下logloss得分为： {}'.format(log_loss(test_feat.is_attributed,preds)))


# train_pool = cb.Pool(train_feat[predictors],train_feat['is_attributed'])
# test_pool = cb.Pool(test_feat[predictors],test_feat['is_attributed'])
#
# cb_model = cb.CatBoostClassifier(iterations=2000, depth=7, learning_rate=0.05, eval_metric='AUC',od_wait=50,)
# print("define model done")
# cb_model.fit(train_pool, use_best_model=True, eval_set=test_pool, verbose=True)
#
# cb_pred = cb_model.predict(test_pool)
# print('线下auc得分为： {}'.format(roc_auc_score(test_feat.is_attributed,cb_pred)))
# print('线下logloss得分为： {}'.format(log_loss(test_feat.is_attributed,cb_pred)))

