from talkingdata.feat2 import *
from tool.tool import *
import gc
import time
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss

# channel 全天流量是否异常,工作时流量比上夜间流量
# 相同app在不同平台下载时间的对比
# 用tfidf的思想找出channel对应的主要ip，  找出ip对应的主要平台, 时间对应的平台
# 平台下载平均点击次数/此用户平均点击次数


inplace = False
data_path = 'C:/Users/cui/Desktop/python/talkingdata/data/'


train = pd.read_hdf(data_path + 'train.hdf')
train = pre_treatment(train,'eval_all')


train_feat = pd.DataFrame()
hours = [12, 13, 17, 18, 21, 22]
for date in [7,8]:
    train_feat_sub = make_feat(train[(train['date']==date) & (train['hour'].isin(hours))].copy(),
                               train,'{}_all'.format(date))
    train_feat_sub = train_feat_sub[train_feat_sub['is_attributed'] == 1].append(
        train_feat_sub[train_feat_sub['is_attributed'] == 0].sample(frac=0.1, random_state=66))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

test_feat = make_feat(train[(train['date']==9) & (train['hour'].isin(hours))].copy(),
                      train,'{}_all'.format(9))

test_feat = test_feat.sample(frac=0.3,random_state=66)
gc.collect()

predictors = [c for c in train_feat.columns if c not in ['attributed_diff_time',
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
    'scale_pos_weight':40,
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

