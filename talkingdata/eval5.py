
from talkingdata.feat5 import *
import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss

# channel 全天流量是否异常,工作时流量比上夜间流量
# 相同app在不同平台下载时间的对比
# 用tfidf的思想找出channel对应的主要ip，  找出ip对应的主要平台
# 平台下载平均点击次数/此用户平均点击次数




data = read(['click_id','date','hour','ip','os','device','app','channel','is_attributed'])

hours = [12, 13, 17, 18, 21, 22]
for date in [0,1,2]:
    temp = make_feat(data[(data['date'] == date) & (data['hour'].isin(hours))].copy(), '{}_5_6'.format(date))
    del temp
    gc.collect()
train_feat = pd.DataFrame()
for date in [0,1]:
    train_feat_sub = make_feat(data[(data['date'] == date) & (data['hour'].isin(hours))].copy(), '{}_5_6'.format(date))
    train_feat_sub = train_feat_sub[train_feat_sub['is_attributed'] == 1].append(
        train_feat_sub[train_feat_sub['is_attributed'] == 0].sample(frac=0.1, random_state=66))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

test_feat = make_feat(data[(data['date'] == 2) & (data['hour'].isin(hours))].copy(), '2_5_6')

test_feat = test_feat.sample(frac=0.3,random_state=66)
gc.collect()

predictors = [c for c in train_feat.columns if c not in ['click_id',
 'click_time','date','ip','is_attributed']]


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
