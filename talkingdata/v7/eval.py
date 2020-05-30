from talkingdata.v7.feat import *
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
# 设计一个可以描述集中程度的指数
# 转化率截断
# 这个时间点个数比上上个时间点个数
# count / nunique

data = read(['click_id','date','hour','ip','os','device','app','channel','is_attributed'])

hours = [12, 13, 17, 18, 21, 22]
# for date in [0,1,2]:
#     data_temp = data[(data['date'] == date) & (~data['is_attributed'].isnull())].copy()
#     data_temp = data_temp[data_temp['is_attributed']==1].append(
#         data_temp[data_temp['is_attributed'] == 0].sample(frac=0.05, random_state=66))
#     train_feat_sub = make_feat(data_temp, '{}_7_all'.format(date))
#     del train_feat_sub
#     gc.collect()

train_feat = pd.DataFrame()
for date in [0,1]:
    data_temp = data[(data['date'] == date) & (~data['is_attributed'].isnull())].copy()
    data_temp = data_temp[data_temp['is_attributed']==1].append(
        data_temp[data_temp['is_attributed'] == 0].sample(frac=0.05, random_state=66))
    train_feat_sub = make_feat(data_temp, '{}_7_all'.format(date))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

data_temp = data[(data['date'] == 2) & (data['hour'].isin(hours) & (~data['is_attributed'].isnull()))].sample(frac=0.3,random_state=66)
test_feat = make_feat(data_temp, '2_7_6')
gc.collect()

predictors = [c for c in train_feat.columns if c not in ['click_id', 'click_time','is_attributed','ip','date','ip_os_rate']]

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 7,
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
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.is_attributed)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_test,
                verbose_eval = 20,
                early_stopping_rounds=50)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
