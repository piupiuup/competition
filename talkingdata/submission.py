from talkingdata.feat2 import *
import gc
import time
import datetime
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss


inplace = False
data_path = 'C:/Users/cui/Desktop/python/talkingdata/data/'


train = pd.read_hdf(data_path + 'train_sub.hdf').drop('attributed_time',axis=1)
test = pd.read_hdf(data_path + 'test.hdf')
submission = pd.DataFrame({'click_id':test.click_id})
data = train.append(test.drop('click_id',axis=1))
data = pre_treatment(data,'sub_data')
del train,test
gc.collect()

train_feat = pd.DataFrame()
for date in ['2017-11-07','2017-11-08','2017-11-09']:
    train_feat_sub = make_feat(data[data['date']==date].copy(),
                               data[(data['date']!=date) & (data['date']<'2017-11-10')].copy(),
                               data,'{}_sub'.format(date),inplace)
    train_feat_sub = train_feat_sub[train_feat_sub['is_attributed'] == 1].append(
        train_feat_sub[train_feat_sub['is_attributed'] == 0].sample(frac=0.1, random_state=66))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

test_feat = make_feat(data[data['date']=='2017-11-10'].copy(),
                      data[data['date']!='2017-11-10'].copy(),
                      data,'{}_sub'.format('2017-11-10'),inplace)



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
    'learning_rate': 0.01,
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

gbm = lgb.train(params,lgb_train,3000)

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
submission['is_attributed'] = preds
submission.to_csv('C:/Users/cui/Desktop/python/talkingdata/sub{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,float_format='%.4f')


