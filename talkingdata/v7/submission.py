from talkingdata.v7.feat import *
import gc
import time
import datetime
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss


inplace = False

data = read(['click_id','date','hour','ip','os','device','app','channel','click_time','is_attributed'])

train_feat = pd.DataFrame()
hours = [12, 13, 17, 18, 21, 22]
for date in [0,1,2]:
    data_temp = data[(data['date'] == date) & (~data['is_attributed'].isnull())].copy()
    data_temp = data_temp[data_temp['is_attributed']==1].append(
        data_temp[data_temp['is_attributed'] == 0].sample(frac=0.05, random_state=66))
    train_feat_sub = make_feat(data_temp, '{}_7_all'.format(date))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()


predictors = [c for c in train_feat.columns if c not in ['click_id', 'click_time','ip','is_attributed','date','ip_os_rate']]

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 32,
    'learning_rate': 0.01,
    # 'subsample': 0.7,
    # 'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'scale_pos_weight':40,
    'verbose': 0,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.is_attributed)
gbm = lgb.train(params,lgb_train,2700)
del train_feat,lgb_train
gc.collect()

test_data = data[data['click_id']>-1].copy()
test_feat = make_feat(test_data, '3_7_6')


print('开始预测...')
preds1 = gbm.predict(test_feat[predictors][:4000000])
preds2 = gbm.predict(test_feat[predictors][4000000:8000000])
preds3 = gbm.predict(test_feat[predictors][8000000:12000000])
preds4 = gbm.predict(test_feat[predictors][12000000:16000000])
preds5 = gbm.predict(test_feat[predictors][16000000:])
preds = np.concatenate([preds1,preds2,preds3,preds4,preds5],axis=0)
test_feat['is_attributed'] = preds

submission = test_feat[['click_id','is_attributed']]
submission.to_csv('C:/Users/cui/Desktop/python/talkingdata/submission/sub{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,float_format='%.6f')














