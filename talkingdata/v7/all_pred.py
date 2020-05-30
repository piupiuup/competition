from talkingdata.v7.feat import *
import gc
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss

# channel 全天流量是否异常,工作时流量比上夜间流量
# 相同app在不同平台下载时间的对比
# 用tfidf的思想找出channel对应的主要ip，  找出ip对应的主要平台
# 平台下载平均点击次数/此用户平均点击次数
# 设计一个可以描述集中程度的指数
# 转化率截断


def lgb_train(train_feat,eval_feat,predictors,label='label'):
    print('开始训练...')
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 8,
        'num_leaves': 32,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'scale_pos_weight': 40,
        'verbose': 0,
        'seed': 66,
    }
    lgb_train = lgb.Dataset(train_feat[predictors], train_feat['is_attributed'])
    lgb_eval = lgb.Dataset(eval_feat[predictors], eval_feat['is_attributed'])

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    verbose_eval=20,
                    early_stopping_rounds=50)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    return gbm,feat_imp

def lgb_pred(gbm,test_feat):
    print('开始预测...')
    preds = []
    for i in tqdm(range(test_feat.shape[0] // 4000000 + 1)):
        if i == test_feat.shape[0] // 4000000:
            preds.append(gbm.predict(test_feat[predictors][i * 4000000:test_feat.shape[0]]))
        else:
            preds.append(gbm.predict(test_feat[predictors][i * 4000000:(i + 1) * 4000000]))
    preds = np.concatenate(preds, axis=0)
    return preds


data = read(['click_id','date','hour','ip','os','device','app','channel','is_attributed'])

hours = [12, 13, 17, 18, 21, 22]
# for date in [0, 1, 2, 3]:
#     data_temp = data[(data['date'] == date) & ((data['hour'] % 2) != 1)].copy()
#     train_feat_sub = make_feat(data_temp, '{}_all0_all'.format(date))
#     del train_feat_sub
#     gc.collect()
# for date in [0,1,2,3]:
#     data_temp = data[(data['date'] == date) & ((data['hour']%2)==1)].copy()
#     train_feat_sub = make_feat(data_temp, '{}_all1_all'.format(date))
#     del train_feat_sub
#     gc.collect()

preds = pd.Series()
test_preds = []
for train_days,test_day in [[[0],1],[[1],0]]:
    print('训练日期为：{}，预测日期为：{}'.format(train_days,test_day))
    train_feat = pd.DataFrame()
    for date in train_days:
        data_temp = data[(data['date'] == date) & (~data['is_attributed'].isnull())].copy()
        data_temp = data_temp[data_temp['is_attributed']==1].append(
            data_temp[data_temp['is_attributed'] == 0].sample(frac=0.1, random_state=66))
        train_feat_sub = make_feat(data_temp, '{}_5_all'.format(date))
        train_feat = train_feat.append(train_feat_sub)
        del train_feat_sub
        gc.collect()

    eval_feat1 = make_feat(data[(data['date'] == test_day) & ((data['hour']%2)!=1)].copy(), '{}_all0_all'.format(test_day))
    eval_feat_sub = eval_feat1.sample(frac=0.2,random_state=66)
    gc.collect()

    predictors = [c for c in train_feat.columns if c not in ['click_id', 'click_time','is_attributed','ip']]

    gbm,feat_imp = lgb_train(train_feat,eval_feat_sub,predictors,'is_attributed')
    del train_feat,eval_feat_sub
    gc.collect()
    eval_feat1['preds'] = lgb_pred(gbm,eval_feat1)
    preds = preds.append(eval_feat1['preds'].copy())
    del eval_feat1
    gc.collect()
    eval_feat2 = make_feat(data[(data['date'] == test_day) & ((data['hour'] % 2) == 1)].copy(),'{}_all1_all'.format(test_day))
    eval_feat2['preds'] = lgb_pred(gbm,eval_feat2)
    preds = preds.append(eval_feat2['preds'].copy())
    del eval_feat2
    gc.collect()

    test_feat1 = make_feat(data[(data['date'] == 2) & ((data['hour'] % 2) != 1)].copy(),'{}_all0_all'.format(2))
    test_feat1['preds'] = lgb_pred(gbm, test_feat1)
    test_pred = test_feat1['preds']
    del test_feat1
    gc.collect()
    test_feat2 = make_feat(data[(data['date'] == 2) & ((data['hour'] % 2) == 1)].copy(), '{}_all1_all'.format(2))
    test_feat2['preds'] = lgb_pred(gbm, test_feat2)
    test_pred = test_pred.append(test_feat2['preds'])
    del test_feat2
    gc.collect()
    test_preds.append(test_pred)
    del test_pred
    gc.collect()
test_preds = sum(test_preds)/len(test_preds)
preds = preds.append(test_preds)
preds.to_hdf(data_path+'eval_pred.hdf', 'w', complib='blosc', complevel=5)



