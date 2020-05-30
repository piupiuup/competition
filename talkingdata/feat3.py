import gc
import os
import time
import hashlib
import numpy as np
import pandas as pd

columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed',
       'date', 'hour', 'minute', 'minute10']
cache_path = 'C:/Users/cui/Desktop/python/talkingdata/cache/'
inplace = False

############################### 工具函数 ###########################
# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

def group_rank(data,stat, key,data_key,ascending=True):
    cname = '_'.join(key) + '_rank1_' + str(data_key)
    result_path = cache_path + '{}_{}.hdf'.format(cname, int(ascending))
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        stat_temp = stat[key + ['click_id']].copy()
        stat_temp['1'] = 1
        if ascending:
            stat_temp['rank'] = stat_temp.groupby(key)['1'].cumsum().astype('int16')
        else:
            stat_temp['rank'] = stat_temp[::-1].groupby(key)['1'].cumsum().astype('int16')
        index = data.index
        data = data[['click_id']].merge(stat_temp[['click_id','rank']],how='left')
        result = data['rank']
        result.index = index
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# groupby 直接拼接
def groupby(data,stat,key,value,func,data_key):
    result_path = cache_path + '{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        key = key if type(key)==list else [key]
        data_temp = data[key].copy()
        feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
        data_temp = data_temp.merge(feat,on=key,how='left')
        result = data_temp['feat']
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前后时间差的函数：
def group_diff_time(data,stat,key,value,n):
    cname = '_'.join(key) + '_diff_time{}'.format(n)
    stat_temp = stat[key+[value,'click_id']].copy()
    shift_value = stat_temp.groupby(key)[value].shift(n)
    stat_temp[cname] = stat_temp[value] - shift_value
    data = data.merge(stat_temp[['click_id',cname]],on='click_id',how='left')
    return data

############################### 预处理函数 ###########################
def pre_treatment(data,data_key):
    result_path = cache_path + 'data_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_hdf(result_path, 'w')
    else:
        if 'click_id' not in data.columns:
            data['click_id'] = list(range(len(data)))
        data['date'] = data['click_time'].str[8:10].astype('int8')
        data['hour'] = data['click_time'].str[11:13].astype('int8')
        data['minute'] = (data['hour'] * 60 + data['click_time'].str[14:16].astype('int')).astype('int16')
        data['minute10'] = (data['minute'] // 10).astype('int8')
        data['hour'] = data['hour'].astype('int8')
        data['ip'] = data['ip'].astype('int32')
        data['app'] = data['app'].astype('int16')
        data['device'] = data['device'].astype('int16')
        data['os'] = data['os'].astype('int16')
        data['channel'] = data['channel'].astype('int16')
        data['is_attributed'] = data['is_attributed'].astype('int8')
        # data['click_id'] = data['click_id'].astype('int32')
        data.reset_index(drop=True,inplace=True)
        data.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return data


############################### 特征函数 ###########################
# 基础个数特征
def get_base_feat(data,stat,data_key):
    result_path = cache_path + 'base_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        stat['date_hour'] = stat['click_time'].str[:13]
        stat['ten_minute'] = stat['click_time'].str[:15]
        def nunique(x):return len(set(x))
        # 全部数据特征
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['ip'],['ip','date','hour'],['ip','date'],['ip','date','minute10'],['ip','app'],
                    ['ip','os'],['ip','os','date','hour'],['ip','os','date'],['ip','os','date','minute10'],
                    ['ip', 'app','device'], ['ip', 'app','device', 'date', 'hour'], ['ip', 'app','device', 'date'],
                    ['ip', 'app','device', 'date', 'minute10'],
                    ]:
            cname = '_'.join(key)+'_count'
            data_temp[cname] = groupby(data_temp,stat,key,'ip',len,cname+data_key)
        for key,value in [[['ip'],'date_hour'],[['ip'],'ten_minute'],[['ip'],'device'],[['ip'],'app'],
                          ]:
            cname = '_'.join(key)+'_n'+value
            data_temp[cname] = groupby(data_temp,stat,key,value,nunique,cname+str(data_key))
        for i,j in [('ip_date_hour_count','ip_count'),('ip_date_count','ip_count'),
                    ('ip_date_minute10_count', 'ip_count'),('ip_app_count','ip_count'),
                    ('ip_os_date_hour_count','ip_os_count'),('ip_os_date_count','ip_os_count'),('ip_os_date_minute10_count','ip_os_count'),
                    ('ip_app_device_date_hour_count', 'ip_app_device_count'), ('ip_app_device_date_count', 'ip_app_device_count'),
                    ('ip_app_device_date_minute10_count', 'ip_app_device_count'),]:
            cname = i + '/' + j + '_rate'
            data_temp[cname] = data_temp[i]/(data_temp[j]+0.01)
        feat = data_temp
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间rank特征
def get_rank_feat(data,stat,data_key):
    result_path = cache_path + 'rank_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:

        data_temp = data.copy()
        for key in [['ip', 'device', 'os'],['ip', 'device', 'os', 'date'],
                    ['ip', 'device', 'app', 'date', 'hour']]:
            data_temp['_'.join(key) + '_rank1'] = group_rank(data, stat, key, data_key,ascending=True)
            data_temp['_'.join(key) + '_rank-1'] = group_rank(data, stat, key,data_key, ascending=False)
        feat = data_temp.drop(columns,axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 转化率特征
def get_label_encode_feat(data, stat, data_key):
    result_path = cache_path + 'label_encode_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        stat = stat[(~stat['is_attributed'].isnull()) & (stat['date']!=data.date.min())].copy()
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['hour'],['minute15'],['ip'],['os'],['device'],['channel'],['app'],
                    ['ip','os'],['ip','device'],['ip','channel'],['os','device'],['os','app'],['os','channel'],
                    ['device','channel'],['device','app'],['channel','app']]:
            cname = '_'.join(key)+'_rate'
            data_temp[cname] = groupby(data_temp,stat,key,'is_attributed',np.mean,cname+data_key)
        for key in [['ip'],['os'],['device'],['channel'],['app'],['ip','os','device']]:
            cname = '_'.join(key)+'_attributed_mean'
            data_temp[cname] = groupby(data_temp,stat,key,'attributed_diff_time',np.mean,cname+data_key)
        for key in [['ip'],['os'],['device'],['channel'],['app'],['ip','os','device']]:
            cname = '_'.join(key)+'_attributed_std'
            data_temp[cname] = groupby(data_temp,stat,key,'attributed_diff_time',np.std,cname+data_key)
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 前后时间差
def get_diff_time_feat(data, stat, data_key):
    result_path = cache_path + 'diff_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        stat['click_time_temp'] = (stat['date']*86400 + stat['minute']*60 + stat['click_time'].str[-2:].astype('int')).astype('int32')
        stat = stat[['click_id', 'ip', 'os', 'device', 'app', 'click_time_temp']].copy()
        # 'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['ip','device'], ['ip','os'],['ip','device','os'],['ip','app'],['ip','app','os'],['ip']]:
            gc.collect()
            data_temp = group_diff_time(data_temp, stat, key, 'click_time_temp', -2)
            gc.collect()
            data_temp = group_diff_time(data_temp, stat, key, 'click_time_temp', -1)
            gc.collect()
            data_temp = group_diff_time(data_temp, stat, key, 'click_time_temp', 1)
            gc.collect()
            data_temp = group_diff_time(data_temp, stat, key, 'click_time_temp', 2)
            gc.collect()

        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 二次处理特征
def second_feat(result):
    return result


def make_feat(data,all_date,data_key):
    t0 = time.time()
    # data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    # print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        print('开始构造特征...')
        #ip  app  device  os  channel
        data.reset_index(drop=True,inplace=True)

        result = []
        result.append(get_base_feat(data, all_date, data_key))      # 基础个数特征
        result.append(get_rank_feat(data, all_date, data_key))      # rank特征
        # result.append(get_label_encode_feat(data, all_date, data_key))# 转化率特征
        result.append(get_diff_time_feat(data, all_date, data_key))  # 前后时间差

        result = concat(result)
        result = second_feat(result)
        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result





# from tool.tool import *
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
train = pre_treatment(train,'eval_all4')


train_feat = pd.DataFrame()
hours = [12, 13, 17, 18, 21, 22]
for date in [7,8]:
    train_feat_sub = make_feat(train[(train['date']==date) & (train['hour'].isin(hours))].copy(),
                               train,'{}_all4'.format(date))
    train_feat_sub = train_feat_sub[train_feat_sub['is_attributed'] == 1].append(
        train_feat_sub[train_feat_sub['is_attributed'] == 0].sample(frac=0.1, random_state=66))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

del train_feat
gc.collect()
gc.collect()
gc.collect()
gc.collect()

test_feat = make_feat(train[(train['date']==9) & (train['hour'].isin(hours))].copy(),
                      train,'{}_all4'.format(9))

test_feat = test_feat.sample(frac=0.3,random_state=66)
gc.collect()

train_feat = pd.DataFrame()
hours = [12, 13, 17, 18, 21, 22]
for date in [7,8]:
    train_feat_sub = make_feat(train[(train['date']==date) & (train['hour'].isin(hours))].copy(),
                               train,'{}_all4'.format(date))
    train_feat_sub = train_feat_sub[train_feat_sub['is_attributed'] == 1].append(
        train_feat_sub[train_feat_sub['is_attributed'] == 0].sample(frac=0.1, random_state=66))
    train_feat = train_feat.append(train_feat_sub)
    del train_feat_sub
    gc.collect()

predictors = [c for c in train_feat.columns if c not in ['attributed_diff_time','ten_minute','date_hour',
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



