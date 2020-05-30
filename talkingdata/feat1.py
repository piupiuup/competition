import os
import time
import hashlib
import numpy as np
import pandas as pd

columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed',
       'date', 'hour', 'minute', 'minute15', 'minute3']
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

def group_rank(data, key,ascending=True):
    data_temp = data[key].copy()
    data_temp['1'] = 1
    if ascending:
        data_temp['rank'] = data_temp.groupby(key)['1'].cumsum()
    else:
        data_temp['rank'] = data_temp[::-1].groupby(key)['1'].cumsum()
    return data_temp['rank']

# groupby 直接拼接
def groupby(data,stat,key,value,func):
    key = key if type(key)==list else [key]
    data_temp = data[key].copy()
    feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
    data_temp = data_temp.merge(feat,on=key,how='left')
    return data_temp['feat']

# 前后时间差的函数：
def group_diff_time(data,key,value,n):
    cname = '_'.join(key) + '_diff_time{}'.format(n)
    data_temp = data[key+[value]].copy()
    shift_value = data_temp.groupby(key)[value].shift(n)
    data[cname] = data_temp[value] - shift_value
    return data

############################### 预处理函数 ###########################
def pre_treatment(data,data_key):
    result_path = cache_path + 'data_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_hdf(result_path, 'w')
    else:
        def f2(x):
            if x>1260: return x-1260
            elif x>1020: return x-1020
            elif x>720: return x-720
            return x
        data['date'] = data['click_time'].str[8:10].astype('int8')
        data['hour'] = data['click_time'].str[11:13].astype('int')
        data['minute'] = (data['hour'] * 60 + data['click_time'].str[14:16].astype('int')).astype('int16')
        data['minute15'] = (data['minute'] // 15).astype('int8')
        data['minute3'] = (data['minute'].apply(f2)).astype('int16')
        data['hour'] = data['hour'].astype('int8')
        data['ip'] = data['ip'].astype('int32')
        data['app'] = data['app'].astype('int16')
        data['device'] = data['device'].astype('int16')
        data['os'] = data['os'].astype('int16')
        data['channel'] = data['channel'].astype('int16')
        data['attributed_diff_time'] = data['attributed_time']
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
        def nunique(x):return len(set(x))
        # 当天特征
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['date'],['hour'],['minute15'],['ip'],['os'],['device'],['channel'],['app'],

                    ['date','hour'],['date','ip'],['date','os'],['date','channel'],['hour','channel'],
                    ['date','minute15'],['ip', 'app'],['ip', 'device'],
                    ['ip', 'os'],['ip', 'channel'],['os', 'app'],['device', 'app'],['device', 'os'],
                    ['channel','app'],['ip', 'os', 'device'],['device', 'channel'],['date','hour','ip'],
                    ['date', 'hour', 'channel'],['date', 'hour', 'ip', 'channel'],['ip', 'os','device'],
                    ['ip','os','channel'],['ip','device','os','channel'],['ip', 'device', 'os', 'app'],
                    ['ip','hour'],['ip','minute15'],['ip','device','os','hour'],['ip','device','os','minute15'],
                    ]:
            cname = '_'.join(key)+'_count'
            data_temp[cname] = groupby(data_temp,stat,key,'ip',len)
        for key,value in [[['ip'],'app'],[['ip'],'channel'],[['ip'],'device'],[['ip'],'os'],
                          [['channel'],'app'],[['ip','os','device'],'date'],[['ip','device','os'],'channel'],
                          [['ip', 'device', 'os'], 'app'],[['ip', 'device', 'os','channel'], 'app'],
                          [['ip', 'os', 'device','app'], 'date'],[['ip','os','device','channel'],'date'],
                          ]:
            cname = '_'.join(key)+'_n'+value
            data_temp[cname] = groupby(data_temp,stat,key,value,nunique)
        data_temp['date_count/count_rate'] = data_temp['date_count']/stat.shape[0]
        data_temp['hour_sum'] = data_temp['date_count'] / stat.shape[0]
        for i,j in [('channel_app_count','channel_count'),('ip_minute15_count','ip_count'),
                    ('date_hour_channel_count','hour_channel_count'),('date_hour_count','hour_count'),
                    ('date_minute15_count','minute15_count'),('date_os_count','os_count'),
                    ('date_channel_count','channel_count'),('date_channel_count/channel_count_rate','date_count/count_rate')]:
            cname = i + '/' + j + '_rate'
            data_temp[cname] = data_temp[i]/(data_temp[j]+0.01)
        feat = data_temp.drop(['date_count'],axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间rank特征
def get_rank_feat(data,data_key):
    result_path = cache_path + 'rank_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:

        data_temp = data.copy()
        data_temp['hour2'] = ((data_temp['minute'] + 30) // 60).astype('int16')
        data_temp['minute30_1'] = ((data_temp['minute']) // 30).astype('int16')
        data_temp['minute30_2'] = ((data_temp['minute'] + 15) // 30).astype('int16')
        data_temp['minute15_2'] = ((data_temp['minute'] + 5) // 15).astype('int16')
        data_temp['minute15_3'] = ((data_temp['minute'] + 10) // 15).astype('int16')
        data_temp['minute10_1'] = (data_temp['minute'] // 10).astype('int16')
        data_temp['minute10_2'] = ((data_temp['minute']+5) // 10).astype('int16')
        data_temp['minute5'] = (data_temp['minute'] // 5).astype('int16')
        data_temp['minute5_2'] = ((data_temp['minute'] + 3) // 5).astype('int16')
        data_temp['minute4'] = (data_temp['minute'] // 4).astype('int16')
        data_temp['minute3'] = (data_temp['minute'] // 3).astype('int16')
        data_temp['minute2'] = (data_temp['minute'] // 2).astype('int16')
        for key in [['ip', 'device', 'os'],['ip', 'device', 'os', 'app'],
                    ['ip', 'device', 'os', 'app', 'channel'],['ip', 'device', 'os', 'channel'],
                    ['ip', 'device', 'os', 'hour'],['ip', 'device', 'os', 'hour2'],
                    ['ip', 'device', 'os', 'hour','app'], ['ip', 'device', 'os', 'hour2','channel'],
                    ['ip', 'device', 'os', 'minute30_1'],['ip', 'device', 'os', 'minute30_2'],['ip', 'device', 'os', 'minute15'],
                    ['ip', 'device', 'os', 'channel', 'hour'],['ip', 'device', 'os', 'channel','minute15'],
                    ['ip', 'device', 'os','minute15_2'],['ip', 'device', 'os','minute15_3'],
                    ['ip', 'device', 'os', 'minute10_1'], ['ip', 'device', 'os', 'minute10_2'],
                    ['ip', 'device', 'os', 'channel', 'minute5'],['ip', 'device', 'os', 'app','minute5'],
                    ['ip', 'device', 'os', 'channel', 'minute5_2'], ['ip', 'device', 'os', 'app', 'minute5_2'],
                    ['ip', 'device', 'os', 'minute5'], ['ip', 'device', 'os', 'minute4'],
                    ['ip', 'device', 'os', 'channel', 'minute3'], ['ip', 'device', 'os', 'app', 'minute3'],
                    ['ip', 'device', 'os', 'minute3'], ['ip', 'device', 'os', 'minute2'],
                    ['ip', 'device', 'os', 'minute'],['ip', 'device', 'os', 'app', 'channel','click_time']]:
            data_temp['_'.join(key) + '_rank1'] = group_rank(data_temp, key, ascending=True)
            data_temp['_'.join(key) + '_rank-1'] = group_rank(data_temp, key, ascending=False)
            data_temp['_'.join(key) + '_rank_rate'] = data_temp['_'.join(key) + '_rank1'] / (data_temp['_'.join(key) + '_rank1'] + data_temp['_'.join(key) + '_rank-1'])
        feat = data_temp.drop(columns+['minute15_2','minute15_3','minute10_2','minute10_1','hour2','minute30_1','minute30_2',
                                       'minute5','minute4','minute3','minute2'],axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 转化率特征
def get_label_encode_feat(data, stat, data_key):
    result_path = cache_path + 'label_encode_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['hour'],['minute15'],['ip'],['os'],['device'],['channel'],['app'],
                    ['ip','os'],['ip','device'],['ip','channel'],['os','device'],['os','app'],['os','channel'],
                    ['device','channel'],['device','app'],['channel','app']]:
            cname = '_'.join(key)+'_rate'
            data_temp[cname] = groupby(data_temp,stat,key,'is_attributed',np.mean)


        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 前后时间差
def get_diff_time_feat(data, data_key):
    result_path = cache_path + 'diff_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp['click_time_temp'] = (data_temp['date']*86400 + data_temp['minute']*60 + data['click_time'].str[-2:].astype('int')).astype('int32')
        # 'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['ip','device','os'], ['ip','device','os','channel'],['ip','device','os','app'],['ip','device','os','channel','app']]:
            data_temp = group_diff_time(data_temp, key, 'click_time_temp', -2)
            data_temp = group_diff_time(data_temp, key, 'click_time_temp', -1)
            data_temp = group_diff_time(data_temp, key, 'click_time_temp', 1)
            data_temp = group_diff_time(data_temp, key, 'click_time_temp', 2)

        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 二次处理特征
def second_feat(result):
    return result


def make_feat(data,stat,all_data,data_key):
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
        result.append(get_base_feat(data, all_data, data_key))      # 基础个数特征
        result.append(get_rank_feat(data, data_key))                # rank特征
        result.append(get_label_encode_feat(data, stat, data_key))  # 转化率特征
        result.append(get_diff_time_feat(data, data_key))           # 前后时间差

        result = concat(result)
        result = second_feat(result)
        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result







