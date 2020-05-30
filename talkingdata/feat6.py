import gc
import os
import time
import hashlib
import numpy as np
import pandas as pd

columns = ['click_id','date','hour','ip','os','device','app','channel']
cache_path = 'D:/talkingdata/cache/'
data_path = 'D:/talkingdata/data/'
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

def group_rank(data, key,data_key,ascending=True):
    cname = '_'.join(key) + '_rank1_' + str(data_key)
    result_path = cache_path + '{}_{}.hdf'.format(cname,int(ascending))
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data[key].copy()
        data_temp['1'] = 1
        if ascending:
            data_temp['rank'] = data_temp.groupby(key)['1'].cumsum()
        else:
            data_temp['rank'] = data_temp[::-1].groupby(key)['1'].cumsum()
        result = data_temp['rank'].astype('int32')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# groupby 直接拼接
def groupby(data,stat,key,value,func,data_key,dtype):
    result_path = cache_path + '{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        key = key if type(key)==list else [key]
        feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
        feat['feat'] = feat['feat'].astype(dtype)
        stat = stat.merge(feat,on=key,how='left')
        result = stat['feat']
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
def groupby2(data,stat,key,value,func,data_key,dtype):
    result_path = cache_path + '{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data[key].copy()
        key = key if type(key)==list else [key]
        feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
        feat['feat'] = feat['feat'].astype(dtype)
        index = data_temp.index.copy()
        data_temp = data_temp.merge(feat,on=key,how='left')
        data_temp.index = index
        result = data_temp['feat']
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前后时间差的函数：
def group_diff_time(stat,key,value,n,cname):
    result_path = cache_path + '{}.hdf'.format(cname)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        stat_temp = stat[key+[value]].copy()
        shift_value = stat_temp.groupby(key)[value].shift(n)
        stat_temp[cname] = stat_temp[value] - shift_value
        result = stat_temp[cname]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

def read(columns):
    data = pd.DataFrame()
    for c in columns:
        result_path = data_path + c + '.hdf'
        data[c] = pd.read_hdf(result_path)
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
        data['hour'] = data['click_time'].str[11:13].astype('int8')
        data['minute'] = (data['hour'] * 60 + data['click_time'].str[14:16].astype('int')).astype('int16')
        data['minute15'] = (data['minute'] // 15).astype('int8')
        data['minute3'] = (data['minute'].apply(f2)).astype('int16')
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
def get_base_feat(data,data_key):
    result_path = cache_path + 'base_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        def nunique(x):return len(set(x))
        stat = read(['date','hour','minute15','ip','os','device','channel','app'])
        # 全部数据特征
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['date'],['hour'],['minute15'],['ip'],['os'],['channel'],['app'],
                    ['date','hour'],['date','ip'],['date','os'],['date','channel'],['hour','channel'],
                    ['date','minute15'],['ip', 'app'],['ip', 'device'],
                    ['ip', 'os'],['ip', 'channel'],['os', 'app'],['device', 'app'],['device', 'os'],
                    ['channel','app'],['ip', 'os', 'device'],['device', 'channel'],['date','hour','ip'],
                    ['date', 'hour', 'channel'],['date', 'hour', 'ip', 'channel'],['ip', 'os','device'],
                    ['ip','os','channel'],['ip','device','os','channel'],['ip', 'device', 'os', 'app'],
                    ['ip','hour'],['ip','minute15'],['ip','device','os','hour'],['ip','device','os','minute15'],
                    ]:
            cname = '_'.join(key)+'_count'
            data_temp[cname] = groupby(data_temp,stat,key,key[0],len,cname,dtype='int32')
            gc.collect()
        for key,value in [[['ip'],'app'],[['ip'],'channel'],[['ip'],'device'],[['ip'],'os'],
                          [['channel'],'app'],[['ip','os','device'],'date'],[['ip','device','os'],'channel'],
                          [['ip', 'device', 'os'], 'app']
                          ]:
            cname = '_'.join(key)+'_n'+value
            data_temp[cname] = groupby(data_temp,stat,key,value,nunique,cname,dtype='int32')
            gc.collect()
        data_temp['date_count/count_rate'] = data_temp['date_count'] / stat.shape[0]
        for i,j in [('channel_app_count','channel_count'),('ip_minute15_count','ip_count'),
                    ('date_hour_channel_count','hour_channel_count'),('date_hour_count','hour_count'),
                    ('date_minute15_count','minute15_count'),('date_os_count','os_count'),
                    ('date_channel_count','channel_count'),('date_channel_count/channel_count_rate','date_count/count_rate')]:
            cname = i + '/' + j + '_rate'
            data_temp[cname] = data_temp[i]/(data_temp[j]+0.01)
        feat = data_temp.drop(['date_count','date_count/count_rate'],axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间rank特征
def get_rank_feat(data,data_key):
    result_path = cache_path + 'rank_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:

        data_temp = data.copy()
        for key in [['ip', 'device', 'os'],['date','ip', 'device', 'os'],['date','ip', 'device', 'os', 'app'],
                    ['date','ip', 'device', 'os', 'app', 'channel'],['date','ip', 'device', 'os', 'channel'],
                    ['ip', 'device', 'os', 'date_hour'],['ip', 'device', 'os', 'date_hour_2'],
                    ['ip', 'device', 'os', 'date_hour','app'], ['ip', 'device', 'os', 'date_hour_2','channel'],
                    ['ip', 'device', 'os', 'date_minute30'],['ip', 'device', 'os', 'date_minute15'],
                    ['ip', 'device', 'os', 'channel', 'date_hour'],['ip', 'device', 'os', 'channel','date_minute15'],
                    ['ip', 'device', 'os', 'date_minute10'], ['ip', 'device', 'os', 'date_minute10_2'],
                    ['ip', 'device', 'os', 'channel', 'date_minute5'],['ip', 'device', 'os', 'app','date_minute5'],
                    ['ip', 'device', 'os', 'date_minute5'], ['ip', 'device', 'os', 'date_minute4'],
                    ['ip', 'device', 'os', 'date_minute2'],
                    ['ip', 'device', 'os', 'date_minute'],['ip', 'device', 'os', 'app', 'channel','click_time']]:
            stat = read(key)
            cname = '_'.join(key) + '_rank'
            data_temp['_'.join(key) + '_rank1'] = group_rank(stat, key, cname, ascending=True)
            data_temp['_'.join(key) + '_rank-1'] = group_rank(stat, key,cname, ascending=False)
            del stat
            gc.collect()
            data_temp['_'.join(key) + '_rank_rate'] = data_temp['_'.join(key) + '_rank1'] / (data_temp['_'.join(key) + '_rank1'] + data_temp['_'.join(key) + '_rank-1'])
        feat = data_temp.drop(columns,axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 转化率特征
def get_label_encode_feat(data, data_key):
    result_path = cache_path + 'label_encode_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        stat = read(['click_id','date','hour','minute15','minute15','ip','os','channel','app','device','is_attributed','attributed_diff_time'])
        date = data.date.min()
        stat1 = stat[(stat['date'] == date)].copy()
        stat2 = stat[(stat['click_id']==-1) & (stat['date']!=date)].copy()
        del stat
        stat1.drop(['click_id','date'],axis=1)
        stat2.drop(['date','click_id'],axis=1)
        gc.collect()
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['hour'],['minute15'],['ip'],['os'],['device'],['channel'],['app'],
                    ['ip','os'],['ip','device'],['ip','channel'],['os','device'],['os','app'],['os','channel'],
                    ['device','channel'],['device','app'],['channel','app']]:
            cname = '_'.join(key)+'_rate'
            data_temp[cname] = groupby2(stat1,stat2,key,'is_attributed',np.mean,cname+str(date),dtype='float32')
            gc.collect()
        for key in [['hour'],['minute15'],['ip'],['os'],['channel'],['app'],['ip','os','device']]:
            cname = '_'.join(key)+'_attributed_mean'
            data_temp[cname] = groupby2(stat1,stat2,key,'attributed_diff_time',np.mean,cname+str(date),dtype='float32')
            gc.collect()
        for key in [['hour'],['minute15'],['ip'],['os'],['channel'],['app'],['ip','os','device']]:
            cname = '_'.join(key)+'_attributed_std'
            data_temp[cname] = groupby2(stat1,stat2,key,'attributed_diff_time',np.std,cname+str(date),dtype='float32')
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 前后时间差
def get_diff_time_feat(data, data_key):
    result_path = cache_path + 'diff_time_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        stat = read(['ip','os','device','channel','app','click_time'])
        data_temp = data.copy()
        # 'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['ip','device','os'], ['ip','device','os','channel'],['ip','device','os','app'],
                    ['ip','device','os','channel','app']]:
            for i in [-2,-1,1,2]:
                gc.collect()
                cname = '_'.join(key) + '_diff_time{}'.format(i)
                data_temp[cname] = group_diff_time(stat, key, 'click_time', i,cname)
                gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 二次处理特征
def second_feat(result):
    return result


def make_feat(data,data_key):
    t0 = time.time()
    # data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    # print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        print('开始构造特征...')
        #ip  app  device  os  channel
        # data.reset_index(drop=True,inplace=True)

        result = []
        result.append(get_base_feat(data, data_key))      # 基础个数特征
        # result.append(get_rank_feat(data, data_key))                # rank特征
        result.append(get_label_encode_feat(data, data_key))# 转化率特征
        result.append(get_diff_time_feat(data, data_key))  # 前后时间差

        result = concat(result)
        result = second_feat(result)
        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result







