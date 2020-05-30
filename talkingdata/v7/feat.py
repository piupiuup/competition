import gc
import os
import time
import hashlib
import numpy as np
import pandas as pd

columns = ['click_id','date','hour','ip','os','device','app','channel']
cache_path = 'E:/talkingdata/cache/'
data_path = 'E:/talkingdata/data/'
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

def group_rank(data, key,cname,ascending=True):
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
def groupby(stat,key,value,func,data_key,dtype):
    result_path = cache_path + '{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        key = key if type(key)==list else [key]
        stat_temp = stat[list(set(key+[value]))].copy()
        feat = stat_temp.groupby(key,as_index=False)[value].agg({'feat':func})
        feat['feat'] = feat['feat'].astype(dtype)
        stat_temp = stat_temp.merge(feat,on=key,how='left')
        result = stat_temp['feat']
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

# 不同时间内的统计值的比
def groupby_time_rate(stat, key1, key2, cname):
    result_path = cache_path + '{}.hdf'.format(cname)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        stat_temp = stat[key1 + key2].copy()
        stat_temp2 = stat_temp.groupby(key1+key2,as_index=False)[key1[0]].agg({'count':'count'})
        stat_temp2.sort_values(key2,ascending=True,inplace=True)
        shift_value1 = stat_temp2.groupby(key1)[key2[0]].shift(1)
        shift_value2 = stat_temp2.groupby(key1)[key2[0]].shift(-1)
        stat_temp2[cname + '_1'] = (shift_value1) / (stat_temp2['count'] + 0.1)
        stat_temp2[cname + '_2'] = (shift_value2) / (stat_temp2['count'] + 0.1)
        stat_temp = stat_temp.merge(stat_temp2[key1+key2+[cname+'_1',cname+'_2']],on=key1+key2,how='left')
        result = stat_temp[[cname+'_1',cname+'_2']].astype('float32')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前后是否相同
def group_diff_action(stat,key,value,n,cname):
    result_path = cache_path + '{}.hdf'.format(cname)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        stat_temp = stat[key+[value]].copy()
        shift_value = stat_temp.groupby(key)[value].shift(n)
        stat_temp[cname] = (stat_temp[value] == shift_value).astype('int8')
        stat_temp[cname+'_rate'] = groupby(stat_temp,key,cname,np.mean, cname+'_rate','float32')
        result = stat_temp[[cname,cname+'_rate']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

def read(columns):
    data = pd.DataFrame()
    for c in columns:
        result_path = data_path + c + '.hdf'
        data[c] = pd.read_hdf(result_path)
    return data

def tfidf2(stat,key1,key2):
    key = key1 + key2
    cname = '_'.join(key1) + '&' + '_'.join(key2) + '_tfidf2'
    result_path = cache_path + '{}.hdf'.format(cname)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        stat_temp = stat[key].copy()
        df1 = stat_temp.groupby(key,as_index=False)[key[0]].agg({'key_count': 'size'})
        df2 = df1.groupby(key1,as_index=False)['key_count'].agg({'key1_count': 'sum'})
        df3 = df1.groupby(key2, as_index=False)['key_count'].agg({'key2_count': 'sum'})
        df1 = df1.merge(df2,on=key1,how='left').merge(df3,on=key2,how='left')
        df1[cname] = df1['key_count'] / df1['key2_count'] / df1['key1_count']
        result = stat_temp.merge(df1[key+[cname]],on=key,how='left')[cname]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
def get_gini(arr):
    arr = list(arr)
    arr = sorted(arr)
    for i in reversed(range(len(arr))):
        arr[i] = sum(arr[:(i + 1)])
    gini = 1+1/len(arr)-2*sum(arr)/arr[-1]/len(arr)
    return gini
# 基尼指数
def groupby_gini(stat, i,j, cname):
    result_path = cache_path + '{}.hdf'.format(cname)
    if os.path.exists(result_path) & 1:
        result = pd.read_hdf(result_path, 'w')
    else:
        stat1 = stat.groupby(i+j,as_index=False)[i[0]].agg({'count1':'count'})
        stat2 = stat1.groupby(i,as_index=False)['count1'].agg({'gine':get_gini})
        result = stat[i].merge(stat2,on=i,how='left')['gine']
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

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
        stat = read(['date','hour','minute15','date_minute15','date_minute','ip','os','device','channel','app'])
        # 全部数据特征
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for i,j in [[['ip'],'date'],[['ip'],'minute15'],[['channel'],'date'],
                    [['channel'],'minute15']]:
            cname = '_'.join(i) +'&' + j + '_var'
            data_temp[cname] = groupby(stat, i, j, np.var, cname, dtype='float32')
            gc.collect()
        for key in [['date'],['minute15'],['ip'],['os'],['channel'],['app'],
                    ['date','hour'],['date','ip'],['date','os'],['hour','channel'],
                    ['date','minute15'],['ip', 'app'],['ip', 'device'],
                    ['ip', 'os'],['ip', 'channel'],['os', 'app'],['device', 'app'],['device', 'os'],
                    ['channel','app'],['ip', 'os', 'device'],['device', 'channel'],['date','hour','ip'],
                    ['ip','device','app'],['ip','os','app'],
                    ['date', 'hour', 'channel'],['date', 'hour', 'ip', 'channel'],['date', 'hour', 'ip', 'os'],
                    ['date', 'hour', 'ip', 'device'],['date', 'hour', 'ip', 'app'],
                    ['date', 'hour', 'ip', 'device','channel','app'],['ip', 'os','device'],
                    ['ip','os','channel'],['ip','device','os','channel'],['ip', 'device', 'os', 'app'],
                    ['ip','hour'],['ip','minute15'],['ip','device','os','hour'],['ip','device','os','minute15'],
                    ['ip','device','os','date_minute']
                    ]:
            cname = '_'.join(key)+'_count'
            data_temp[cname] = groupby(stat,key,key[0],len,cname,dtype='int32')
            gc.collect()
        stat['os_app'] = stat['os'].astype('int32')*1000 + stat['app']
        stat['device_app'] = stat['device'].astype('int32') * 1000 + stat['app']
        stat['device_os'] = stat['channel'].astype('int32') * 1000 + stat['os']
        for key,value in [[['ip'],'app'],[['ip'],'channel'],[['ip'],'device'],[['ip'],'os'],
                          [['ip'], 'os_app'],[['ip'],'device_app'],[['ip'],'device_os'],[['app'],'channel'],
                          [['channel'],'app'],[['ip','app'],'os'],[['date','ip'],'hour'],
                          [['date', 'ip'], 'os'],[['date','ip'],'device'],
                          [['date', 'hour', 'ip'], 'os'],[['date','hour', 'ip'], 'channel'],
                          [['date', 'hour', 'ip'], 'device'],
                          [['date', 'hour', 'ip'], 'app'],[['date', 'hour', 'device'], 'app'],
                          [['date', 'hour', 'ip','os'], 'app'],[['date', 'hour', 'ip','channel'], 'app'],
                          [['ip','os','device'],'minute15'],[['ip','device','os'],'channel'],
                          [['ip', 'device', 'os'], 'app'], [['ip', 'device', 'os', 'app'], 'channel'],
                          [['ip', 'device', 'os', 'date_minute'], 'channel'],
                          [['ip', 'channel', 'date_minute'], 'channel'],
                          [['date_minute15','ip','channel'], 'device_os'],
                          [['date_minute15', 'ip', 'app'], 'device_os']
                          ]:
            cname = '_'.join(key)+'_n'+value
            data_temp[cname] = groupby(stat,key,value,nunique,cname,dtype='int32')
            gc.collect()
        stat.drop(['os_app','device_app','device_os'],axis=1,inplace=True)
        for i,j in [('channel_app_count','channel_count'),('ip_minute15_count','ip_count'),
                    ('ip_app_count','ip_count'),('ip_os_count','ip_count'),
                    ('ip_channel_count', 'ip_count'),('ip_os_count', 'ip_count'),
                    ('ip_app_count', 'ip_count'),('ip_app_count', 'app_count'),
                    ('os_app_count', 'os_count'), ('os_app_count', 'app_count'),
                    ('os_app_count/app_count_rate', 'os_count'),
                    ('ip_count', 'ip_napp'),('channel_app_count','channel_count'),
                    ('channel_app_count', 'app_count'),
                    ('date_ip_count','ip_count'),('date_hour_ip_count','date_ip_count'),
                    ('date_hour_channel_count','hour_channel_count'),
                    ('date_hour_ip_channel_count', 'date_hour_ip_count'),
                    ('date_hour_ip_device_count', 'date_hour_ip_count'),
                    ('date_hour_ip_app_count', 'date_hour_ip_count'),
                    ('date_hour_ip_channel_count', 'date_hour_ip_count'),
                    ('date_minute15_count','minute15_count'),('date_os_count','os_count'),
                    ('date_hour_ip_count','date_hour_ip_napp'),
                    ('date_hour_ip_count', 'date_hour_ip_nos'),
                    ('date_hour_ip_count','date_hour_ip_nchannel'),
                    ('date_hour_ip_count','date_hour_ip_ndevice'),
                    ]:
            cname = i + '/' + j + '_rate'
            data_temp[cname] = data_temp[i]/(data_temp[j]+0.01)
            gc.collect()
        feat = data_temp.drop(['date_count','date'],axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 基础个数特征
def get_tfidf2_feat(data,data_key):
    result_path = cache_path + 'tfidf2_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        def nunique(x):return len(set(x))
        stat = read(['date','hour','minute15','ip','os','device','channel','app'])
        # 全部数据特征
        #'date','hour','minute15','minute','ip','os','device','channel','app'
        for i,j in [[['ip'],['channel']],[['ip'],['app']],[['ip'],['os']],[['ip'],['device']],
                    [['ip'], ['hour']],[['ip'],['os','device']],[['ip'],['date','hour']],[['ip'],['minute15']],
                    [['channel'],['date','hour']],[['os'],['app']],[['os'],['device']],[['device'],['channel']],
                    [['device'], ['app']],[['channel'],['app']],[['channel'],['hour']],[['app'],['hour']],
                    [['channel'],['minute15']]]:
            cname = '_'.join(i) + '&' + '_'.join(j) + '_tfidf2'
            data_temp[cname] = tfidf2(stat,i,j)
            gc.collect()
        feat = data_temp.drop(columns,axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间rank特征
def get_rank_feat(data,data_key):
    result_path = cache_path + 'rank_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        stat = read(['date','date_hour','date_minute30','date_minute15','minute15','date_minute4','date_minute4',
                     'click_time','ip','os','device','channel','app'])
        data_temp = data.copy()
        for key in [['ip', 'device', 'os'],['ip', 'device', 'os', 'app'],
                    ['ip', 'device', 'os', 'app', 'channel']
                    ]:
            cname = '_'.join(key) + '_rank'
            data_temp['_'.join(key) + '_rank1'] = group_rank(stat, key, cname, ascending=True)
            stat['_'.join(key) + '_rank1'] = group_rank(stat, key, cname, ascending=True)
            gc.collect()
        for key in [['ip', 'device', 'os'],['ip', 'device', 'os', 'app'],
                    ['ip', 'device', 'os', 'app', 'channel'],['date','ip', 'device', 'os', 'channel'],
                    ['date','ip', 'device', 'os', 'app'],
                    ]:
            cname = '_'.join(key) + '_rank'
            data_temp['_'.join(key) + '_rank1'] = group_rank(stat, key, cname, ascending=True)
            data_temp['_'.join(key) + '_rank-1'] = group_rank(stat, key,cname, ascending=False)
            gc.collect()
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
        stat = read(['click_id','date','hour','minute15','ip','os','channel','app','device','is_attributed','attributed_diff_time'])
        date = data.date.min()
        stat1 = stat[(stat['date'] == date)].copy()
        stat2 = stat[(~stat['is_attributed'].isnull()) & (stat['date']!=date)].copy()
        del stat
        stat1.drop(['click_id','date'],axis=1,inplace=True)
        stat2.drop(['date','click_id'],axis=1,inplace=True)
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
        stat = read(['ip','os','device','channel','app','date_hour_2','hour','click_time'])
        data_temp = data.copy()
        # 'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['ip','app'],['ip','channel'],['ip','device','app'],['ip','device','os'],
                    ['ip','device','os','channel'],['ip','device','os','app'],
                    ['ip','device','os','channel','app'],
                    ['device', 'channel'],['app', 'device', 'channel']]:
            for i in [-3,-2,-1,1,2]:
                gc.collect()
                cname = '_'.join(key) + '_diff_time{}'.format(i)
                data_temp[cname] = group_diff_time(stat, key, 'click_time', i,cname)
                gc.collect()
        stat['ip_device_os_app_diff_time-1'] = group_diff_time(stat, ['ip','device','os','app'], 'click_time', -1,'ip_device_os_app_diff_time-1')
        stat['ip_device_os_app_diff_time-1'] = (stat['ip_device_os_app_diff_time-1']>-20).astype('int8')
        for key in [['date_hour_2'],['ip'],['channel'],['app'], ['date_hour_2', 'ip'], ['ip', 'os'],['ip','channel'],
            ['ip','device','os']]:
            cname = '_'.join(key)+'_diff_time_rate'
            data_temp[cname] = groupby(stat,key,'ip_device_os_app_diff_time-1',np.mean,cname,dtype='float32')
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 前后时间统计比
def get_time_rate_feat(data, data_key):
    result_path = cache_path + 'time_rate_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        stat = read(['ip', 'os', 'device', 'channel', 'app', 'date','date_hour','date_minute15','date_minute', 'click_time'])
        data_temp = data.copy()
        for i, j in [[['ip'], ['date_hour']], [['ip'], ['date_minute15']], [['ip'], ['date_minute']],
                     [['ip', 'os', 'device'], ['date_minute']], [['device'], ['date_minute']],
                     [['os'], ['date_minute']],
                     [['app'], ['date_minute']], [['channel'], ['date_minute']], [['device'], ['date_minute']],
                     [['ip', 'app'], ['date_minute']], [['ip', 'os'], ['date_minute']],
                     [['ip', 'device'], ['date_minute']],
                     [['ip', 'os', 'app'], ['date_minute']], [['ip', 'device', 'app'], ['date_minute']],
                     [['channel', 'app'], ['date_minute']]
                     ]:
            cname = '_'.join(i) + '_' + '_'.join(j) + '_time_rate'
            data_temp[[cname+'_1',cname+'_2']] = groupby_time_rate(stat, i, j, cname)
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 前后操作是否相同
def get_diff_action_feat(data, data_key):
    result_path = cache_path + 'diff_action_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        stat = read(['ip', 'os', 'device', 'channel', 'app', 'date_hour_2', 'hour'])
        data_temp = data.copy()
        # 'date','hour','minute15','minute','ip','os','device','channel','app'
        for key,action in [[['ip', 'os','device'], 'app'],[['date_hour_2','ip', 'os','device'], 'app'],[['ip', 'os','device'], 'channel'],
                           [['ip', 'os', 'device','app'], 'channel'],[['ip', 'os','device','channel'], 'app'],
                            [['channel'],'ip']
                           ]:
            for i in [1,-1]:
                cname = '_'.join(key) + '_diff_action_{}_{}'.format(action,i)
                data_temp[[cname,cname+'_rate']] = group_diff_action(stat, key, action, i, cname)
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 基尼系数特征
def get_gini_feat(data, data_key):
    result_path = cache_path + 'gini_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        stat = read(['ip', 'os', 'device', 'channel', 'app', 'hour2', 'click_time'])
        data_temp = data.copy()
        for i,j in [[['ip'],['os']],[['ip'],['device']],[['ip'],['app']],[['ip'],['channel']],[['ip'],['hour2']],
                    [['ip','os'],['app']],[['ip','os'],['channel']]]:
            cname = '_'.join(i) + '_'.join(j) + '_gini'
            data_temp[cname] = groupby_gini(stat, i,j, cname)
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 植物的特征
def get_plantsgo_feat(data, data_key):
    result_path = cache_path + 'plantsgo_feat_{}.hdf'.format(data_key)
    import os
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        stat = read(['date', 'hour2', 'minute15', 'ip', 'os', 'device', 'channel', 'app'])
        # hours = stat['hour2'].value_counts().index[:10]
        # for hour in hours:
        #     stat['value'] = (stat['hour2']==hour).astype('int8')
        #     cname = 'ip_hour' + '_plantsgo' + '_' + str(hour)
        #     data_temp[cname] = groupby(stat, 'ip', 'value', np.mean, cname, dtype='float32')
        #     gc.collect()
        oss = stat['os'].value_counts().index[:10]
        for os in oss:
            stat['value'] = (stat['os'] == os).astype('int8')
            cname = 'ip_os' + '_plantsgo' + '_' + str(os)
            data_temp[cname] = groupby(stat, 'ip', 'value', np.mean, cname, dtype='float32')
            gc.collect()
        devices = stat['device'].value_counts().index[:10]
        for device in devices:
            stat['value'] = (stat['device'] == device).astype('int8')
            cname = 'ip_device' + '_plantsgo' + '_' + str(os)
            data_temp[cname] = groupby(stat, 'ip', 'value', np.mean, cname, dtype='float32')
            gc.collect()
        apps = stat['app'].value_counts().index[:10]
        for app in apps:
            stat['value'] = (stat['app'] == app).astype('int8')
            cname = 'ip_app' + '_plantsgo' + '_' + str(os)
            data_temp[cname] = groupby(stat, 'ip', 'value', np.mean, cname, dtype='float32')
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 通过pred提取信息
def get_pred_feat(data,data_key,flag = 'eval'):
    result_path = cache_path + 'pred_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        stat = read(['date','hour_2','minute15','ip', 'os', 'device', 'channel', 'app', '{}_pred'.format(flag)])
        data_temp = data.copy()
        # 'date','hour','minute15','minute','ip','os','device','channel','app'
        for key in [['hour_2'],['minute15'],['ip'],['os'],['device'],['channel'],['app'],
                    ['ip','os'],['ip','device'],['ip','channel'],['os','device'],['os','app'],['os','channel'],
                    ['device','channel'],['device','app'],['channel','app'],['hour_2','ip'],['hour_2','channel'],
                    ['hour_2','app'],['hour_2','ip','channel'],['hour_2','ip','os'],['hour_2','os','channel']]:
            cname = '_'.join(key) + '_{}_pred_rate'.format(flag)
            data_temp[cname] = groupby(stat, key+['date'], '{}_pred'.format(flag), np.mean, cname, dtype='float32')
            gc.collect()
        feat = data_temp.drop(columns, axis=1)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 二次处理特征
def second_feat(result):
    result['a'] = result['ip_device_os_app_diff_time-1'] - result['ip_device_os_app_diff_time-2']
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
        result.append(get_diff_action_feat(data, data_key))  # 前后操作是否相同
        # result.append(get_time_rate_feat(data, data_key))  # 前后时间统计比
        result.append(get_plantsgo_feat(data, data_key))
        result.append(get_base_feat(data, data_key))      # 基础个数特征
        result.append(get_tfidf2_feat(data, data_key))      # tiidf2
        result.append(get_gini_feat(data, data_key))        # gini系数
        # result.append(get_rank_feat(data, data_key))       # rank特征
        result.append(get_label_encode_feat(data, data_key))# 转化率特征
        result.append(get_diff_time_feat(data, data_key))  # 前后时间差
        # result.append(get_yestoday_feat(data, data_key)) # 昨天是否下载过
        # result.append(get_gini_feat(data, data_key))        # 基尼系数特征
        # result.append(get_pred_feat(data,data_key,'eval'))         # 通过pred提取信息

        result = concat(result)
        result = second_feat(result)
        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result







