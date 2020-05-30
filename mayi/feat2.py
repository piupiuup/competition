import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from collections import defaultdict
from joblib import Parallel, delayed


cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'
flag = True

# 线下测评
def acc(data,name='shop_id'):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:
        print('没有发现真实数据，无法测评')
    return sum(data['row_id'].map(true)==data[name])/data.shape[0]

# 线下测评
def get_label(data):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:
        print('没有发现真实数据，无法测评')
    data['label'] = (data['shop_id']==data['row_id'].map(true)).astype(int)
    return data

# 分组标准化
def grp_standard(data,key,names):
    for name in names:
        mean_std = data.groupby(key, as_index=False)[name].agg({'mean': 'mean',
                                                               'std': 'std'})
        data = data.merge(mean_std, on=key, how='left')
        data[name] = ((data[name]-data['mean'])/data['std']).fillna(0)
        data[name] = data[name].replace(-np.inf, 0).fillna(0)
        data.drop(['mean','std'],axis=1,inplace=True)
    return data

# 分组归一化
def grp_normalize(data,key,names,start=0):
    for name in names:
        max_min = data.groupby(key,as_index=False)[name].agg({'max':'max',
                                                'min':'min'})
        data = data.merge(max_min,on=key,how='left')
        data[name] = (data[name]-data['min'])/(data['max']-data['min'])
        data[name] = data[name].replace(-np.inf, start).fillna(start)
        data.drop(['max','min'],axis=1,inplace=True)
    return data

# 分组排序
def grp_rank(data,key,names,ascending=True):
    for name in names:
        data.sort_values([key, name], inplace=True, ascending=ascending)
        data['rank'] = range(data.shape[0])
        min_rank = data.groupby(key, as_index=False)['rank'].agg({'min_rank': 'min'})
        data = pd.merge(data, min_rank, on=key, how='left')
        data['rank'] = data['rank'] - data['min_rank']
        data[names] = data['rank']
    data.drop(['rank'],axis=1,inplace=True)
    return data

# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

def group_rank(data, key, values, ascending=True):
    data.sort_values([key,values],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(key,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=key,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 商店对应的连接wiif
def get_connect_wifi(wifi_infos):
    if wifi_infos != '':
        for wifi_info in wifi_infos.split(';'):
            bssid,signal,flag = wifi_info.split('|')
            if flag == 'true':
                return bssid
    return np.nan

# 商店对应的连接wiif
def get_most_wifi(wifi_infos):
    if wifi_infos != '':
        bssid = sorted([wifi.split('|') for wifi in wifi_infos.split(';')],key=lambda x:float(x[1]),reverse=True)[0][0]
        return bssid
    return np.nan

# 排名对应的权重
def rank_weight(i):
    return np.exp((0 - i) * 0.6)

# wifi连接过的商店的个数
def get_shop_cwifi_count(data,data_key):
    result_path = cache_path + 'shop_cwifi_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp['bssid'] = data_temp['wifi_infos'].apply(get_connect_wifi)
        result = data_temp.groupby(['bssid', 'shop_id'], as_index=False)['shop_id'].agg({'shop_cwifi_count': 'count'})
        result.sort_values(['bssid', 'shop_cwifi_count'], ascending=False, inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取商店-wifi对应的次数
def get_shop_wifi_tfidf(data,data_key):
    result_path = cache_path + 'shop_wifi_tfidf_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, float(signal)])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi = group_rank(shop_wifi, 'row_id', 'signal', ascending=False)
        shop_wifi['rank'] = shop_wifi['rank'].apply(rank_weight)
        shop_wifi.rename(columns={'rank':'weight'},inplace=True)
        shop_wifi_tfidf = shop_wifi.groupby(['shop_id', 'bssid'], as_index=False)['weight'].agg({'shop_wifi_tfidf': 'sum'})
        shop_count = shop_wifi.groupby(['shop_id'], as_index=False)['weight'].agg({'shop_count': 'sum'})
        shop_wifi_tfidf = shop_wifi_tfidf.merge(shop_count, on='shop_id', how='left')
        shop_wifi_tfidf['tfidf'] = shop_wifi_tfidf['shop_wifi_tfidf'] / shop_wifi_tfidf['shop_count']
        wifi_count = shop_wifi_tfidf.groupby(['bssid'], as_index=False)['tfidf'].agg({'wifi_count': 'sum'})
        shop_wifi_tfidf = shop_wifi_tfidf.merge(wifi_count, on='bssid', how='left')
        shop_wifi_tfidf['tfidf'] = shop_wifi_tfidf['tfidf'] / shop_wifi_tfidf['wifi_count']
        result = shop_wifi_tfidf
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 最强wifi对应过的商店
def get_shop_mwifi_count(data,data_key):
    result_path = cache_path + 'shop_mwifi_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp['bssid'] = data_temp['wifi_infos'].apply(get_most_wifi)
        result = data_temp.groupby(['bssid','shop_id'],as_index=False)['shop_id'].agg({'shop_mwifi_count':'count'})
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 清洗无用wifi
def clear(data,candidate,data_key):
    data_path = cache_path + 'data_clear_{}.hdf'.format(data_key)
    candidate_path = cache_path + 'candidate_clear_{}.hdf'.format(data_key)
    if os.path.exists(data_path) & os.path.exists(candidate_path) & flag:
        data = pd.read_hdf(data_path, 'w')
        candidate = pd.read_hdf(candidate_path, 'w')
    else:
        train_shop_wifi_count_dict = defaultdict(lambda: defaultdict(lambda: 0))
        for mall_id, wifi_infos in zip(data['mall_id'].values, data['wifi_infos'].values):
            for wifi_info in wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                train_shop_wifi_count_dict[mall_id][bssid] += 1
        test_shop_wifi_count_dict = defaultdict(lambda: defaultdict(lambda: 0))
        for mall_id, wifi_infos in zip(candidate['mall_id'].values, candidate['wifi_infos'].values):
            for wifi_info in wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                test_shop_wifi_count_dict[mall_id][bssid] += 1
        def f_1(row):
            mall_id = row.mall_id
            result = [wifi_info.split('|') for wifi_info in row.wifi_infos.split(';')]
            result = ';'.join(
                ['|'.join([wifi_info[0], str(1.014 ** float(wifi_info[1])), wifi_info[2]]) for wifi_info in result
                 if test_shop_wifi_count_dict[mall_id][wifi_info[0]] > 0])
            return result
        def f_2(row):
            mall_id = row.mall_id
            result = [wifi_info.split('|') for wifi_info in row.wifi_infos.split(';')]
            result = ';'.join(
                ['|'.join([wifi_info[0], str(1.014 ** float(wifi_info[1])), wifi_info[2]]) for wifi_info in result
                 if train_shop_wifi_count_dict[mall_id][wifi_info[0]] > 0])
            return result
        data['wifi_infos'] = data.apply(f_1, axis=1)
        candidate['wifi_infos'] = candidate.apply(f_2, axis=1)
        shop = pd.read_csv(shop_path)
        data = data.merge(shop[['shop_id', 'category_id']], on='shop_id', how='left')
        data.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        candidate.to_hdf(candidate_path, 'w', complib='blosc', complevel=5)
    return (data,candidate)

###################################################
#................... 选取备选样本 ...................
###################################################
# 用连接的wifi选取样本
def get_cwifi_sample(data,candidate,data_key):
    result_path = cache_path + 'cwifi_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp['bssid'] = candidate_temp['wifi_infos'].apply(get_connect_wifi)
        shop_cwifi_count = get_shop_cwifi_count(data, data_key)
        shop_cwifi_count = group_rank(shop_cwifi_count,'bssid','shop_cwifi_count',ascending=False)
        shop_cwifi_count = shop_cwifi_count[shop_cwifi_count['rank']<3]
        result = candidate_temp.merge(shop_cwifi_count,on='bssid',how='inner')
        result = result[['row_id', 'shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用简单tfidf选取前3样本
def get_tfidf_sample(data,candidate,data_key):
    result_path = cache_path + 'tfidf_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_tfidf = get_shop_wifi_tfidf(data, data_key)
        shop_wifi_tfidf_dict = {}
        for shop_id, grp in shop_wifi_tfidf.groupby('shop_id'):
            wifi_idf = {}
            for tuple in grp.itertuples():
                wifi_idf[tuple.bssid] = tuple.tfidf
                shop_wifi_tfidf_dict[shop_id] = wifi_idf
        shop = pd.read_csv(shop_path)
        mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()
        result = []
        for row in candidate.itertuples():
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                wifi_infos = [wifi.split('|') for wifi in wifi_infos.split(';')]
                shops = mall_shop_dict[row.mall_id]
                for shop_id in shops:
                    shop_tfidf = 0
                    for i, (bssid,signal,Flag) in enumerate(wifi_infos):
                        try:
                            idf = shop_wifi_tfidf_dict[shop_id][bssid] * rank_weight(i)
                            shop_tfidf += idf
                        except:
                            pass
                    if shop_tfidf>0:
                        result.append([row.row_id,shop_id,shop_tfidf])
        result = pd.DataFrame(result,columns=['row_id','shop_id','shop_tfidf'])
        result = group_rank(result,'row_id','shop_tfidf',ascending=False)
        result = result[result['rank']<3][['row_id', 'shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用最强信号选取前二样本
def get_mwifi_sample(data,candidate,data_key):
    result_path = cache_path + 'mwifi_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp['bssid'] = candidate_temp['wifi_infos'].apply(get_most_wifi)
        shop_mwifi_count = get_shop_mwifi_count(data, data_key)
        shop_mwifi_count = group_rank(shop_mwifi_count, 'bssid', 'shop_mwifi_count', ascending=False)
        shop_mwifi_count = shop_mwifi_count[shop_mwifi_count['rank'] < 3]
        result = candidate_temp.merge(shop_mwifi_count, on='bssid', how='inner')[['row_id', 'shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用简单knn选取前3样本
def get_knn_sample(data, candidate, data_key):
    result_path = cache_path + 'knn_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            for wifi_info in wifi_infos.split(';'):
                try:
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, float(signal)])
                except:
                    pass
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi['signal'] = shop_wifi['signal'].apply(lambda x: x ** 1.1)
        shop_wifi_mean_signal_dict = defaultdict(lambda: 1.014 ** (-104))
        shop_wifi_mean_signal_dict.update((shop_wifi.groupby(['shop_id', 'bssid'])['signal'].mean() ** (1 / 1.1)).to_dict())
        mall_shop_dict = data.groupby('mall_id')['shop_id'].unique().to_dict()
        def knn_loss(shop_id, wifis):
            loss = 0
            for bssid in wifis:
                diff = wifis[bssid] - shop_wifi_mean_signal_dict[(shop_id, bssid)]
                loss += diff * diff
            return loss
        result = []
        for row in candidate.itertuples():
            wifis = {}
            for wifi_infos in row.wifi_infos.split(';'):
                try:
                    bssid, signal, Flag = wifi_infos.split('|')
                except:
                    pass
                wifis[bssid] = float(signal)
            shops = mall_shop_dict[row.mall_id]
            for shop_id in shops:
                result.append([row.row_id,shop_id,knn_loss(shop_id, wifis)])
        result = pd.DataFrame(result,columns=['row_id','shop_id','knn'])
        result = group_rank(result, 'row_id', 'knn', ascending=True)
        result = result[result['rank'] < 3][['row_id','shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户去过的商店
def get_people_sample(data,candidate,data_key):
    result_path = cache_path + 'people_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_shop_count = data.groupby(['user_id', 'shop_id'], as_index=False)['user_id'].agg(
            {'user_shop_count': 'count'})
        result = candidate.merge(user_shop_count,on='user_id',how='inner')
        result = result[['row_id','shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 根据坐标添加前三
def get_loc_sample(data,candidate,data_key):
    result_path = cache_path + 'loc_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data['cwifi'] = data['wifi_infos'].apply(get_connect_wifi)
        data = data[~data['cwifi'].isnull()]
        shop_loc_dict = {}
        for shop_id, grp in data.groupby('shop_id'):
            locs = []
            for row in grp.itertuples():
                locs.append((row.longitude, row.latitude))
            shop_loc_dict[shop_id] = locs
        shop = pd.read_csv(shop_path)
        mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()
        def loc_knn2_loss(shop_id, longitude, latitude):
            loss = 0
            try:
                locs = shop_loc_dict[shop_id]
                for (lon, lat) in locs:
                    loss += 0.1 ** (((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5 * 100000)
            except:
                loss = np.nan
            return loss
        result = []
        candidate_temp = candidate.copy()
        candidate_temp['cwifi'] = candidate_temp['wifi_infos'].apply(get_connect_wifi)
        for row in candidate_temp.itertuples():
            if row.cwifi is np.nan:
                longitude = row.longitude
                latitude = row.latitude
                for shop_id in mall_shop_dict[row.mall_id]:
                    result.append([row.row_id,shop_id,loc_knn2_loss(shop_id, longitude, latitude)])
        result = pd.DataFrame(result, columns=['row_id','shop_id','loc_knn2'])
        result = group_rank(result, 'row_id', 'loc_knn2', ascending=False)
        result = result[result['rank'] < 3][['row_id', 'shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 根据多分类结果获取top3
def get_multi_sample(candidate,data_key):
    result_path = cache_path + 'multi_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        multi_pred = pd.read_hdf(r'C:\Users\csw\Desktop\python\mayi\data\multi_pred_25-31.hdf', 'w')
        multi_pred = group_rank(multi_pred, 'row_id', 'multi_pred', ascending=False)
        multi_pred = multi_pred[multi_pred['rank']<3]
        result = candidate.merge(multi_pred, on=['row_id'], how='left')[['row_id','shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 根据plant结果获取top3
def get_plant_sample(candidate, data_key):
    result_path = cache_path + 'plant_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        plant_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\mayi\data\plant_pred.csv')
        plant_pred = group_rank(plant_pred, 'row_id', 'plant_pred', ascending=False)
        plant_pred = plant_pred[plant_pred['rank'] < 3]
        result = candidate.merge(plant_pred, on=['row_id'], how='left')[['row_id', 'shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

###################################################
#..................... 构造特征 ....................
###################################################
# 连接wifi特征
def get_cwifi_feat(data,sample,data_key):
    result_path = cache_path + 'cwifi_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sample_temp = sample.copy()
        sample_temp['bssid'] = sample_temp['wifi_infos'].apply(get_connect_wifi)
        shop_cwifi_count = get_shop_cwifi_count(data, data_key)
        cwifi_count = shop_cwifi_count.groupby('bssid',as_index=False)['shop_cwifi_count'].agg({'cwifi_count':'sum'})
        shop_cwifi_count = shop_cwifi_count.merge(cwifi_count,on='bssid',how='left')
        shop_cwifi_count['shop_cwifi_rate'] = shop_cwifi_count['shop_cwifi_count']/shop_cwifi_count['cwifi_count']
        result = sample_temp.merge(shop_cwifi_count,on=['shop_id','bssid'],how='left')
        result = result[['shop_cwifi_count', 'cwifi_count', 'shop_cwifi_rate']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# tfidf特征
def get_tfidf_feat(data,sample,data_key):
    result_path = cache_path + 'tfidf_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_tfidf = get_shop_wifi_tfidf(data, data_key)
        shop_wifi_tfidf_dict = {}
        for shop_id, grp in shop_wifi_tfidf.groupby('shop_id'):
            wifi_idf = {}
            for tuple in grp.itertuples():
                wifi_idf[tuple.bssid] = tuple.tfidf
                shop_wifi_tfidf_dict[shop_id] = wifi_idf
        result = []
        row_tfidf_dict = defaultdict(lambda : 0)
        for row in sample.itertuples():
            wifi_infos = row.wifi_infos
            shop_id = row.shop_id
            tfidf = 0
            if wifi_infos != '':
                wifi_infos = [wifi.split('|') for wifi in wifi_infos.split(';')]
                for i, (bssid,signal,Flag) in enumerate(wifi_infos):
                    try:
                        tfidf += shop_wifi_tfidf_dict[shop_id][bssid] * rank_weight(i)
                    except:
                        pass
            result.append([row.row_id,row.shop_id,tfidf])
            row_tfidf_dict[row.row_id] += tfidf
        result = pd.DataFrame(result,columns=['row_id','shop_id','tfidf'])
        result = sample.merge(result,on=['row_id','shop_id'],how='left')
        result['row_tfidf'] = result['row_id'].map(row_tfidf_dict)
        result = result[['tfidf','row_tfidf']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用最强信号选取前二样本
def get_mwifi_feat(data,sample,data_key):
    result_path = cache_path + 'mwifi_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sample_temp = sample.copy()
        sample_temp['bssid'] = sample_temp['wifi_infos'].apply(get_most_wifi)
        shop_mwifi_count = get_shop_mwifi_count(data, data_key)
        mwifi_count = shop_mwifi_count.groupby('bssid',as_index=False)['shop_mwifi_count'].agg({'mwifi_count':'sum'})
        shop_mwifi_count = shop_mwifi_count.merge(mwifi_count, on='bssid', how='left')
        shop_mwifi_count['shop_mwifi_rate'] = shop_mwifi_count['shop_mwifi_count'] / shop_mwifi_count['mwifi_count']
        result = sample_temp.merge(shop_mwifi_count, on=['shop_id','bssid'], how='left')
        result = result[['shop_mwifi_count', 'mwifi_count', 'shop_mwifi_rate']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# knn
def get_knn_feat(data,sample,data_key):
    result_path = cache_path + 'knn_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_mean_signal_dict = defaultdict(lambda: 1.014**(-104))
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, float(signal)])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi['signal'] = shop_wifi['signal']**0.0005
        shop_wifi_mean_signal_dict.update((shop_wifi.groupby(['shop_id', 'bssid'])['signal'].mean()**(1/0.0005)).to_dict())
        def knn_loss(shop_id, wifis):
            loss = 0
            for bssid in wifis:
                diff = wifis[bssid] - shop_wifi_mean_signal_dict[(shop_id, bssid)]
                loss += diff * diff
            return loss
        result = []
        for row in sample.itertuples():
            shop_id = row.shop_id
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_infos in row.wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_infos.split('|')
                    wifis[bssid] = float(signal)
            result.append([knn_loss(shop_id, wifis)])
        result = pd.DataFrame(result,columns=['knn'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# wifi余弦相似度
def get_cos_feat(data,sample,data_key):
    result_path = cache_path + 'cos_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_mean_signal_dict = defaultdict(lambda: 1.014**(-104))
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, float(signal)])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi['signal'] = shop_wifi['signal']**0.0005
        shop_wifi_mean_signal_dict.update((shop_wifi.groupby(['shop_id', 'bssid'])['signal'].mean()**(1/0.0005)).to_dict())
        def knn_loss(shop_id, wifis):
            loss = 0; wlen = 0; slen = 0
            for bssid in wifis:
                a = wifis[bssid]
                b = shop_wifi_mean_signal_dict[(shop_id, bssid)]
                loss += a * b
                slen += a * a
                wlen += b * b
            loss_cos = loss / (slen ** 0.5 * wlen ** 0.5)
            return loss_cos
        result = []
        for row in tqdm(sample.itertuples()):
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifis[bssid] = float(signal)
                result.append(knn_loss(row.shop_id, wifis))
            else:
                result.append(np.nan)
        result = pd.DataFrame(result, columns=['cos'])
        mean_cos = result['cos'].values.mean()
        result.fillna(mean_cos,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# knn2
def get_knn2_feat(data,sample,data_key):
    result_path = cache_path + 'knn2_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_signal_dict = defaultdict(lambda: [])
        for shop_id, wifi_infos in zip(data['shop_id'].values, data['wifi_infos'].values):
            if wifi_infos != '':
                wifi_signal_dict = defaultdict(lambda: 1.014**(-104))
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifi_signal_dict[bssid] = float(signal)
                shop_wifi_signal_dict[shop_id].append(wifi_signal_dict)
        def knn2_loss(shop_id, wifis):
            loss = []
            for dt_wifis in shop_wifi_signal_dict[shop_id]:
                single_loss = 0
                for bssid in wifis:
                    diff = wifis[bssid] - dt_wifis[bssid]
                    single_loss += diff * diff
                loss.append(0.87 ** (single_loss * 1000))
            if len(loss)!=0:
                return [sum(loss),np.max(loss),np.min(loss),np.mean(loss),np.median(loss),np.std(loss)]
            return [0,0,0,0,0,0]
        result = []
        for row in tqdm(sample.itertuples()):
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_infos in row.wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_infos.split('|')
                    wifis[bssid] = float(signal)
            result.append(knn2_loss(row.shop_id, wifis))
        result = pd.DataFrame(result,columns=['knn2','knn2_max','knn2_min','knn2_mean','knn2_median','knn2_std'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 商店出现的次数
def get_shop_count(data,sample,data_key):
    result_path = cache_path + 'shop_count_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_count = data.groupby('shop_id',as_index=False)['user_id'].agg({'shop_count':'count',
                                                                            'shop_n_user':'nunique'})
        result = sample.merge(shop_count,on='shop_id',how='left')[['shop_count','shop_n_user']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户去过此商店的次数
def get_user_shop_count(data,sample,data_key):
    result_path = cache_path + 'user_shop_count_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_count = data.groupby(['user_id'], as_index=False)['user_id'].agg(
            {'user_count': 'count'})
        user_shop_count = data.groupby(['user_id','shop_id'],as_index=False)['user_id'].agg(
            {'user_shop_count':'count'})
        result = sample.merge(user_count, on=['user_id'], how='left')
        result = result.merge(user_shop_count, on=['user_id', 'shop_id'], how='left')
        result['user_shop_rate'] = result['user_shop_count']/result['user_count']
        result = result[['user_shop_count','user_count','user_shop_rate']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 经纬度knn2
def get_loc_knn2_feat(data,sample,data_key):
    result_path = cache_path + 'loc_knn2_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_loc_dict = {}
        for shop_id, grp in data.groupby('shop_id'):
            locs = []
            for row in grp.itertuples():
                locs.append((row.longitude, row.latitude))
            shop_loc_dict[shop_id] = locs

        def loc_knn2_loss(shop_id, longitude, latitude):
            loss = []
            try:
                locs = shop_loc_dict[shop_id]
                for (lon, lat) in locs:
                    loss.append(0.1 ** (((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5 * 100000))
            except:
                return (0,0,0,0,0)
            return (sum(loss),np.mean(loss),np.median(loss),np.min(loss),np.max(loss))

        result = []
        for row in tqdm(sample.itertuples()):
            longitude = row.longitude;
            latitude = row.latitude
            result.append(loc_knn2_loss(row.shop_id, longitude, latitude))
        result = pd.DataFrame(result,columns=['loc_knn2','loc_knn2_mean','loc_knn2_median','loc_knn2_min','loc_knn2_max'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 距离店铺中心的距离
def get_loc_knn_feat(data, sample, data_key):
    result_path = cache_path + 'loc_knn_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data['cwifi'] = data['wifi_infos'].apply(get_connect_wifi)
        data = data[~data['cwifi'].isnull()]
        shop_loc_mean_dict = data[['shop_id', 'longitude', 'latitude']].groupby('shop_id').mean()
        shop_loc_mean_dict = dict(zip(shop_loc_mean_dict.index, shop_loc_mean_dict.values))

        def loc_knn_loss(shop_id, longitude, latitude):
            try:
                lon, lat = shop_loc_mean_dict[shop_id]
                loss = ((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5
            except:
                loss = np.nan
            return loss

        result = []
        sample_temp = sample.copy()
        sample_temp['cwifi'] = sample_temp['wifi_infos'].apply(get_connect_wifi)
        for row in sample_temp.itertuples():
            if row.cwifi is np.nan:
                longitude = row.longitude
                latitude = row.latitude
                result.append(loc_knn_loss(row.shop_id, longitude, latitude))
        result = pd.DataFrame(result, columns=['loc_knn'])
        mean_loc_knn = result['loc_knn'].values.mean()
        result.fillna(mean_loc_knn,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# multi_pred
def get_multi_pred(sample):
    multi_pred = pd.read_hdf(r'C:\Users\csw\Desktop\python\mayi\data\multi_pred_25-31.hdf', 'w')
    result = sample.merge(multi_pred,on=['row_id','shop_id'],how='left')
    result = result[['multi_pred']]
    return result

# plant_pred
def get_plant_pred(sample):
    plant_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\mayi\data\plant_pred.csv')
    result = sample.merge(plant_pred, on=['row_id', 'shop_id'], how='left')
    result = result[['plant_pred']]
    return result

# kunkun_pred
def get_kunkun_pred(sample):
    plant_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\mayi\data\kunkun_pred.csv')
    result = sample.merge(plant_pred, on=['row_id', 'shop_id'], how='left')
    result = result[['kunkun_pred']]
    return result

# wajue_pred
def get_wajue_pred(sample):
    wajue_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\mayi\data\wajue_pred.csv').drop_duplicates(['row_id','shop_id'])
    result = sample.merge(wajue_pred, on=['row_id', 'shop_id'], how='left')
    result = result[['wajue_pred']]
    return result

# category knn
def get_category_knn2_feat(data, sample, data_key):
    result_path = cache_path + 'category_knn2_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.sort_values('time_stamp').tail(100000)
        category_wifi_signal_dict = defaultdict(lambda: [])
        for category_id, wifi_infos in zip(data_temp['category_id'].values, data_temp['wifi_infos'].values):
            if wifi_infos != '':
                wifi_signal_dict = defaultdict(lambda: 1.014 ** (-104))
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifi_signal_dict[bssid] = float(signal)
                category_wifi_signal_dict[category_id].append(wifi_signal_dict)

        def knn2_loss(category_id, wifis):
            loss = 0
            for dt_wifis in category_wifi_signal_dict[category_id]:
                single_loss = 0
                for bssid in wifis:
                    diff = wifis[bssid] - dt_wifis[bssid]
                    single_loss += diff * diff
                loss += 0.87 ** (single_loss * 1000)
            return loss
            # if len(loss) != 0:
            #     return [sum(loss), np.max(loss), np.min(loss), np.mean(loss), np.median(loss), np.std(loss)]
            # return [0, 0, 0, 0, 0, 0]

        result = []
        for row in tqdm(sample.itertuples()):
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_infos in row.wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_infos.split('|')
                    wifis[bssid] = float(signal)
            result.append(knn2_loss(row.category_id, wifis))
        # result = pd.DataFrame(result, columns=['knn2', 'knn2_max', 'knn2_min', 'knn2_mean', 'knn2_median', 'knn2_std'])
        result = pd.DataFrame(result, columns=['category_knn2'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 二次处理特征
def second_feat(result):
    shop = pd.read_csv(shop_path)
    result = result.merge(shop[['shop_id','price']],on='shop_id',how='left')
    result['category_id'] = result['category_id'].str[2:].astype(int)
    result['mall_id'] = result['mall_id'].str[2:].astype(int)
    return result

# 构造样本
def get_sample(data, candidate, data_key):
    result_path = cache_path + 'sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        cwifi_sample = get_cwifi_sample(data,candidate,data_key)    # 用连接的wifi选取样本
        tfidf_sample = get_tfidf_sample(data,candidate,data_key)    # 用简单tfidf选取前3样本
        mwifi_sample = get_mwifi_sample(data,candidate,data_key)    # 用最强信号选取前二样本
        knn_sample = get_knn_sample(data, candidate, data_key)      # 用简单knn选取前3样本
        people_sample = get_people_sample(data,candidate,data_key)  # 用户去过的商店（全部）
        loc_sample = get_loc_sample(data,candidate,data_key)        # 根据坐标添加前二
        multi_sample = get_multi_sample(candidate, data_key)        # 根据多分类结果获取top3
        plant_sample = get_plant_sample(candidate, data_key)        # 根据plant结果获取top3

        # 汇总样本id
        result = pd.concat([cwifi_sample,
                            tfidf_sample,
                            mwifi_sample,
                            knn_sample,
                            people_sample,
                            loc_sample,
                            multi_sample,
                            plant_sample]).drop_duplicates()
        # 剔除错误样本
        shop = pd.read_csv(shop_path)
        shop_mall_dict = dict(zip(shop['shop_id'].values,shop['mall_id']))
        result = result.merge(candidate[['row_id','user_id','mall_id','longitude','latitude',
                                         'time_stamp','wifi_infos']],on='row_id',how='left')
        result = result[result['shop_id'].map(shop_mall_dict) == result['mall_id']]
        shop = pd.read_csv(shop_path)
        result = result.merge(shop[['shop_id', 'category_id']], on='shop_id', how='left')
        result.index = list(range(result.shape[0]))
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('样本个数为：{}'.format(result.shape))
    return result


# 制作训练集
def make_feats(data, candidate):
    t0 = time.time()
    data_key = hashlib.md5(data['time_stamp'].to_string().encode()).hexdigest()+\
               hashlib.md5(candidate['time_stamp'].to_string().encode()).hexdigest()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'train_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        print('清洗WiFi...')
        data, candidate = clear(data, candidate, data_key)
        print('开始构造样本...')
        sample = get_sample(data, candidate, data_key)

        print('开始构造特征...')
        cwifi_feat = get_cwifi_feat(data, sample, data_key)             # 连接的wifi个数
        tfidf_feat = get_tfidf_feat(data, sample, data_key)             # tfidf
        mwifi_feat = get_mwifi_feat(data, sample, data_key)             # 最强信号个数
        knn_feat = get_knn_feat(data,sample,data_key)                   # knn
        cos_feat = get_cos_feat(data,sample,data_key)                   # wifi余弦相似度
        knn2_feat = get_knn2_feat(data, sample, data_key)               # knn2
        # 从单个wifi角度来判断
        # 用auc从相对大小来判断
        # 用户是否连接过wifi与商店对应的最强wifi是否有过交集

        shop_count = get_shop_count(data,sample,data_key)                       # 商店出现的次数
        user_shop_count = get_user_shop_count(data,sample,data_key)             # 用户去过此商店的次数
        # 用户是否去过此种类型的商店

        loc_knn2_feat = get_loc_knn2_feat(data, sample, data_key)               # 经纬度knn2
        loc_knn_feat = get_loc_knn_feat(data, sample, data_key)                 # 距离店铺中心的距离

        # categroy_knn2_feat = get_category_knn2_feat(data, sample, data_key)     # category knn
        # category loc_knn
        # category 出现的次数
        # 用户去过这个category的次数

        # # multi_pred
        # multi_pred = get_multi_pred(sample)
        # # plant——pred
        # plant_pred = get_plant_pred(sample)
        # # kunkun——pred
        # kunkun_pred = get_kunkun_pred(sample)
        # # wajue——pred
        # wajue_pred = get_wajue_pred(sample)


        print('开始合并特征...')
        result = concat([sample,cwifi_feat,tfidf_feat,mwifi_feat,knn_feat,cos_feat,
                         shop_count,user_shop_count,knn2_feat,loc_knn2_feat,
                         loc_knn_feat])
        result = second_feat(result)


        print('添加label')
        result = get_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result






















