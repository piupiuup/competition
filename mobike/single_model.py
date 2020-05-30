import os
import gc
import time
import random
import pickle
import Geohash
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold


cache_path = 'F:/mobike_single_cache/'
train_path = 'C:/Users/csw/Desktop/python/mobike/data/train.csv'
test_path = 'C:/Users/csw/Desktop/python/mobike/data/test.csv'

flag = True
# 相差的分钟数
def diff_of_minutes(time1, time2):
    d = {'5': 0, '6': 31, }
    try:
        days = (d[time1[6]] + int(time1[8:10])) - (d[time2[6]] + int(time2[8:10]))
        try:
            minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        except:
            minutes1 = 0
        try:
            minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        except:
            minutes2 = 0
        return (days * 1440 - minutes2 + minutes1)
    except:
        return np.nan

# 判断是否为节假日:
def if_holiday(date):
    holiday_list = [13,14,20,21,28,29,30]
    return 1 if int(date) in holiday_list else 0

# 计算夹角余弦值
def cal_cos(start_lat,start_lon,end_lat,end_lon,act_start_lat,act_start_lon,act_end_lat,act_end_lon):
    x1 = end_lon - start_lon
    y1 = end_lat - start_lat
    x2 = act_end_lon - act_start_lon
    y2 = act_end_lat - act_start_lat
    result = (x1*x2 + y1*y2) / (x1**2 + y1**2 + x2**2 + y2**2)**0.5
    return result

# 组内标准化
def group_normalize(data,key,feat):
    grp = data.groupby(key,as_index=False)[feat].agg({'std':'std','avg':'mean'})
    result = pd.merge(data,grp,on=key,how='left')
    result[feat] = ((result[feat]-result['std']) / result['avg']).fillna(1)
    return result[feat]

# 计算两点之间距离
def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

# 计算两点之间距离
def cal_mht_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = Lx + Ly
    return L

# 相差的分钟数
def diff_of_minutes(time1, time2):
    try:
        minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        return min((np.abs(minutes1-minutes2)),(1440-np.abs(minutes1-minutes2)))
    except:
        return np.nan

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 解码字典
def get_loc_dict():
    dump_path = cache_path + 'loc_dict.pkl'
    if os.path.exists(dump_path):
        loc_dict = pickle.load(open(dump_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        locs = list(set(train['geohashed_start_loc']) | set(train['geohashed_end_loc']) | set(test['geohashed_start_loc']))
        deloc = []
        for loc in locs:
            deloc.append(Geohash.decode(loc))
        loc_dict = dict(zip(locs, deloc))
        pickle.dump(loc_dict, open(dump_path, 'wb+'))
    return loc_dict

# 计算两点之间的欧氏距离和曼哈顿距离
def get_distance(sample):
    result = sample.copy()
    loc_dict = get_loc_dict()
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    mht_distance = []
    for i in geohashed_loc:
        loc1, loc2 = i
        if (loc1 is np.nan) | (loc2 is np.nan):
            distance.append(np.nan)
            mht_distance.append(np.nan)
            continue
        lat1, lon1 = loc_dict[loc1]
        lat2, lon2 = loc_dict[loc2]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
        mht_distance.append(cal_mht_distance(lat1,lon1,lat2,lon2))
    result['distance'] = distance
    result['mht_distance'] = mht_distance
    result = result[['distance','mht_distance']]
    return result

# 对结果进行整理
def reshape(pred):
    result = pred.copy()
    result = rank(result,'orderid','pred',ascending=False)
    result = result[result['rank']<3][['orderid','geohashed_end_loc','rank']]
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid', 0, 1, 2]
    return result

# 测评函数
def map(result):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values,train['geohashed_end_loc']))
        pickle.dump(true,open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true']==data[0])
             +sum(data['true']==data[1])/2
             +sum(data['true']==data[2])/3)/data.shape[0]
    return score

# 对结果添加噪音
def get_noise(data,rate=0.1,seed=66):
    random.seed(seed)
    data_temp = data.copy()
    index = list(range(data.shape[0]))
    data.index = index
    random.shuffle(index)
    n = int(rate*data.shape[0])
    sub_index = index[:n]
    data_temp.loc[sub_index,[0,1,2]] = 'a'
    return data_temp

# 获取争取标签
def get_label(data):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values, train['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data.loc[:,'label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    data['label'] = data['label'].fillna(0)
    return data

# 热点目的地
def get_loc_matrix():
    result_path = cache_path + 'loc_matrix.hdf'
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        end_loc = pd.DataFrame({'geohashed_end_loc':list(train['geohashed_end_loc'].unique())})
        end_loc['end_loc_lat'] = end_loc['geohashed_end_loc'].apply(lambda x: Geohash.decode(x)[0])
        end_loc['end_loc_lon'] = end_loc['geohashed_end_loc'].apply(lambda x: Geohash.decode(x)[1])
        end_loc['end_loc_lat_box'] = end_loc['end_loc_lat'].apply(lambda x: x//0.003)
        end_loc['end_loc_lon_box'] = end_loc['end_loc_lon'].apply(lambda x: x//0.00375)
        count_of_loc = train.groupby('geohashed_end_loc',as_index=False)['geohashed_end_loc'].agg({'count_of_loc':'count'})
        end_loc = pd.merge(end_loc,count_of_loc,on='geohashed_end_loc',how='left')
        max_index = end_loc.groupby(['end_loc_lat_box','end_loc_lon_box']).apply(lambda x: x['count_of_loc'].argmax())
        end_loc = end_loc.loc[max_index.tolist(),['geohashed_end_loc', 'end_loc_lat', 'end_loc_lon']]
        end_loc.sort_values('end_loc_lat',inplace=True)
        end_loc = end_loc.values
        start_loc = pd.DataFrame({'geohashed_start_loc': list(pd.concat([train,test])['geohashed_start_loc'].unique())})
        start_loc['start_loc_lat'] = start_loc['geohashed_start_loc'].apply(lambda x: Geohash.decode(x)[0])
        start_loc['start_loc_lon'] = start_loc['geohashed_start_loc'].apply(lambda x: Geohash.decode(x)[1])
        start_loc = start_loc.values
        start_end_loc_arr = []
        for i in start_loc:
            for j in end_loc:
                if (np.abs(i[1]-j[1])<0.012) & (np.abs(i[2]-j[2])<0.015):
                    start_end_loc_arr.append([i[0],j[0]])
        result = pd.DataFrame(start_end_loc_arr,columns=['geohashed_start_loc','geohashed_end_loc'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



####################构造负样本##################

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(data,candidate):
    candidate_temp = candidate.copy()
    candidate_temp.rename(columns={'geohashed_end_loc':'true_loc'},inplace=True)
    user_n_end_loc = data.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_count':'count'})
    user_n_end_loc = user_n_end_loc[~user_n_end_loc['geohashed_end_loc'].isnull()]
    result = pd.merge(candidate_temp[['orderid','userid','true_loc']],user_n_end_loc,on=['userid'],how='left')
    result['user_eloc_count'] = result['user_eloc_count'] - (result['geohashed_end_loc']==result['true_loc'])
    result = result[result['user_eloc_count'] > 0]
    result = result[['orderid', 'geohashed_end_loc']]
    return result

# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(data,candidate):
    candidate_temp = candidate.copy()
    candidate_temp.rename(columns={'geohashed_start_loc': 'true_loc'}, inplace=True)
    user_n_start_loc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
        {'user_sloc_count': 'count'})
    result = pd.merge(candidate_temp[['orderid', 'userid', 'true_loc']], user_n_start_loc, on=['userid'], how='left')
    result['user_sloc_count'] = result['user_sloc_count'] - (result['geohashed_start_loc'] == result['true_loc'])
    result = result[result['user_sloc_count'] > 0]
    result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
    result = result[['orderid', 'geohashed_end_loc']]
    return result

# 筛选起始地点去向最多的3个地点
def get_loc_to_loc(data,candidate):
    candidate_temp = candidate.copy()
    candidate_temp.rename(columns={'geohashed_end_loc':'true_loc'},inplace=True)
    sloc_eloc_count = data.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['orderid'].agg({'sloc_eloc_count':'count'})
    sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
    sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
    sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(4)
    result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']],sloc_eloc_count,on='geohashed_start_loc',how='left')
    result['sloc_eloc_count'] = result['sloc_eloc_count'] - (result['geohashed_end_loc']==result['true_loc'])
    result.sort_values('sloc_eloc_count', inplace=True)
    result = result.groupby('orderid').tail(3)
    result = result[result['sloc_eloc_count'] > 0]
    result = result[['orderid', 'geohashed_end_loc']]
    return result

# 与其交互最多的三个地点
def get_loc_with_loc(data,candidate):
    candidate_temp = candidate.copy()
    candidate_temp.rename(columns={'geohashed_end_loc': 'true_loc'}, inplace=True)
    sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
        {'sloc_eloc_count': 'count'})
    sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
    eloc_sloc_count = sloc_eloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc',
                                                    'geohashed_end_loc':'geohashed_start_loc',
                                                    'sloc_eloc_count':'eloc_sloc_count'}).copy()
    sloc_eloc_2count = pd.merge(sloc_eloc_count,eloc_sloc_count,
                                  on=['geohashed_start_loc','geohashed_end_loc'],how='outer').fillna(0)
    sloc_eloc_2count['sloc_eloc_2count'] = sloc_eloc_2count['sloc_eloc_count'] + sloc_eloc_2count['eloc_sloc_count']
    sloc_eloc_2count.sort_values('sloc_eloc_2count', inplace=True)
    sloc_eloc_2count = sloc_eloc_2count.groupby('geohashed_start_loc').tail(4)
    result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']], sloc_eloc_2count,
                      on='geohashed_start_loc', how='left')
    result['sloc_eloc_2count'] = result['sloc_eloc_2count'] - (result['geohashed_end_loc'] == result['true_loc'])
    result.sort_values('sloc_eloc_2count', inplace=True)
    result = result.groupby('orderid').tail(3)
    result = result[result['sloc_eloc_2count']>0]
    result = result[['orderid', 'geohashed_end_loc']]
    return result

# 筛选起始地点周围流入量最大的3个地点
def get_loc_near_loc(data,candidate):
    candidate_temp = candidate.copy()
    candidate_temp.rename(columns={'geohashed_end_loc':'true_loc'},inplace=True)
    loc_matrix = get_loc_matrix()
    loc_input = data.groupby('geohashed_end_loc',as_index=False)['geohashed_end_loc'].agg({'eloc_n_input':'count'})
    loc_matrix = pd.merge(loc_matrix,loc_input,on='geohashed_end_loc',how='left')
    loc_matrix = loc_matrix[~loc_matrix['eloc_n_input'].isnull()]
    loc_matrix.sort_values('eloc_n_input',inplace=True)
    loc_matrix = loc_matrix.groupby('geohashed_start_loc').tail(4)
    result = pd.merge(candidate_temp[['orderid','geohashed_start_loc','true_loc']],loc_matrix,on='geohashed_start_loc',how='left')
    result['eloc_n_input'] = result['eloc_n_input'] - (result['geohashed_end_loc']==result['true_loc'])
    result.sort_values('eloc_n_input', inplace=True)
    result = result.groupby('orderid').tail(3)
    result = result[result['eloc_n_input'] > 0]
    result = result[['orderid','geohashed_end_loc']]
    return result

# 用户后续的起始地点
def get_user_next_loc(data,candidate):
    data_temp = data.copy()
    data_temp = data_temp[data_temp['userid'].isin(candidate['userid'].values)]
    if data_temp.shape[0] == 0:
        return pd.DataFrame(columns=['orderid','geohashed_end_loc','user_eloc_sep_time'])
    data_temp = rank(data_temp,'userid','starttime',ascending=True)
    data_temp_temp = data_temp.copy()
    data_temp_temp['rank'] = data_temp_temp['rank']-1
    result = pd.merge(data_temp[['orderid','userid','rank','starttime']],
                      data_temp_temp[['userid','rank','geohashed_start_loc','starttime']],on=['userid','rank'],how='inner')
    result['user_eloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
    result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = result[['orderid','geohashed_end_loc','user_eloc_sep_time']]
    return result

# bike后续的起始地点
def get_bike_next_loc(data,candidate):
    data_temp = data.copy()
    data_temp = data_temp[data_temp['bikeid'].isin(candidate['bikeid'].values)]
    if data_temp.shape[0] == 0:
        return pd.DataFrame(columns=['orderid', 'geohashed_end_loc', 'bike_eloc_sep_time'])
    data_temp = rank(data_temp,'bikeid','starttime',ascending=True)
    data_temp_temp = data_temp.copy()
    data_temp_temp['rank'] = data_temp_temp['rank'] - 1
    result = pd.merge(data_temp[['orderid','bikeid','rank','starttime']],
                      data_temp_temp[['bikeid','rank','geohashed_start_loc','starttime']],on=['bikeid','rank'],how='inner')
    result['bike_eloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
    result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = result[['orderid','geohashed_end_loc','bike_eloc_sep_time']]
    return result


# 构造样本
def get_sample(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'sample_%d.hdf' % (data['orderid'].sum() * candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(data, candidate)            # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_end_loc']
        user_start_loc = get_user_start_loc(data, candidate)        # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_start_loc']
        loc_to_loc = get_loc_to_loc(data, candidate)                # 筛选起始地点去向最多的3个地点
        loc_with_loc = get_loc_with_loc(data, candidate)            # 与起始地点交互最多的三个地点
        loc_near_loc = get_loc_near_loc(data, candidate)            # 筛选起始地点周围3个热门地点
        bike_next_loc = get_bike_next_loc(data, candidate)          # 自行车后续起始地点
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid','geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            loc_with_loc[['orderid', 'geohashed_end_loc']],
                            loc_near_loc[['orderid', 'geohashed_end_loc']],
                            bike_next_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc':'label'},inplace=True)
        result = pd.merge(result, candidate_temp, on='orderid', how='left')
        result['label'] = (result['label']==result['geohashed_end_loc']).astype(int)
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.index = list(range(result.shape[0]))
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 构造特征
def get_single_feat(data,candidate):
    print('样本个数：{}'.format(candidate.shape[0]))
    loc_dict = get_loc_dict()
    data_temp = data[['userid', 'starttime','geohashed_start_loc', 'geohashed_end_loc']]
    data_temp = data_temp[~data_temp['geohashed_end_loc'].isnull()]
    user_actions = dict()
    for tuple in data_temp.itertuples():
        user = int(tuple[1])
        if user_actions.__contains__(user):
            user_actions[user].append(tuple[2:])
        else:
            user_actions[user] = [tuple[2:]]

    result = []
    candidate_temp = candidate[['orderid','userid', 'starttime','geohashed_start_loc','geohashed_end_loc','label']]
    for tuple in tqdm(candidate_temp.itertuples()):
        orderid,userid,time,start_loc,end_loc,label = tuple[1:]
        start_lat, start_lon = loc_dict[start_loc]
        end_lat, end_lon = loc_dict[end_loc]
        if user_actions.__contains__(userid):
            tup = user_actions[userid]
            for action in tup:
                act_time, act_start_loc, act_end_loc = action
                act_start_lat,  act_start_lon   = loc_dict[act_start_loc]
                act_end_lat,    act_end_lon     = loc_dict[act_end_loc]
                start_dis   = int(cal_distance(start_lat, start_lon, act_start_lat, act_start_lon))
                end_dis     = int(cal_distance(end_lat, end_lon, act_end_lat, act_end_lon))
                diff_time = diff_of_minutes(time,act_time)
                holiday = sum((if_holiday(time[11:13]),if_holiday(act_time[11:13])))
                cos = cal_cos(start_lat,start_lon,end_lat,end_lon,act_start_lat,act_start_lon,act_end_lat,act_end_lon)
                result.append([orderid,end_loc,start_dis,end_dis,diff_time,holiday,cos,label])
    del loc_dict,data_temp,user_actions
    gc.collect()
    result = pd.DataFrame(result,columns=['orderid','geohashed_end_loc','start_dis','end_dis','diff_time','holiday','cos','label'])
    return result


# 制作训练集
def make_train_set(data,candidate):
    t0 = time.time()
    result_path = cache_path + 'single_train_set_%d.hdf' % (data['orderid'].sum() * candidate['orderid'].sum())
    if os.path.exists(result_path) & flag:
        feat = pd.read_hdf(result_path, 'w')
    else:
        # 汇总样本id
        print('开始构造样本...')
        sample = get_sample(data,candidate)
        sample = get_label(sample)
        gc.collect()

        print('开始构造特征...')
        feat = get_single_feat(data,sample)
        gc.collect()

        print('存储数据...')
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('生成特征一共用时{}秒'.format(time.time()-t0))
    return feat
#
# print('制作训练集...')
# train = pd.read_csv(train_path)
# train_eval = train[train['starttime']>='2017-05-11 00:00:00'].copy()
# test_eval = train[train['starttime']<'2017-05-11 00:00:00'].copy()
# test_eval.loc[:,'geohashed_end_loc'] = np.nan
# train_eval_feat = make_train_set(train_eval,test_eval)
#
# print('开始训练...')
# predictors = ['start_dis', 'end_dis', 'diff_time', 'holiday', 'cos']
# kf = KFold(len(train_eval_feat), n_folds = 5, shuffle=True, random_state=66)
# for (train_index, test_index) in kf:
#     lgb_train = lgb.Dataset(train_eval_feat.iloc[train_index][predictors], train_eval_feat.iloc[train_index]['label'])
#     lgb_eval = lgb.Dataset(train_eval_feat.iloc[test_index][predictors], train_eval_feat.iloc[test_index]['label'])
#
#     params = {
#         'task': 'train',
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'max_depth': 6,
#         'num_leaves': 31,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.95,
#         'bagging_freq': 5,
#         'verbose': 0,
#         'seed': 66
#     }
#
#     gbm = lgb.train(params,lgb_train,num_boost_round=1000)
#     pickle.dump(gbm, open(cache_path + 'lgb10.model', 'wb+'))
#     break

print('制作测试集...')
predictors = ['start_dis', 'end_dis', 'diff_time', 'holiday', 'cos']
gbm = pickle.load(open(cache_path + 'lgb10.model', 'rb+'))
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train,test])
data_list1 = pd.date_range('2017-05-10 00:00:00','2017-05-25 00:00:00')
for i in range(len(data_list1)-1):
    data_path = cache_path + str(data_list1[i])[:10] + '.hdf'
    if (os.path.exists(data_path) | (i == 8)) & flag:
        continue
    else:
        gc.collect()
        print('构造{}号训练集...'.format(str(data_list1[i])[8:10]))
        train1_eval = data[data['starttime'] < str(data_list1[i])].copy()
        train2_eval = data[data['starttime'] >= str(data_list1[i+1])].copy()
        test_eval = data[(data['starttime'] < str(data_list1[i+1])) &
                          (data['starttime'] >= str(data_list1[i]))].copy()
        test_eval.loc[:,'geohashed_end_loc'] = np.nan
        train_eval = pd.concat([train1_eval,train2_eval,test_eval])
        test_eval_feat = make_train_set(train_eval,test_eval)
        del train1_eval,train2_eval,test_eval
        gc.collect()
        test_eval_feat.to_hdf(data_path, 'w', complib='blosc', complevel=5)
    pred_path = cache_path + str(data_list1[i])[:10] + '_pred.hdf'
    if (os.path.exists(pred_path) | (i == 8)) & flag:
        continue
    else:
        print('开始预测...')
        pred = gbm.predict(test_eval_feat[predictors])
        loc_pred = test_eval_feat[['orderid', 'geohashed_end_loc']].copy()
        loc_pred.loc[:, 'pred'] = pred
        del test_eval_feat
        gc.collect()
        loc_pred.to_hdf(pred_path, 'w', complib='blosc', complevel=5)
        del loc_pred
data_list2 = pd.date_range('2017-05-25 00:00:00', '2017-06-02 00:00:00')
for i in range(len(data_list2) - 1):
    data_path = cache_path + str(data_list2[i])[:10] + '.hdf'
    if os.path.exists(data_path) & flag:
        continue
    else:
        gc.collect()
        print('构造{}号训练集...'.format(str(data_list2[i])[8:10]))
        test_eval = test[(test['starttime'] < str(data_list2[i + 1])) &
                          (test['starttime'] >= str(data_list2[i]))].copy()
        test_eval.loc[:, 'geohashed_end_loc'] = np.nan
        test_eval_feat = make_train_set(data, test_eval)
        del test_eval
        gc.collect()
        test_eval_feat.to_hdf(data_path, 'w', complib='blosc', complevel=5)
    pred_path = cache_path + str(data_list1[i])[:10] + '_pred.hdf'
    if os.path.exists(pred_path) & (i == 8) & flag:
        continue
    else:
        print('开始预测...')
        pred = gbm.predict(test_eval_feat[predictors])
        loc_pred = test_eval_feat[['orderid', 'geohashed_end_loc']].copy()
        loc_pred.loc[:, 'pred'] = pred
        del test_eval_feat
        gc.collect()
        loc_pred.to_hdf(cache_path + str(data_list2[i])[:10] + '_pred.hdf', 'w', complib='blosc', complevel=5)
        del loc_pred
print('开始预测...')
gbm = pickle.load(open(cache_path + 'lgb10.model','rb+'))
pred = gbm.predict(test_eval_feat[predictors])
loc_pred = test_eval_feat[['orderid','geohashed_end_loc']]
loc_pred.iloc[:,'pred'] = pred
loc_pred.to_hdf(cache_path + 'loc_pred.hdf', 'w', complib='blosc', complevel=5)
loc_pred = pd.read_hdf(cache_path + 'loc_pred.hdf', 'w')
print('贝叶斯概率转化...')

order_eloc_pred = loc_pred.groupby(['orderid','geohashed_end_loc'],as_index=False)['pred'].agg({
    'pred':lambda x:np.prod(x),
    'no_pred':lambda x:np.prod(1-x)})
order_eloc_pred['sum_pred'] = order_eloc_pred['no_pred'] + order_eloc_pred['pred']
order_eloc_pred['no_pred'] = order_eloc_pred['no_pred'] / order_eloc_pred['sum_pred']
order_eloc_pred['pred'] = order_eloc_pred['pred'] / order_eloc_pred['sum_pred']

# order_eloc_pred = pd.merge(order_eloc_pred,order_pred,on='orderid',how='left')
# order_eloc_pred['pred'] = order_eloc_pred['order_pred'] - (1-order_eloc_pred['order_pred'])/order_eloc_pred['order_eloc_no_pred']
order_eloc_pred = loc_pred.groupby(['orderid','geohashed_end_loc'],as_index=False)['pred'].agg({
    'pred':lambda x:'sum'})
result = reshape(order_eloc_pred)
result = pd.merge(test_eval[['orderid']],result,on='orderid',how='left')
print('map得分为:{}'.format(map(result)))











