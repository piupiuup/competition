import os
import gc
import time
import pickle
import Geohash
import numpy as np
import pandas as pd


cache_path = 'mobike_cache1/'
train_path = 'data/train.csv'
test_path = 'data/test.csv'
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

# 计算两点之间距离
def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

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

# 获取真实标签
def get_label(data):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test['geohashed_end_loc'] = np.nan
        data = pd.concat([train,test])
        true = dict(zip(data['orderid'].values, data['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data['label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    return data


####################构造负样本##################

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(train,test):
    result_path = cache_path + 'user_end_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_eloc = train[['userid','geohashed_end_loc']].drop_duplicates()
        result = pd.merge(test[['orderid','userid']],user_eloc,on='userid',how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(train,test):
    result_path = cache_path + 'user_start_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_sloc = train[['userid', 'geohashed_start_loc']].drop_duplicates()
        result = pd.merge(test[['orderid', 'userid']], user_sloc, on='userid', how='left')
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 筛选起始地点去向最多的3个地点
def get_loc_to_loc(train,test):
    result_path = cache_path + 'loc_to_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
        sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(3)
        result = pd.merge(test[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 获取用户历史行为次数
def get_user_count(train,result):
    user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
    result = pd.merge(result,user_count,on=['userid'],how='left')
    return result

# 获取用户去过某个地点历史行为次数
def get_user_eloc_count(train, result):
    user_eloc_count = train.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_count':'count'})
    result = pd.merge(result,user_eloc_count,on=['userid','geohashed_end_loc'],how='left')
    return result

# 获取用户从某个地点出发的行为次数
def get_user_sloc_count(train,result):
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'],as_index=False)['userid'].agg({'user_sloc_count':'count'})
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = pd.merge(result, user_sloc_count, on=['userid', 'geohashed_end_loc'], how='left')
    return result

# 获取用户从这个路径走过几次
def get_user_sloc_eloc_count(train,result):
    user_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_sloc_eloc_count':'count'})
    result = pd.merge(result,user_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 获取用户从这个路径折返过几次
def get_user_eloc_sloc_count(train,result):
    user_eloc_sloc_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_sloc_count':'count'})
    user_eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    result = pd.merge(result,user_eloc_sloc_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 计算两点之间的欧氏距离
def get_distance(result):
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(Geohash.decode(loc))
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
    result.loc[:,'distance'] = distance
    return result

# 获取目标地点的热度(目的地)
def get_eloc_count(train,result):
    eloc_count = train.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'eloc_count': 'count'})
    result = pd.merge(result, eloc_count, on='geohashed_end_loc', how='left')
    return result

# 获取目标地点的热度(出发地地)
def get_eloc_as_sloc_count(train,result):
    eloc_as_sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'eloc_as_sloc_count': 'count'})
    eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_start_loc'})
    result = pd.merge(result, eloc_as_sloc_count, on='geohashed_start_loc', how='left')
    return result

# 构造样本
def get_sample(train,test):
    result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(train, test)            # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_end_loc']
        user_start_loc = get_user_start_loc(train, test)        # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_start_loc']
        loc_to_loc = get_loc_to_loc(train, test)                # 筛选起始地点去向最多的3个地点
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid','geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            ]).drop_duplicates()
        # 根据end_loc添加标签(0,1)
        test_temp = test.copy()
        test_temp.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
        result = pd.merge(result, test_temp, on='orderid', how='left')
        result['label'] = (result['label'] == result['geohashed_end_loc']).astype(int)
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 制作训练集
def make_train_set(train,test):
    print('开始构造样本...')
    result = get_sample(train,test)                                         # 构造备选样本

    print('开始构造特征...')
    result = get_user_count(train,result)                                   # 获取用户历史行为次数
    result = get_user_eloc_count(train, result)                             # 获取用户去过这个地点几次
    result = get_user_sloc_count(train, result)                             # 获取用户从目的地点出发过几次
    result = get_user_sloc_eloc_count(train, result)                        # 获取用户从这个路径走过几次
    result = get_user_eloc_sloc_count(train, result)                        # 获取用户从这个路径折返过几次
    result = get_distance(result)                                           # 获取起始点和最终地点的欧式距离
    result = get_eloc_count(train, result)                                  # 获取目的地点的热度(目的地)
    result = get_eloc_as_sloc_count(train, result)                          # 获取目的地点的热度(出发地)
    result.fillna(0,inplace=True)
    print('result.columns:\n{}'.format(result.columns))

    print('添加真实label')
    result = get_label(result)
    return result



# 训练提交

if __name__ == "__main__":
    t0 = time.time()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
    train2 = train[(train['starttime']>= '2017-05-23 00:00:00')]
    train2.loc[:,'geohashed_end_loc'] = np.nan
    test.loc[:,'geohashed_end_loc'] = np.nan

    print('构造训练集')
    train_feat = make_train_set(train1,train2)
    print('构造线上测试集')
    test_feat = make_train_set(train,test)
    del train,test,train1,train2


    import xgboost as xgb
    predictors = [ 'biketype','user_count',
       'user_eloc_count', 'user_sloc_count', 'user_sloc_eloc_count',
       'user_eloc_sloc_count', 'distance', 'eloc_count', 'eloc_as_sloc_count']
    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 10,
        'subsample': 0.886,
        'alpha': 10,
        'gamma': 30,
        'lambda':50,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'auc',
        'scale_pos_weight': 10,
        'seed': 201703,
        'missing':-1
    }

    xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['label'])
    xgbtest = xgb.DMatrix(test_feat[predictors])
    model = xgb.train(params, xgbtrain, num_boost_round=120)
    del train_feat,xgbtrain
    gc.collect()

    test_feat.loc[:,'pred'] = model.predict(xgbtest)
    result = reshape(test_feat)
    test = pd.read_csv(test_path)
    result = pd.merge(test[['orderid']],result,on='orderid',how='left')
    result.fillna('0',inplace=True)
    result.to_csv('result.csv',index=False,header=False)
    print('一共用时{}秒'.format(time.time()-t0))








