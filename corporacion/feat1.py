import os
import time
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error


cache_path = 'F:/corporacion_cache/'
data_path = 'C:/Users/csw/Desktop/python/Corporacion/data/'

holiday = pd.read_csv(data_path + 'holidays_events.csv')
item = pd.read_csv(data_path + 'items.csv')
oil = pd.read_csv(data_path + 'oil.csv')
sample = pd.read_csv(data_path + 'sample_submission.csv')
store = pd.read_csv(data_path + 'stores.csv')
# test = pd.read_csv(data_path + 'test.csv')
# train = pd.read_csv(data_path + 'train.csv')
# train = train[train.date>='2017-01-01']
# train = pd.concat([train,test]).fillna(0)
# train['unit_sales'] = train['unit_sales'].apply(lambda x: np.log1p(x) if x>=0 else 0)
# train.to_hdf(data_path + 'train_2017.hdf', 'w', complib='blosc', complevel=5)
train = pd.read_hdf(data_path + 'train_2016.hdf', 'w')
transaction = pd.read_csv(data_path + 'transactions.csv')
load = 1




# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result

# 日期的加减
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

# 相差的日期数
def diff_of_days(day1, day2):
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return days

# 相差的分钟数
def diff_of_minutes(time1,time2):
    minutes = (parse(time1) - parse(time2)).total_seconds()//60
    return abs(minutes)

# 分组排序
def rank(data, feat1, feat2, ascending=True):
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 对于新出现的id随机填充0 1
def random_fill(data):
    for i in range(16):
        n_null = data['onpromotion{}'.format(i)].isnull().sum()
        n_prom = (data['onpromotion{}'.format(i)]==True).sum()
        n_noprom = len(data)-n_null-n_prom
        n1 = int(n_prom/(n_prom+n_noprom)*n_null*0.2)
        l01 = [1]*n1 + [0]*(n_null-n1)
        np.random.seed(66)
        np.random.shuffle(l01)
        data.loc[data['onpromotion{}'.format(i)].isnull(),'onpromotion{}'.format(i)] = l01
    data = data.astype(int)
    return data

# 获取标签
def get_label(end_date):
    result_path = cache_path + 'label_{}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        label = pd.read_hdf(result_path, 'w')
    else:
        label_end_date = date_add_days(end_date, 16)
        label = train[(train['date'] < label_end_date) & (train['date'] >= end_date)]
        label = label.set_index(['store_nbr','item_nbr','date'])['unit_sales'].unstack().fillna(0)
        label.columns = [diff_of_days(f,end_date) for f in label.columns]
        index = train[(train['date'] < end_date) & (train['date'] >= '2017-01-01')]
        index = index[['store_nbr', 'item_nbr']].drop_duplicates()
        label = index.merge(label.reset_index(),on=['store_nbr', 'item_nbr'],
                            how='left').set_index(['store_nbr', 'item_nbr']).fillna(0)
        label.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return label


###################################################
#..................... 构造特征 ....................
###################################################
# 前一周每天的值
def get_lastdays_of_st(label, end_date,n_day):
    result_path = cache_path + 'lastdays_of_st{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['last_{}day'.format(diff_of_days(end_date,f)) for f in train_temp.columns]
        result = train_temp.reindex(label.index).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前n天的和
def get_sum_of_store_item(label, end_date,n_day):
    result_path = cache_path + 'get_sum_of_store_item_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date,  -n_day)
        train_temp = train[(train.date<end_date) & (train.date>=start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr','item_nbr','date'])['unit_sales'].unstack().fillna(0)
        names = train_temp.columns
        train_temp['sum_of_store_item{}'.format(n_day)] = train_temp[names].sum(axis=1)
        train_temp['median_of_store_item{}'.format(n_day)] = train_temp[names].median(axis=1)
        train_temp['std_of_store_item{}'.format(n_day)] = train_temp[names].std(axis=1)
        train_temp['skew_of_store_item{}'.format(n_day)] = train_temp[names].skew(axis=1)
        result = train_temp[[f for f in train_temp.columns if f not in names]].copy()
        result = result.reindex(label.index).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前一周每天的值
def get_lastdays_of_prom(label, end_date,n_day):
    result_path = cache_path + 'lastdays_of_prom{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['onpromotion'].unstack().fillna(0)
        train_temp.columns = ['last_{}day_prom'.format(diff_of_days(end_date,f)) for f in train_temp.columns]
        result = train_temp.reindex(label.index).fillna(0).astype(int)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前7天是否促销
def get_sum_of_prom(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_prom_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['onpromotion'].unstack().fillna(0)
        train_temp['sum_of_prom{}'.format(n_day)] = train_temp.sum(axis=1)
        result = train_temp.reindex(label.index).fillna(0)[['sum_of_prom{}'.format(n_day)]]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# # 前n天的和
# def get_promo_of_store_item(label, end_date,n):
#     result_path = cache_path + 'get_promo_of_store_item_{0}_{1}.hdf'.format(end_date, n)
#     if os.path.exists(result_path) & load:
#         result = pd.read_hdf(result_path, 'w')
#     else:
#         now_date = date_add_days(end_date, n)
#         train_temp = train[(train.date<=now_date) & (train.onpromotion==True)].copy()
#         train_temp.set_index(['store_nbr','item_nbr'],drop=False,inplace=True)
#         train_temp.sort_values('date',ascending=True,inplace=True)
#         last_prom = train_temp.drop_duplicates(['store_nbr','item_nbr'],keep='last')
#         last_prom['last_prom'] = last_prom['date'].apply(lambda x:diff_of_days(now_date,x))
#         train_temp = train[(train.date <= now_date) & (train.onpromotion == False)].copy()
#         train_temp.set_index(['store_nbr', 'item_nbr'], drop=False, inplace=True)
#         train_temp.sort_values('date', ascending=True, inplace=True)
#         last_noprom = train_temp.drop_duplicates(['store_nbr', 'item_nbr'], keep='last')
#         last_noprom['last_noprom'] = last_noprom['date'].apply(lambda x: diff_of_days(now_date, x))
#         result = pd.concat([last_prom,last_noprom],axis=1)[['last_prom','last_noprom']]
#         result = result.reindex(label.index).fillna(-1)
#         result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
#     return result

# 是否促销
def get_promo_of_store_item(label, end_date):
    result_path = cache_path + 'get_promo_of_store_item_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        now_date = date_add_days(end_date, 16)
        train_temp = train[(train.date < now_date) & (train.date >= end_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr','date'])['onpromotion'].unstack()
        train_temp.columns = ['onpromotion{}'.format(diff_of_days(f,end_date)) for f in train_temp.columns]
        result = train_temp.reindex(label.index)
        result = random_fill(result)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 二次处理特征
def second_feat(result):
    return result

# 制作训练集
def make_feats(end_date):
    t0 = time.time()
    print('数据key为：{}'.format(end_date))
    result_path = cache_path + 'train_set_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & 0:
        result = pd.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    else:
        print('添加label')
        label = get_label(end_date)

        print('开始构造特征...')
        result = []
        result.append(get_lastdays_of_st(label, end_date,30))        # 前一周每天的值
        # result.append(get_sum_of_store_item(label, end_date, 1))        # 前1天的和
        # result.append(get_sum_of_store_item(label, end_date, 3))        # 前3天的和
        result.append(get_sum_of_store_item(label, end_date, 7))        # 前7天的和
        result.append(get_sum_of_store_item(label, end_date, 14))       # 前14天的和
        result.append(get_sum_of_store_item(label, end_date, 21))       # 前21天的和
        result.append(get_sum_of_store_item(label, end_date, 28))       # 前28天的和
        result.append(get_sum_of_store_item(label, end_date, 42))       # 前42天的和
        result.append(get_sum_of_store_item(label, end_date, 70))       # 前70天的和
        # result.append(get_sum_of_store_item(label, end_date, 98))       # 前98天的和
        result.append(get_sum_of_store_item(label, end_date, 140))      # 前140天的和
        result.append(get_lastdays_of_prom(label, end_date, 7))      # 前7天是否促销
        result.append(get_sum_of_prom(label, end_date, 14))      # 前14天促销次数
        result.append(get_sum_of_prom(label, end_date, 28))      # 前28天促销次数
        result.append(get_sum_of_prom(label, end_date, 140))      # 前140天促销次数
        # result.append(get_sum_of_week(label, end_date, 140))    #获取前一个月的week和

        #上次购买时间
        result.append(get_promo_of_store_item(label, end_date))      # 上次促销开始的时间和结束时间

        result.append(label)


        print('开始合并特征...')
        result = concat(result).reindex()

        result = second_feat(result)

        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result






















