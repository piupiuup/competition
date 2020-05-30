import os
import time
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
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

lbl = LabelEncoder()
store.city = lbl.fit_transform(store.city)
store.type = lbl.fit_transform(store.type)
store.state = lbl.fit_transform(store.state)
item.family = lbl.fit_transform(item.family)
item.perishable = (item.perishable+4)/4



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

# left merge操作，删除key
def left_merge(data1,data2,on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
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
    columns = data.columns
    for c in columns:
        n_null = data[c].isnull().sum()
        n_prom = (data[c]==True).sum()
        n_noprom = len(data)-n_null-n_prom
        n1 = int(n_prom/(n_prom+n_noprom)*n_null*0.2)
        l01 = [1]*n1 + [0]*(n_null-n1)
        np.random.seed(66)
        np.random.shuffle(l01)
        data.loc[data[c].isnull(),c] = l01
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
        index = train[(train['date'] < end_date) & (train['date'] >= '2016-06-01')]
        index = index[['store_nbr', 'item_nbr']].drop_duplicates().reset_index(drop=True)
        label = index.merge(label.reset_index(),on=['store_nbr', 'item_nbr'], how='left').fillna(0)
        label.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return label


###################################################
#..................... 构造特征 ....................
###################################################
# 基础特征
def get_base_feat(label,end_date):
    result_path = cache_path + 'base_feat_{}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        result = label[['store_nbr','item_nbr']].merge(item,on='item_nbr',how='left')
        result = result.merge(store, on='store_nbr', how='left')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前一周每天的值
def get_lastdays_of_store_item(base_feat, end_date,n_day):
    result_path = cache_path + 'lastdays_of_store_item{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['last_{}day'.format(diff_of_days(end_date,f)) for f in train_temp.columns]
        result = left_merge(base_feat,train_temp,on=['store_nbr','item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前n天的和
def get_sum_of_store_item(base_feat, end_date,n_day):
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
        train_temp['max_of_store_item{}'.format(n_day)] = train_temp[names].max(axis=1)
        result = train_temp[[f for f in train_temp.columns if f not in names]].copy()
        result = left_merge(base_feat,result,on=['store_nbr','item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前一周每天的值
def get_lastdays_of_prom(base_feat, end_date,n_day):
    result_path = cache_path + 'lastdays_of_prom{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['onpromotion'].unstack().fillna(0)
        train_temp.columns = ['last_{}day_prom'.format(diff_of_days(end_date,f)) for f in train_temp.columns]
        result = left_merge(base_feat, train_temp, on=['store_nbr', 'item_nbr']).fillna(0).astype(int)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 前7天是否促销
def get_sum_of_prom(base_feat, end_date, n_day):
    result_path = cache_path + 'get_sum_of_prom_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        sum_of_prom = train_temp.groupby(['store_nbr', 'item_nbr'])['onpromotion'].agg({'sum_of_prom{}'.format(n_day):'sum'})
        result = left_merge(base_feat, sum_of_prom, on=['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取前一个月的week和
def get_sum_of_week(base_feat, end_date, n_day):
    result_path = cache_path + 'sum_of_week_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp['weekofday'] = pd.to_datetime(train_temp['date']).dt.dayofweek
        result = train_temp.groupby(['store_nbr', 'item_nbr', 'weekofday'])['onpromotion'].sum().unstack().fillna(0)
        result = result.add_prefix('sum_store_item_weekday_{0}_'.format(n_day))
        result2 = result.divide(result.sum(axis=1)+0.001,axis=0)
        result2 = result2.add_prefix('rate_')
        result = pd.concat([result,result2],axis=1).reset_index()
        result = left_merge(base_feat, result, on=['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取去年同期的数据销量数据
def get_lastyear_of_store_item(base_feat, end_date):
    result_path = cache_path + 'lastyear_of_store_item_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -364)
        end_date2 = date_add_days(end_date, -348)
        train_temp = train[(train.date < end_date2) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['lastyear_{}day'.format(diff_of_days(end_date, f)) for f in train_temp.columns]
        result = left_merge(base_feat, train_temp, on=['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 上次购买时间
def get_lastday_of_store_item(base_feat, end_date):
    result_path = cache_path + 'lastday_of_store_item_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_temp = train[(train.date < end_date)].copy()
        train_temp = train_temp.sort_values('date',ascending=True)
        train_temp = train_temp.drop_duplicates(['store_nbr', 'item_nbr'],keep='last')
        train_temp['lastday_of_store_item'] = train_temp['date'].apply(lambda x:diff_of_days(end_date, x))
        result = left_merge(base_feat, train_temp, on=['store_nbr', 'item_nbr']).fillna(-1)[['lastday_of_store_item']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# # 前n天的和
# def get_promo_of_store_item(base_feat, end_date,n):
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
#         result = result.reindex(base_feat.index).fillna(-1)
#         result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
#     return result

# 是否促销
def get_promo_of_store_item(base_feat, end_date):
    result_path = cache_path + 'get_promo_of_store_item_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        now_date = date_add_days(end_date, 16)
        train_temp = train[(train.date < now_date) & (train.date >= end_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr','date'])['onpromotion'].unstack()
        train_temp.columns = ['onpromotion{}'.format(diff_of_days(f,end_date)) for f in train_temp.columns]
        result = left_merge(base_feat, train_temp, on=['store_nbr', 'item_nbr'])
        result = random_fill(result)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



#############################item特征################################
# 前一周每天的值
def get_lastdays_of_item(base_feat, end_date,n_day):
    result_path = cache_path + 'lastdays_of_item{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['item_nbr', 'date'], as_index=False)['unit_sales'].sum()
        train_temp = train_temp.set_index(['item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['last_{}day_of_item'.format(diff_of_days(end_date,f)) for f in train_temp.columns]
        result = left_merge(base_feat,train_temp,on=['item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# item 前7天的和
def get_sum_of_item(base_feat, end_date, n_day):
    result_path = cache_path + 'get_sum_of_store_item_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['item_nbr', 'date'],as_index=False)['unit_sales'].sum()
        train_temp = train_temp.set_index(['item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        names = train_temp.columns
        train_temp['sum_of_item{}'.format(n_day)] = train_temp[names].sum(axis=1)
        train_temp['median_of_item{}'.format(n_day)] = train_temp[names].median(axis=1)
        result = train_temp[[f for f in train_temp.columns if f not in names]].copy()
        result = left_merge(base_feat, result, on=['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取前一个月的week和
def get_sum_of_week_item(base_feat, end_date, n_day):
    result_path = cache_path + 'sum_of_week_item_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp['weekofday'] = pd.to_datetime(train_temp['date']).dt.dayofweek
        result = train_temp.groupby(['item_nbr', 'weekofday'])['onpromotion'].sum().unstack().fillna(0)
        result = result.add_prefix('sum_item_weekday_{0}_'.format(n_day))
        result2 = result.divide(result.sum(axis=1)+0.001,axis=0)
        result2 = result2.add_prefix('rate_')
        result = pd.concat([result,result2],axis=1)
        result = left_merge(base_feat, result, on=['item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 获取去年同期的数据销量数据
def get_lastyear_of_item(base_feat, end_date):
    result_path = cache_path + 'lastyear_of_item_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -364)
        end_date2 = date_add_days(end_date, -348)
        train_temp = train[(train.date < end_date2) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['item_nbr', 'date'], as_index=False)['unit_sales'].sum()
        train_temp = train_temp.set_index(['item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['lastyear_item_{}day'.format(diff_of_days(end_date, f)) for f in train_temp.columns]
        result = left_merge(base_feat, train_temp, on=['item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


########################store特征##########################
# 获取去年同期的数据销量数据
def get_lastyear_of_store(base_feat, end_date):
    result_path = cache_path + 'lastyear_of_store_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -364)
        end_date2 = date_add_days(end_date, -348)
        train_temp = train[(train.date < end_date2) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['item_nbr', 'date'], as_index=False)['unit_sales'].sum()
        train_temp = train_temp.set_index(['item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['lastyear_item_{}day'.format(diff_of_days(end_date, f)) for f in train_temp.columns]
        result = left_merge(base_feat, train_temp, on=['item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

#######################store特征########################
# 前一周每天的值
def get_lastdays_of_store(base_feat, end_date,n_day):
    result_path = cache_path + 'lastdays_of_store{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['store_nbr', 'date'], as_index=False)['unit_sales'].sum()
        train_temp = train_temp.set_index(['store_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['last_{}day_of_store'.format(diff_of_days(end_date,f)) for f in train_temp.columns]
        result = left_merge(base_feat,train_temp,on=['store_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取前一个月的week和
def get_sum_of_week_store(base_feat, end_date, n_day):
    result_path = cache_path + 'sum_of_week_store_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp['weekofday'] = pd.to_datetime(train_temp['date']).dt.dayofweek
        result = train_temp.groupby(['store_nbr', 'weekofday'])['onpromotion'].sum().unstack().fillna(0)
        result = result.add_prefix('sum_store_weekday_{0}_'.format(n_day))
        result2 = result.divide(result.sum(axis=1)+0.001,axis=0)
        result2 = result2.add_prefix('rate_')
        result = pd.concat([result,result2],axis=1)
        result = left_merge(base_feat, result, on=['store_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取去年同期的数据销量数据
def get_lastyear_of_store(base_feat, end_date):
    result_path = cache_path + 'lastyear_of_store_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -364)
        end_date2 = date_add_days(end_date, -348)
        train_temp = train[(train.date < end_date2) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['store_nbr', 'date'], as_index=False)['unit_sales'].sum()
        train_temp = train_temp.set_index(['store_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['lastyear_store_{}day'.format(diff_of_days(end_date, f)) for f in train_temp.columns]
        result = left_merge(base_feat, train_temp, on=['store_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 商场的销量
def get_target_store(base_feat, end_date):
    result_path = cache_path + 'target_store_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -7)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['store_nbr'], as_index=False)['unit_sales'].agg({'target_store':'sum'})
        result = left_merge(base_feat, train_temp, on=['store_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



########################store×class######################
# 前n天的和
def get_sum_of_store_class(base_feat, end_date,n_day):
    result_path = cache_path + 'get_sum_of_store_class_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date,  -n_day)
        train_temp = train[(train.date<end_date) & (train.date>=start_date)].copy()
        train_temp = train_temp.merge(item[['item_nbr','class']],on='item_nbr',how='left')
        train_temp = train_temp.groupby(['store_nbr','class','date'])['unit_sales'].sum().unstack().fillna(0)
        names = train_temp.columns
        train_temp['sum_of_store_class{}'.format(n_day)] = train_temp[names].sum(axis=1)
        train_temp['median_of_store_class{}'.format(n_day)] = train_temp[names].median(axis=1)
        result = train_temp[[f for f in train_temp.columns if f not in names]].copy()
        result = left_merge(base_feat,result,on=['store_nbr','class']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

########################store×family######################
# 前n天的和
def get_sum_of_store_family(base_feat, end_date,n_day):
    result_path = cache_path + 'get_sum_of_store_family_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date,  -n_day)
        train_temp = train[(train.date<end_date) & (train.date>=start_date)].copy()
        train_temp = train_temp.merge(item[['item_nbr','family']],on='item_nbr',how='left')
        train_temp = train_temp.groupby(['store_nbr','family','date'])['unit_sales'].sum().unstack().fillna(0)
        names = train_temp.columns
        train_temp['sum_of_store_family{}'.format(n_day)] = train_temp[names].sum(axis=1)
        result = train_temp[[f for f in train_temp.columns if f not in names]].copy()
        result = left_merge(base_feat,result,on=['store_nbr','family']).fillna(0)
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
        base_feat = get_base_feat(label,end_date)                        # 基础特征

        print('开始构造特征...')
        result = [base_feat]
        result.append(get_lastdays_of_store_item(base_feat, end_date,30))        # 前一周每天的值
        result.append(get_sum_of_store_item(base_feat, end_date, 7))        # 前7天的和
        result.append(get_sum_of_store_item(base_feat, end_date, 14))       # 前14天的和
        result.append(get_sum_of_store_item(base_feat, end_date, 21))       # 前21天的和
        result.append(get_sum_of_store_item(base_feat, end_date, 28))       # 前28天的和
        result.append(get_sum_of_store_item(base_feat, end_date, 42))       # 前42天的和
        result.append(get_sum_of_store_item(base_feat, end_date, 70))       # 前70天的和
        result.append(get_sum_of_store_item(base_feat, end_date, 140))      # 前140天的和
        result.append(get_lastdays_of_prom(base_feat, end_date, 7))      # 前7天是否促销
        result.append(get_sum_of_prom(base_feat, end_date, 14))         # 前14天促销次数
        result.append(get_sum_of_prom(base_feat, end_date, 28))         # 前28天促销次数
        result.append(get_sum_of_prom(base_feat, end_date, 140))        # 前140天促销次数
        result.append(get_sum_of_week(base_feat, end_date, 28))         # 获取前一个月的week和
        result.append(get_sum_of_week(base_feat, end_date, 140))        # 获取前一个月的week和
        result.append(get_sum_of_week(base_feat, end_date, 490))      # 获取前一个月的week和
        result.append(get_lastyear_of_store_item(base_feat, end_date))  # 获取去年同期的数据销量数据
        result.append(get_lastday_of_store_item(base_feat, end_date))   #上次购买时间
        result.append(get_promo_of_store_item(base_feat, end_date))     # 是否促销

        result.append(get_lastdays_of_item(base_feat, end_date,30))     # item 前7天的值
        result.append(get_sum_of_item(base_feat, end_date, 7))          # item 前7天的和
        result.append(get_sum_of_item(base_feat, end_date, 28))         # item 前7天的和
        result.append(get_sum_of_item(base_feat, end_date, 70))         # item 前7天的和
        result.append(get_sum_of_item(base_feat, end_date, 140))        # item 前7天的和
        # result.append(get_sum_of_week_item(base_feat, end_date, 140))   # 获取前一个月的week和
        result.append(get_lastyear_of_item(base_feat, end_date))        # 获取去年同期的数据销量数据

        # result.append(get_lastdays_of_store(base_feat, end_date, 7))    # store 前7天的值
        # result.append(get_sum_of_week_store(base_feat, end_date, 140))  # 获取前一个月的week和
        # result.append(get_lastyear_of_store(base_feat, end_date))       # 获取去年同期的数据销量数据
        result.append(get_target_store(base_feat, end_date))            # 商场的销量

        # 商场 品类特征
        # result.append(get_sum_of_store_class(base_feat, end_date, 7))  # store×class 前7天的

        # 商店 大类特征
        # result.append(get_sum_of_store_family(base_feat, end_date, 7))  # store×class 前7天的

        ########################日期趋势信息###########################




        result.append(label)


        print('开始合并特征...')
        result = concat(result)

        result = second_feat(result)

        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result




#
# import datetime
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
#
# train_feat = pd.DataFrame()
# end_date = '2017-07-26'
# for j in range(6):
#     train_feat_sub = make_feats(date_add_days(end_date, j * (-7))).fillna(-1)
#     train_feat = pd.concat([train_feat, train_feat_sub])
# # eval_feat = make_feats(date_add_days(end_date, 14)).fillna(-1)
# test_feat = make_feats(date_add_days(end_date, 21)).fillna(-1)
# predictors = [f for f in test_feat.columns if f not in (list(range(16)))]
#
# def evalerror(pred, df):
#     label = df.get_label().values.copy()
#     rmse = mean_squared_error(label,pred)**0.5
#     return ('rmse',rmse,False)
#
# print('开始训练...')
# params = {
#     'learning_rate': 0.02,
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     # 'metric': 'mse',
#     # 'sub_feature': 0.7,
#     'num_leaves': 30,
#     'colsample_bytree': 0.7,
#     'feature_fraction': 0.7,
#     'min_data': 100,
#     'min_hessian': 1,
#     'verbose': -1,
# }
# submission = pd.DataFrame()
# for i in range(16):
#
#     print('第{}轮循环'.format(i))
#     date = date_add_days('2017-08-16',i)
#     lgb_train = lgb.Dataset(train_feat[predictors], train_feat[i], weight=train_feat["perishable"])
#     # lgb_eval = lgb.Dataset(eval_feat[predictors], eval_feat[i], weight=eval_feat["perishable"])
#
#     gbm = lgb.train(params,lgb_train,2000)
#     # gbm = lgb.train(params,
#     #                 lgb_eval,
#     #                 num_boost_round=3000,
#     #                 valid_sets=lgb_train,
#     #                 early_stopping_rounds=200,
#     #                 feval=evalerror,
#     #                 verbose_eval=200,
#     # )
#     pred = gbm.predict(test_feat[predictors])
#     submission = pd.concat([submission,pd.DataFrame({'store_nbr':test_feat['store_nbr'].values,
#                                                      'item_nbr':test_feat['item_nbr'].values,
#                                                      'date':date,
#                                                      'unit_sales':np.exp(pred)-1})])
# test = pd.read_csv(data_path + 'test.csv')
# submission = test.merge(submission,on=['store_nbr','item_nbr','date'],how='left')
# submission['unit_sales'] = submission['unit_sales'].apply(lambda x: x if x>0 else 0)
# submission[['id','unit_sales']].to_csv(r'C:\Users\csw\Desktop\python\Corporacion\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                   index=False, float_format='%.4f')
#













