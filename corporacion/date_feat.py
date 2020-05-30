import os
import gc
import time
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 日期的加减
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

cache_path = 'F:/corporacion_date_cache/'
data_path = 'C:/Users/csw/Desktop/python/Corporacion/data/'

holiday = pd.read_csv(data_path + 'holidays_events.csv')
item = pd.read_csv(data_path + 'items.csv')
oil = pd.read_csv(data_path + 'oil.csv')
sample = pd.read_csv(data_path + 'sample_submission.csv')
store = pd.read_csv(data_path + 'stores.csv')
test = pd.read_csv(data_path + 'test.csv')
train = pd.read_csv(data_path + 'train.csv')
# ########################数据预处理##############################
# print('用51商店的值填充52商店2017年4月20日之前的数据')
# train_sub = train[(train.date<'2017-04-20') & (train.store_nbr==51)].copy()
# train_sub['store_nbr'] = 52
# train = pd.concat([train,train_sub])
# print('对没有销量的日期商店用前一周的值填充')
# date_store_sum = train.groupby(['date','store_nbr'])['unit_sales'].sum().unstack()
# dates = pd.date_range('2015-12-01','2017-08-15')
# dates = [str(d)[:10] for d in dates]
# date_store_sum = date_store_sum.reindex(dates).fillna(0)
# train = train[train.date>='2016-01-01'].copy()
# result = pd.DataFrame()
# for s in tqdm(date_store_sum.columns):
#     train_sub = train[train.store_nbr==s].copy()
#     for d in reversed(date_store_sum.index):
#         if date_store_sum.loc[d,s]==0:
#             train_sub2 = train_sub[train_sub.date==date_add_days(d,7)].copy()
#             train_sub2['date'] = d
#             train_sub = pd.concat([train_sub,train_sub2])
#     result = result.append(train_sub)
#     gc.collect()
# train = result[result.date>='2016-01-01'].copy()
# train.loc[(train.onpromotion == True), 'unit_sales'] *= 0.66
# train = pd.concat([train,test]).fillna(0)
# train['unit_sales'] = train['unit_sales'].apply(lambda x: np.log1p(x) if x>=0 else 0)
# train.to_hdf(data_path + 'train_20163.hdf', 'w', complib='blosc', complevel=5)
train = pd.read_hdf(data_path + 'train_20163.hdf', 'w')
train_label = pd.read_hdf(data_path + 'train_2016.hdf', 'w')
transaction = pd.read_csv(data_path + 'transactions.csv')
load = 1

lbl = LabelEncoder()
store.city = lbl.fit_transform(store.city)
store.type = lbl.fit_transform(store.type)
store.state = lbl.fit_transform(store.state)
item.family = lbl.fit_transform(item.family)
item.perishable = (item.perishable+4)/4
holiday.type = lbl.fit_transform(holiday.type)
holiday.locale = lbl.fit_transform(holiday.locale)


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
        label = train_label[(train_label['date'] < end_date) & (train_label['date'] > '2016-01-01')]
        label = label.groupby(['item_nbr','date'])['unit_sales'].sum().unstack().fillna(0).stack()
        label = label.to_frame('unit_salse').reset_index()
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
        result = label[['item_nbr']].merge(item,on='item_nbr',how='left')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 时间特征
def get_date_feat(label,end_date):
    result_path = cache_path + 'date_feat_{}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        result = label[['date']].drop_duplicates()
        result['week'] = pd.to_datetime(result['date']).dt.weekday
        result['month'] = pd.to_datetime(result['date']).dt.month
        result['dayofmonth'] = result['date'].str[8:].astype(int)
        result = result.merge(holiday[['date','type','locale']],on='date',how='left').fillna(-1)
        result = left_merge(label, result, on=['date']).fillna(0)
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
        date_feat = get_date_feat(label,end_date)                        # 时间特征

        print('开始合并特征...')
        result = concat([label,base_feat,date_feat])

        result = second_feat(result)

        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result








import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

train_feat = pd.DataFrame()
end_date = '2017-08-16'
data_feat = make_feats(end_date).fillna(-1)
train_feat = data_feat[data_feat['date']<'2016-06-01']
test_feat = data_feat[data_feat['date']>='2016-06-01']

predictors = [f for f in test_feat.columns if f not in ['unit_salse','date']]

def evalerror(pred, df):
    label = df.get_label().values.copy()
    # a = df.perishable
    rmse = mean_squared_error(label,pred)**0.5
    return ('rmse',rmse,False)

print('开始训练...')
params = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 30,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('第1天...')
lgb_train = lgb.Dataset(train_feat[predictors], train_feat['unit_salse'])
lgb_test = lgb.Dataset(test_feat[predictors], test_feat['unit_salse'])

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_test,
                verbose_eval = 100,
                feval = evalerror,
                early_stopping_rounds=100)
pred = gbm.predict(test_feat[predictors])

feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')


































