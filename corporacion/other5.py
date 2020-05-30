# -*- coding: utf-8 -*
from __future__ import division
import sys

import os
import time
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from datetime import date, timedelta, datetime
from sklearn.metrics import mean_squared_error

cache_path = '../cache/'
data_path = '../input/'

holiday_data = pd.read_csv(data_path + 'holidays_events.csv')
item = pd.read_csv(data_path + 'items.csv')
oil = pd.read_csv(data_path + 'oil.csv')
sample = pd.read_csv(data_path + 'sample_submission.csv')
store = pd.read_csv(data_path + 'stores.csv')
test = pd.read_csv(data_path + 'test.csv')
# train = pd.read_csv(data_path + 'train.csv')
# train = train[train.date>='2016-01-01']
# train = pd.concat([train,test]).fillna(0)
# train['unit_sales'] = train['unit_sales'].apply(lambda x: np.log1p(x) if x>=0 else 0)
# train.to_hdf(data_path + 'train_2016.hdf', 'w', complib='blosc', complevel=5)

train = pd.read_hdf(data_path + 'train_2016.hdf', 'w')
transaction = pd.read_csv(data_path + 'transactions.csv')
load = 1

fuck_oil = list(set(train['date']))
fuck_oil = pd.DataFrame(sorted(fuck_oil), columns=['date'])
oil = fuck_oil.merge(oil, on=['date'], how='left')

cat_feat = ['city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']


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
def diff_of_minutes(time1, time2):
    minutes = (parse(time1) - parse(time2)).total_seconds() // 60
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
        n_prom = (data['onpromotion{}'.format(i)] == True).sum()
        n_noprom = len(data) - n_null - n_prom
        n1 = int(n_prom / (n_prom + n_noprom) * n_null * 0.2)
        l01 = [1] * n1 + [0] * (n_null - n1)
        np.random.seed(66)
        np.random.shuffle(l01)
        data.loc[data['onpromotion{}'.format(i)].isnull(), 'onpromotion{}'.format(i)] = l01
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
        label = label.set_index(['store_nbr', 'item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        label.columns = [diff_of_days(f, end_date) for f in label.columns]
        index = train[(train['date'] < end_date) & (train['date'] >= '2017-01-01')]
        index = index[['store_nbr', 'item_nbr']].drop_duplicates()
        label = index.merge(label.reset_index(), on=['store_nbr', 'item_nbr'],
                            how='left').set_index(['store_nbr', 'item_nbr']).fillna(0)
        label.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return label


# 化成label需要的格式
def to_label(train_temp, label):
    label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
    result = label_temp.merge(train_temp, on=['store_nbr', 'item_nbr'], how='left').fillna(0)
    result = result.set_index(['store_nbr', 'item_nbr']).fillna(0)
    return result


###################################################
# ..................... 构造特征 ....................
###################################################
# 前一周每天的值
def get_lastdays_of_st(label, end_date, n_day):
    result_path = cache_path + 'lastdays_of_st{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        train_temp.columns = ['last_{}day'.format(diff_of_days(end_date, f)) for f in train_temp.columns]
        result = train_temp.reindex(label.index).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def week0(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 0]
    del x['dayofweek']
    return x


def week1(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 1]
    del x['dayofweek']
    return x


def week2(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 2]
    del x['dayofweek']
    return x


def week3(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 3]
    del x['dayofweek']
    return x


def week4(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 4]
    del x['dayofweek']
    return x


def week5(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 5]
    del x['dayofweek']
    return x


def week6(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x = x[x['dayofweek'] == 6]
    del x['dayofweek']
    return x


def weekend_yes(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x['dayofweek'] = x['dayofweek'].map(lambda x: 0 if x > 4 else 1)
    x = x[x['dayofweek'] == 0]
    del x['dayofweek']
    return x


def weekend_no(x):
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x['dayofweek'] = x['dayofweek'].map(lambda x: 0 if x > 4 else 1)
    x = x[x['dayofweek'] == 1]
    del x['dayofweek']
    return x


def onpromotion_yes(x):
    x = x[x['onpromotion'] == True]
    return x


def onpromotion_no(x):
    x = x[x['onpromotion'] == False]
    return x


def holiday_yes(x):
    holiday_data = pd.read_csv(data_path + 'holidays_events.csv')
    holiday_data = holiday_data[holiday_data['transferred'] == False]
    holiday_list = list(set(holiday_data['date'].values))
    x['holiday'] = x['date'].map(lambda k: 1 if k in holiday_list else 0)
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x['dayofweek'] = x['dayofweek'].map(lambda x: 0 if x > 4 else 1)
    x = x[(x['holiday'] == 1) | (x['dayofweek'] == 0)]
    del x['holiday']
    del x['dayofweek']
    return x


def holiday_no(x):
    holiday_data = pd.read_csv(data_path + 'holidays_events.csv')
    holiday_data = holiday_data[holiday_data['transferred'] == False]
    holiday_list = list(set(holiday_data['date'].values))
    x['holiday'] = x['date'].map(lambda k: 1 if k in holiday_list else 0)
    x['dayofweek'] = pd.DatetimeIndex(pd.to_datetime(x.date)).dayofweek
    x['dayofweek'] = x['dayofweek'].map(lambda x: 0 if x > 4 else 1)
    x = x[(x['holiday'] == 0) & (x['dayofweek'] == 1)]
    del x['holiday']
    del x['dayofweek']
    return x


# 前n天同商品同商铺的和
def get_sum_of_store_item(label, end_date, n_day, value_list=None, choose=lambda x: x, choose_name=''):
    if value_list == None:
        if choose_name == '':
            result_path = cache_path + 'get_sum_of_store_item_{0}_{1}.hdf'.format(end_date, n_day)
            result_name = str(n_day)
        else:
            result_path = cache_path + 'get_sum_of_store_item_{0}_{1}_{2}.hdf'.format(end_date, n_day, choose_name)
            result_name = '{0}_{1}'.format(n_day, choose_name)
    else:
        if choose_name == '':
            result_path = cache_path + 'get_sum_of_store_item_{0}_{1}_{2}.hdf'.format(end_date, n_day, sum(value_list))
            result_name = '{0}_{1}'.format(n_day, sum(value_list))
        else:
            result_path = cache_path + 'get_sum_of_store_item_{0}_{1}_{2}_{3}.hdf'.format(end_date, n_day,
                                                                                          sum(value_list), choose_name)
            result_name = '{0}_{1}_{2}'.format(n_day, sum(value_list), choose_name)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = choose(train_temp)
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['unit_sales'].unstack().fillna(0)
        names = train_temp.columns
        if value_list == None:
            train_temp['mean_of_store_item{}'.format(result_name)] = train_temp[names].mean(axis=1)
            train_temp['median_of_store_item{}'.format(result_name)] = train_temp[names].median(axis=1)
            train_temp['std_of_store_item{}'.format(result_name)] = train_temp[names].std(axis=1)
            train_temp['skew_of_store_item{}'.format(result_name)] = train_temp[names].skew(axis=1)
            train_temp['max_of_store_item{}'.format(result_name)] = train_temp[names].max(axis=1)
            train_temp['min_of_store_item{}'.format(result_name)] = train_temp[names].min(axis=1)
        else:
            for k, each in enumerate(names):
                train_temp[each] = train_temp[each] * value_list[k]
            train_temp[result_name] = train_temp[names].sum(axis=1)
        result = train_temp[[f for f in train_temp.columns if f not in names]].copy()
        result = result.reindex(label.index).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 前n天同商品的和
def get_sum_of_item(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_item_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:  # 重命名下
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.groupby(['item_nbr', 'date'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['item_nbr'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_item' + str(n_day): 'mean', 'median_of_item' + str(n_day): 'median',
            'std_of_item' + str(n_day): 'std', 'skew_of_item' + str(n_day): 'skew',
            'max_of_item' + str(n_day): 'max', 'min_of_item' + str(n_day): 'min', })
        label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
        result = label_temp.merge(train_temp, on=['item_nbr'], how='left').fillna(0)
        result = result.set_index(['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_item_city(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_item_city_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.merge(store[['store_nbr', 'city']],
                                      on=['store_nbr'], how='left')
        train_temp = train_temp.groupby(['item_nbr', 'date', 'city'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['item_nbr', 'city'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_item_city' + str(n_day): 'mean', 'median_of_item_city' + str(n_day): 'median',
            'std_of_item_city' + str(n_day): 'std', 'skew_of_item_city' + str(n_day): 'skew',
            'max_of_item_city' + str(n_day): 'max', 'min_of_item_city' + str(n_day): 'min', })
        train_temp = train_temp.merge(store[['store_nbr', 'city']],
                                      on=['city'], how='left')
        del train_temp['city']
        result = to_label(train_temp, label)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_item_state(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_item_state_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.merge(store[['store_nbr', 'state']],
                                      on=['store_nbr'], how='left')
        train_temp = train_temp.groupby(['item_nbr', 'date', 'state'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['item_nbr', 'state'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_item_state' + str(n_day): 'mean', 'median_of_item_state' + str(n_day): 'median',
            'std_of_item_state' + str(n_day): 'std', 'skew_of_item_state' + str(n_day): 'skew',
            'max_of_item_state' + str(n_day): 'max', 'min_of_item_state' + str(n_day): 'min', })
        train_temp = train_temp.merge(store[['store_nbr', 'state']],
                                      on=['state'], how='left')
        del train_temp['state']
        result = to_label(train_temp, label)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_item_type(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_item_type_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.merge(store[['store_nbr', 'type']],
                                      on=['store_nbr'], how='left')
        train_temp = train_temp.groupby(['item_nbr', 'date', 'type'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['item_nbr', 'type'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_item_type' + str(n_day): 'mean', 'median_of_item_type' + str(n_day): 'median',
            'std_of_item_type' + str(n_day): 'std', 'skew_of_item_type' + str(n_day): 'skew',
            'max_of_item_type' + str(n_day): 'max', 'min_of_item_type' + str(n_day): 'min', })
        train_temp = train_temp.merge(store[['store_nbr', 'type']],
                                      on=['type'], how='left')
        del train_temp['type']
        result = to_label(train_temp, label)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_item_cluster(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_item_cluster_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.merge(store[['store_nbr', 'cluster']],
                                      on=['store_nbr'], how='left')
        train_temp = train_temp.groupby(['item_nbr', 'date', 'cluster'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['item_nbr', 'cluster'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_item_cluster' + str(n_day): 'mean', 'median_of_item_cluster' + str(n_day): 'median',
            'std_of_item_cluster' + str(n_day): 'std', 'skew_of_item_cluster' + str(n_day): 'skew',
            'max_of_item_cluster' + str(n_day): 'max', 'min_of_item_cluster' + str(n_day): 'min', })
        train_temp = train_temp.merge(store[['store_nbr', 'cluster']],
                                      on=['cluster'], how='left')
        del train_temp['cluster']
        result = to_label(train_temp, label)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_store_family(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_store_family_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.merge(item[['item_nbr', 'family']],
                                      on=['item_nbr'], how='left')
        train_temp = train_temp.groupby(['store_nbr', 'date', 'family'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['store_nbr', 'family'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_store_family' + str(n_day): 'mean', 'median_of_store_family' + str(n_day): 'median',
            'std_of_store_family' + str(n_day): 'std', 'skew_of_store_family' + str(n_day): 'skew',
            'max_of_store_family' + str(n_day): 'max', 'min_of_store_family' + str(n_day): 'min', })
        train_temp = train_temp.merge(item[['item_nbr', 'family']],
                                      on=['family'], how='left')
        del train_temp['family']
        result = to_label(train_temp, label)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_store_class(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_store_class_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.merge(item[['item_nbr', 'class']],
                                      on=['item_nbr'], how='left')
        train_temp = train_temp.groupby(['store_nbr', 'date', 'class'],
                                        as_index=False)['unit_sales'].agg({'unit_sales': 'sum'})
        train_temp = train_temp.groupby(['store_nbr', 'class'],
                                        as_index=False)['unit_sales'].agg({
            'mean_of_store_class' + str(n_day): 'mean', 'median_of_store_class' + str(n_day): 'median',
            'std_of_store_class' + str(n_day): 'std', 'skew_of_store_class' + str(n_day): 'skew',
            'max_of_store_class' + str(n_day): 'max', 'min_of_store_class' + str(n_day): 'min', })
        train_temp = train_temp.merge(item[['item_nbr', 'class']],
                                      on=['class'], how='left')
        del train_temp['class']
        result = to_label(train_temp, label)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_oil(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_oil_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        oil['dcoilwtico'].interpolate(method='linear', inplace=True)
        start_date = date_add_days(end_date, -n_day)
        oil_temp = oil[(oil.date < end_date) & (oil.date >= start_date)].copy()
        mean = oil_temp['dcoilwtico'].mean()
        max = oil_temp['dcoilwtico'].max()
        min = oil_temp['dcoilwtico'].min()
        std = oil_temp['dcoilwtico'].std()
        skew = oil_temp['dcoilwtico'].skew()
        label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
        label_temp['mean_dcoilwtico_{}'.format(n_day)] = mean
        label_temp['max_dcoilwtico_{}'.format(n_day)] = max
        label_temp['min_dcoilwtico_{}'.format(n_day)] = min
        label_temp['std_dcoilwtico_{}'.format(n_day)] = std
        label_temp['skew_dcoilwtico_{}'.format(n_day)] = skew
        result = label_temp.set_index(['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_oil(label, end_date, n_day):
    result_path = cache_path + 'get_oil{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        oil['dcoilwtico'].interpolate(method='linear', inplace=True)
        start_date = date_add_days(end_date, -n_day)
        oil_temp = oil[(oil.date < end_date) & (oil.date >= start_date)].copy()
        oil_temp = list(oil_temp['dcoilwtico'].values)
        label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
        for i in range(len(oil_temp)):
            label_temp['oil_' + str(i)] = oil_temp[i]
        result = label_temp.set_index(['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_sum_of_trans(label, end_date, n_day):
    result_path = cache_path + 'get_sum_of_trans_{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        transaction_temp = transaction[(transaction.date < end_date) & (transaction.date >= start_date)].copy()
        transaction_temp = transaction_temp.groupby(['store_nbr'], as_index=False)['transactions'].agg({
            'mean_of_trans' + str(n_day): 'mean', 'median_of_trans' + str(n_day): 'median',
            'std_of_trans' + str(n_day): 'std', 'skew_of_trans' + str(n_day): 'skew',
            'max_of_trans' + str(n_day): 'max', 'min_of_trans' + str(n_day): 'min',
        })
        label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
        result = label_temp.merge(transaction_temp, on=['store_nbr'], how='left')
        result = result.set_index(['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_store_info(label):
    result_path = cache_path + 'get_store_info.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
        store_temp = store.copy()
        city_dict = {k: j for j, k in enumerate(set(store_temp['city'].values))}
        state_dict = {k: j for j, k in enumerate(set(store_temp['state'].values))}
        type_dict = {k: j for j, k in enumerate(set(store_temp['type'].values))}
        store_temp['city'] = store_temp['city'].map(city_dict)
        store_temp['state'] = store_temp['state'].map(state_dict)
        store_temp['type'] = store_temp['type'].map(type_dict)
        result = label_temp.merge(store_temp, on=['store_nbr'], how='left').fillna(0)
        result = result.set_index(['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_item_info(label):
    result_path = cache_path + 'get_item_info.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        label_temp = label.reset_index()[['store_nbr', 'item_nbr']]
        item_temp = item.copy()
        family_dict = {k: j for j, k in enumerate(set(item_temp['family'].values))}
        item_temp['family'] = item_temp['family'].map(family_dict)
        result = label_temp.merge(item_temp, on=['item_nbr'], how='left').fillna(0)
        result = result.set_index(['store_nbr', 'item_nbr']).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 前7天是否促销
def get_lastdays_of_prom(label, end_date, n_day):
    result_path = cache_path + 'lastdays_of_prom{0}_{1}.hdf'.format(end_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        start_date = date_add_days(end_date, -n_day)
        train_temp = train[(train.date < end_date) & (train.date >= start_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['onpromotion'].unstack().fillna(0)
        train_temp.columns = ['last_{}day_prom'.format(diff_of_days(end_date, f)) for f in train_temp.columns]
        result = train_temp.reindex(label.index).fillna(0).astype(int)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 前n天促销数
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


# 是否促销
def get_promo_of_store_item(label, end_date):
    result_path = cache_path + 'get_promo_of_store_item_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        now_date = date_add_days(end_date, 16)
        train_temp = train[(train.date < now_date) & (train.date >= end_date)].copy()
        train_temp = train_temp.set_index(['store_nbr', 'item_nbr', 'date'])['onpromotion'].unstack()
        train_temp.columns = ['onpromotion{}'.format(diff_of_days(f, end_date)) for f in train_temp.columns]
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
    print('time key:{}'.format(end_date))
    result_path = cache_path + 'train_set_{0}.hdf'.format(end_date)
    if os.path.exists(result_path) & load:
        result = pd.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    else:
        print('add label')
        label = get_label(end_date)
        print('make feature...')
        result = []

        result.append(get_sum_of_store_item(label, end_date, 140, None, holiday_no, 'holiday_no'))
        result.append(
            get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], holiday_no, 'holiday_no'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, holiday_yes, 'holiday_yes'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], holiday_yes,
                                            'holiday_yes'))

        result.append(get_sum_of_store_item(label, end_date, 140, None, week0, 'week0'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], week0, 'week0'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, week1, 'week1'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], week1, 'week1'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, week2, 'week2'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], week2, 'week2'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, week3, 'week3'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], week3, 'week3'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, week4, 'week4'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], week4, 'week4'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, week5, 'week5'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], week5, 'week5'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, weekend_no, 'weekend_no'))
        result.append(
            get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], weekend_no, 'weekend_no'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, weekend_yes, 'weekend_yes'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], weekend_yes,
                                            'weekend_yes'))

        result.append(get_sum_of_store_item(label, end_date, 140, None, onpromotion_yes, 'onpromotion_yes'))
        result.append(
            get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], onpromotion_yes,
                                  'onpromotion_yes'))
        result.append(get_sum_of_store_item(label, end_date, 140, None, onpromotion_no, 'onpromotion_no'))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)], onpromotion_no,
                                            'onpromotion_no'))

        result.append(get_sum_of_store_item(label, end_date, 140, [i for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i + 140 for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i + 280 for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 2 for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 3 for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 4 for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 5 for i in range(140)]))
        result.append(get_sum_of_store_item(label, end_date, 140, [i ** 6 / 1000 for i in range(140)]))

        result.append(get_sum_of_item_state(label, end_date, 7))  # 前7天同state同商品销量
        result.append(get_sum_of_item_state(label, end_date, 14))  # 前14天同state同商品销量
        result.append(get_sum_of_item_state(label, end_date, 28))  # 前28天同state同商品销量
        result.append(get_sum_of_item_state(label, end_date, 42))  # 前42天同state同商品销量
        result.append(get_sum_of_item_state(label, end_date, 70))  # 前70天同state同商品销量
        result.append(get_sum_of_item_state(label, end_date, 140))  # 前140天同state同商品销量

        result.append(get_sum_of_item_type(label, end_date, 7))  # 前7天同type同商品销量
        result.append(get_sum_of_item_type(label, end_date, 14))  # 前14天同type同商品销量
        result.append(get_sum_of_item_type(label, end_date, 28))  # 前28天同type同商品销量
        result.append(get_sum_of_item_type(label, end_date, 42))  # 前42天同type同商品销量
        result.append(get_sum_of_item_type(label, end_date, 70))  # 前70天同type同商品销量
        result.append(get_sum_of_item_type(label, end_date, 140))  # 前140天同type同商品销量

        result.append(get_sum_of_item_cluster(label, end_date, 7))  # 前7天同cluster同商品销量
        result.append(get_sum_of_item_cluster(label, end_date, 14))  # 前14天同cluster同商品销量
        result.append(get_sum_of_item_cluster(label, end_date, 28))  # 前28天同cluster同商品销量
        result.append(get_sum_of_item_cluster(label, end_date, 42))  # 前42天同cluster同商品销量
        result.append(get_sum_of_item_cluster(label, end_date, 70))  # 前70天同cluster同商品销量
        result.append(get_sum_of_item_cluster(label, end_date, 140))  # 前140天同cluster同商品销量

        result.append(get_sum_of_store_class(label, end_date, 7))  # 前7天同店铺同class销量
        result.append(get_sum_of_store_class(label, end_date, 14))  # 前14天同店铺同class销量
        result.append(get_sum_of_store_class(label, end_date, 28))  # 前28天同店铺同class销量
        result.append(get_sum_of_store_class(label, end_date, 42))  # 前42天同店铺同class销量
        result.append(get_sum_of_store_class(label, end_date, 70))  # 前70天同店铺同class销量
        result.append(get_sum_of_store_class(label, end_date, 140))  # 前140天同店铺同class销量

        result.append(get_sum_of_trans(label, end_date, 7))  # 前7天trans
        result.append(get_sum_of_trans(label, end_date, 28))  # 前28天trans
        result.append(get_sum_of_trans(label, end_date, 42))  # 前42天trans
        result.append(get_sum_of_trans(label, end_date, 70))  # 前70天trans
        result.append(get_sum_of_trans(label, end_date, 140))  # 前140天trans

        result.append(get_sum_of_oil(label, end_date, 7))  # 前7天油价
        result.append(get_sum_of_oil(label, end_date, 28))  # 前28天油价
        result.append(get_sum_of_oil(label, end_date, 42))  # 前42天油价
        result.append(get_sum_of_oil(label, end_date, 70))  # 前70天油价
        result.append(get_sum_of_oil(label, end_date, 140))  # 前140天油价

        result.append(get_item_info(label))  # 添加商品信息
        result.append(get_store_info(label))  # 添加店铺信息

        result.append(get_sum_of_store_family(label, end_date, 7))  # 前7天同店铺同family销量
        result.append(get_sum_of_store_family(label, end_date, 14))  # 前14天同店铺同family销量
        result.append(get_sum_of_store_family(label, end_date, 28))  # 前28天同店铺同family销量
        result.append(get_sum_of_store_family(label, end_date, 42))  # 前42天同店铺同family销量
        result.append(get_sum_of_store_family(label, end_date, 70))  # 前70天同店铺同family销量
        result.append(get_sum_of_store_family(label, end_date, 140))  # 前140天同店铺同family销量

        result.append(get_sum_of_item_city(label, end_date, 7))  # 前7天同城市同商品销量
        result.append(get_sum_of_item_city(label, end_date, 14))  # 前14天同城市同商品销量
        result.append(get_sum_of_item_city(label, end_date, 21))  # 前21天同城市同商品销量
        result.append(get_sum_of_item_city(label, end_date, 28))  # 前28天同城市同商品销量
        result.append(get_sum_of_item_city(label, end_date, 42))  # 前42天同城市同商品销量
        result.append(get_sum_of_item_city(label, end_date, 70))  # 前70天同城市同商品销量
        result.append(get_sum_of_item_city(label, end_date, 140))  # 前140天同城市同商品销量

        result.append(get_sum_of_item(label, end_date, 7))  # 前7天商品销量
        result.append(get_sum_of_item(label, end_date, 14))  # 前14天商品销量
        result.append(get_sum_of_item(label, end_date, 21))  # 前21天商品销量
        result.append(get_sum_of_item(label, end_date, 28))  # 前28天商品销量
        result.append(get_sum_of_item(label, end_date, 42))  # 前42天商品销量
        result.append(get_sum_of_item(label, end_date, 70))  # 前70天商品销量
        result.append(get_sum_of_item(label, end_date, 140))  # 前140天商品销量

        result.append(get_lastdays_of_st(label, end_date, 30))  # 前一月每天的值
        # result.append(get_sum_of_store_item(label, end_date, 1))        # 前1天的和
        result.append(get_sum_of_store_item(label, end_date, 3))  # 前3天的和
        result.append(get_sum_of_store_item(label, end_date, 7))  # 前7天的和
        result.append(get_sum_of_store_item(label, end_date, 14))  # 前14天的和
        result.append(get_sum_of_store_item(label, end_date, 21))  # 前21天的和
        result.append(get_sum_of_store_item(label, end_date, 28))  # 前28天的和
        result.append(get_sum_of_store_item(label, end_date, 42))  # 前42天的和
        result.append(get_sum_of_store_item(label, end_date, 70))  # 前70天的和
        result.append(get_sum_of_store_item(label, end_date, 98))  # 前98天的和
        result.append(get_sum_of_store_item(label, end_date, 140))  # 前140天的和

        result.append(get_lastdays_of_prom(label, end_date, 16))  # 该时间段内是否促销

        result.append(get_sum_of_prom(label, end_date, 14))  # 前14天促销次数
        result.append(get_sum_of_prom(label, end_date, 28))  # 前28天促销次数
        result.append(get_sum_of_prom(label, end_date, 35))  # 前140天促销次数
        result.append(get_sum_of_prom(label, end_date, 70))  # 前140天促销次数
        result.append(get_sum_of_prom(label, end_date, 280))  # 前140天促销次数
        result.append(get_sum_of_prom(label, end_date, 140))  # 前140天促销次数
        # result.append(get_sum_of_week(label, end_date, 140))    #获取前一个月的week和


        # 上次购买时间
        result.append(get_promo_of_store_item(label, end_date))  # 上次促销开始的时间和结束时间

        result.append(label)

        print('concat feature...')
        result = concat(result).reindex()

        result = second_feat(result)

        # print('存储数据...')
        # result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('shape：{}'.format(result.shape))
    print('used {} second'.format(time.time() - t0))
    return result


if __name__ == '__main__':
    import datetime
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    train_feat = pd.DataFrame()
    end_date = '2017-07-12'
    wight_list = []
    each_j_wight = [1, 1, 1, 1, 1, 1, 0.7, 0.5]
    for j in range(8):
        train_feat_sub = make_feats(date_add_days(date_add_days(end_date, 14), j * (-7))).fillna(-1)
        train_feat = pd.concat([train_feat, train_feat_sub])
        wight_list.extend([each_j_wight[j]] * train_feat_sub.shape[0])
    eval_feat = make_feats(date_add_days(end_date, 14)).fillna(-1)
    test_feat = make_feats(date_add_days(end_date, 35)).fillna(-1)
    test_key = test_feat.reset_index()[['store_nbr', 'item_nbr']]

    predictors = [f for f in test_feat.columns if f not in (['store_nbr', 'item_nbr'] + list(range(16)))]
    # print(test_feat.shape)
    # print(train_feat.shape)
    # ============================处理类别特征=============================
    # i_ = 0
    # while i_ < len(predictors):
    #     if predictors[i_] in cat_feat:
    #         del predictors[i_]
    #         i_ = i_ - 1
    #     i_ = i_ + 1

    for each in cat_feat:
        predictors.insert(0, each)

    print('start train...')
    params = {
        'seed': 23333,
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.7,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
        # 'lambda_l1': 0.2,
        # 'lambda_l2': 0.2,
    }
    submission = pd.DataFrame()
    for i in range(16):
        print('{} times'.format(i))
        date = date_add_days('2017-08-16', i)
        lgb_train = lgb.Dataset(train_feat[predictors], train_feat[i], weight=wight_list,
                                categorical_feature=[i for i in range(len(cat_feat))])
        lgb_eval = lgb.Dataset(eval_feat[predictors], eval_feat[i],
                               categorical_feature=[i for i in range(len(cat_feat))])

        gbm = lgb.train(params,
                        lgb_train,
                        4000,
                        # valid_sets=lgb_eval,
                        # early_stopping_rounds=300,
                        # verbose_eval=300,
                        categorical_feature=[i for i in range(len(cat_feat))]
                        )
        pred = gbm.predict(test_feat[predictors])

        feat_imp = pd.Series(gbm.feature_importance('gain'), index=predictors).sort_values(
            ascending=False)
        feat_imp = pd.DataFrame(feat_imp)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        feat_imp.to_csv('../output/imp/{0}_{1}.csv'.format(time_str, i))

        submission = pd.concat([submission, pd.DataFrame({'store_nbr': test_key['store_nbr'].values,
                                                          'item_nbr': test_key['item_nbr'].values,
                                                          'date': date,
                                                          'unit_sales': np.exp(pred) - 1})])
    test = pd.read_csv(data_path + 'test.csv')
    submission = test.merge(submission, on=['store_nbr', 'item_nbr', 'date'], how='left')
    submission['unit_sales'] = submission['unit_sales'].apply(lambda x: x if x > 0 else 0)
    submission[['id', 'unit_sales']].to_csv(
        r'../output/sub/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
        index=False, float_format='%.4f')
