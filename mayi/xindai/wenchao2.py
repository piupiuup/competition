import pandas as pd

a = a = pd.read_csv("./lightgbm_test_cv_mean1.783868_std0.013101.csv", names=['id', 'labela'])
b = pd.read_csv("./sub20171208_184622.csv", names=['id', 'labelb'])
c = pd.read_csv("./xwc_1211.csv", names=['id', 'labelc'])
d = a.merge(b, how='left', on='id').merge(c, how='left', on='id')
d['labeld'] = (d.labela + d.labelb + d.labelc) / 3

print (d.labela.mean())
print (d.labelb.mean())
print (d.labelc.mean())
print (d.labeld.mean())

d[['id', 'labeld']].to_csv('xwc_test_ensemble_1211.csv', index=None, header=None, encoding='utf8')



import pandas as pd

a = a = pd.read_csv("./ld_test.csv", names=['id', 'labela'])
b = pd.read_csv("./赤子单模型1754.csv", names=['id', 'labelb'])
c = pd.read_csv("./xwc_1211.csv", names=['id', 'labelc'])
d = a.merge(b, how='left', on='id').merge(c, how='left', on='id')
d['labeld'] = (d.labela + d.labelb + d.labelc) / 3

print (d.labela.mean())
print (d.labelb.mean())
print (d.labelc.mean())
print (d.labeld.mean())

d[['id', 'labeld']].to_csv('xwc_test_ensemble_1211.csv', index=None, header=None, encoding='utf8')



import pandas as pd

a = a = pd.read_csv("./lightgbm_train_cv_mean1.783868_std0.013101.csv", names=['uid', 'labela'])
b = pd.read_csv("./2016-11_17939_赤子.csv", names=['uid', 'labelb'])
c = pd.read_csv("./val_1211.csv", names=['uid', 'labelc'])

d = a.merge(b, how='left', on='uid').merge(c, how='left', on='uid').merge(val_submit, how='left', on='uid')
d['labeld'] = (d.labela + d.labelb + d.labelc) / 3

print (d.labela.mean())
print (d.labelb.mean())
print (d.labelc.mean())
print (d.labeld.mean())

print (d.shape)

print (mean_squared_error(d.label.values, d.labeld.values) ** 0.5)



# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import gc
import os
import lightgbm as lgb
import sys
import warnings
import time
from datetime import datetime
warnings.filterwarnings("ignore")


cache_path = 'cache/'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)

t0 = time.time()
t_user = pd.read_csv( os.path.join('data', 't_user.csv') )
t_click = pd.read_csv( os.path.join('data', 't_click.csv') )
t_loan = pd.read_csv( os.path.join('data', 't_loan.csv') )
t_order = pd.read_csv( os.path.join('data', 't_order.csv') )
t_loan_sum = pd.read_csv( os.path.join('data', 't_loan_sum.csv') )

t_user['active_year'] = pd.to_datetime(t_user['active_date']).dt.year
t_user['active_month'] = pd.to_datetime(t_user['active_date']).dt.month
t_user['limit_'] =  5 ** t_user['limit'] - 1
t_user['weekday'] = pd.to_datetime(t_user['active_date']).dt.weekday
t_user['dayofmonth'] = pd.to_datetime(t_user['active_date']).dt.days_in_month
t_user = t_user.merge(t_user.groupby('active_date', as_index=False)['uid'].agg({'date_n_user': 'count'}), how='left', on='active_date')

t_loan['day'] = pd.to_datetime(t_loan.loan_time).dt.dayofyear
max_day = t_loan['day'].max()
t_loan['day'] =  max_day - t_loan['day']
t_loan['month'] = pd.to_datetime(t_loan.loan_time).dt.month
t_loan['loan_amount_'] = 5 ** t_loan['loan_amount'] - 1
t_loan['loan_amount_one_plan'] = t_loan['loan_amount_'].values / t_loan.plannum.values
t_loan = t_loan.merge(t_user, how='left', on='uid')

t_click['day'] = pd.to_datetime(t_click.click_time).dt.dayofyear
t_click['day'] =  max_day - t_click['day']
t_click['pidparam'] = t_click['pid'].astype(str) + "_" + t_click['param'].astype(str)

t_order['price_'] =  5 ** t_order['price'] - 1
t_order['day'] = max_day - pd.to_datetime(t_order.buy_time).dt.dayofyear




t_bt_loan = pd.read_csv( os.path.join('data', 't_bt_loan.csv') )

t_bt_loan['day'] = max_day - pd.to_datetime(t_bt_loan.loan_time).dt.dayofyear

cache_path = 'cache\\'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)


# 分组排序
def rank(data, feat1, feat2, ascending, rank_name):
    # 这部分比自己的实现的要好非常多，好好学习，大概比我实现的快六十倍
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    data[rank_name] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)[rank_name].agg({'min_rank': 'min'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data[rank_name] = data[rank_name] - data['min_rank']
    del data['min_rank']
    return data


def dump_feature(f):  # 定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    def fn(*args, **kw):  # 对传进来的函数进行包装的函数
        t_start = time.time()
        if len(args) == 0:
            dump_path = cache_path + f.__name__ + 'null' + '.pickle'
        else:
            dump_path = cache_path + f.__name__ + str(list(args)) + '.pickle'
        if os.path.exists(dump_path):
            r = pd.read_pickle(dump_path)
        else:
            r = f(*args, **kw)
            r.to_pickle(dump_path)
        gc.collect()
        t_end = time.time()
        print('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r

    return fn


def get_label(start_day, end_day):
    user_loan_amount = t_loan[(t_loan.day >= start_day) & (t_loan.day < end_day)].groupby('uid', as_index=False)[
        'loan_amount_'].sum()
    user_loan_amount['loan_amount_'] = np.log(user_loan_amount['loan_amount_'].values + 1) \
                                       / np.log(5)
    user_loan_amount.rename(columns={'loan_amount_': 'label'}, inplace=True)
    return user_loan_amount


# loan
def get_user_loan_count(start_day, end_day, window_size):
    user_loan_count = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))].groupby('uid', \
                                                                                                       as_index=False)[
        'uid'].agg({'user_loan_count_%s' % window_size: 'count'})
    return user_loan_count


def get_user_loan_day_count(start_day, end_day, window_size):
    user_loan_day_count = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))][
        ['uid', 'day']].drop_duplicates()
    user_loan_day_count = user_loan_day_count.groupby('uid', \
                                                      as_index=False)['uid'].agg(
        {'user_loan_day_count_%s' % window_size: 'count'})
    return user_loan_day_count


def get_user_loan_hour_count(start_day, end_day, window_size):
    user_loan_day_count = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))]
    user_loan_day_count['hour'] = pd.to_datetime(user_loan_day_count.loan_time).dt.hour
    user_loan_day_count = user_loan_day_count[['uid', 'day', 'hour']].drop_duplicates()
    user_loan_day_count = user_loan_day_count.groupby('uid', \
                                                      as_index=False)['uid'].agg(
        {'user_loan_hour_count_%s' % window_size: 'count'})
    return user_loan_day_count


def get_user_loan_amount_stat(start_day, end_day, window_size):
    user_loan_amount = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))].groupby('uid', \
                                                                                                        as_index=False)[
        'loan_amount_'].agg({'user_loan_amount_sum_%s' % window_size: 'sum', \
                             'user_loan_amount_mean_%s' % window_size: 'mean', \
                             'user_loan_amount_max_%s' % window_size: 'max', \
                             'user_loan_amount_median_%s' % window_size: 'median', \
                             'user_loan_amount_min_%s' % window_size: 'min', \
                             'user_loan_amount_var_%s' % window_size: 'var',
                             })
    return user_loan_amount


def get_user_plannum_stat(start_day, end_day, window_size):
    user_plannum_count = \
    t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))].groupby(['uid', 'plannum'], \
                                                                                     )['uid'].count().unstack().fillna(
        0)
    user_plannum_count.columns = ["plannum_%s_count_%s" % (col, window_size) for col in user_plannum_count.columns]
    user_plannum_count.reset_index(inplace=True)
    return user_plannum_count


def get_user_plannum2_stat(start_day, end_day, window_size):
    user_plannum_count = \
    t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))].groupby(['uid'], as_index=False \
                                                                                     )['plannum'].agg(
        {'user_plannum_sum_%s' % window_size: 'sum', \
         'user_plannum_mean_%s' % window_size: 'mean', \
         'user_plannum_max_%s' % window_size: 'max'})
    return user_plannum_count


def get_loan_amount_one_plan_stat(start_day, end_day, window_size):
    user_loan_amount_one_plan = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))].groupby('uid', \
                                                                                                                 as_index=False)[
        'loan_amount_one_plan'].agg({'loan_amount_one_plan_sum_%s' % window_size: 'sum', \
                                     'loan_amount_one_plan_mean_%s' % window_size: 'mean', \
                                     'loan_amount_one_plan_max_%s' % window_size: 'max', })
    return user_loan_amount_one_plan


def get_loan_lasttime(start_day, end_day, num):
    user_lasttime = t_loan[(t_loan.day >= end_day)][['uid', "loan_time"]]
    current_time = pd.to_datetime(t_loan[t_loan.day == (end_day)]["loan_time"]).max()
    print(current_time)
    user_lasttime['loan_time_diff'] = (current_time - pd.DatetimeIndex(user_lasttime.loan_time)).total_seconds().values
    user_lasttime = user_lasttime[['uid', 'loan_time_diff']]
    user_lasttime = rank(user_lasttime, 'uid', 'loan_time_diff', True, 'loan_time_diff_rank')
    user_lasttime = user_lasttime[user_lasttime.loan_time_diff_rank < num].set_index(['uid', \
                                                                                      'loan_time_diff_rank']).unstack().reset_index().fillna(
        -100000000)
    user_lasttime.columns = ['uid'] + ['loan_%s_lasttime_diff' % (i + 1) for i in range(num)]
    return user_lasttime


def get_loan_last(start_day, end_day):
    user_lasttime = t_loan[(t_loan.day >= end_day)][['uid', "loan_time", 'loan_amount_', 'plannum']]
    user_lasttime.sort_values(by="loan_time", inplace=True)
    user_lasttime = user_lasttime.groupby(['uid'], as_index=False).last()[
        ['uid', 'loan_amount_', 'plannum', "loan_time"]]

    user_lasttime.columns = ['uid', 'user_last_loan_amount_', 'user_last_plannum', "loan_time"]
    user_lasttime['last_hour'] = pd.to_datetime(user_lasttime.loan_time).dt.hour
    user_lasttime.drop('loan_time', axis=1, inplace=True)
    return user_lasttime


@dump_feature
def get_user_loan_left(start_day, end_day):
    user_loan = t_loan[t_loan.day >= end_day]
    month = user_loan[user_loan.day == end_day].month.values[0] + 1

    def user_loan_left(df):
        uid = df.uid.values[0]
        loan_amount = df['loan_amount_'].sum()
        df['return_periods'] = month - df.month - 1
        df['if_return'] = df['plannum'] > df['return_periods']
        df.loc[df['plannum'] < df['return_periods'], 'return_periods'] = \
            df.loc[df['plannum'] < df['return_periods'], 'plannum']
        return_amount = (df.return_periods * df.loan_amount_one_plan).sum()
        loan_left = loan_amount - return_amount
        loan_to_return_this_month = (df.if_return.astype(int) * df.loan_amount_one_plan).sum()
        loan_num_to_return_this_month = df.if_return.astype(int).sum()

        loan_amount_inrecord1 = (df.if_return.astype(int) * df.loan_amount_).sum()
        loan_amount_inrecord2 = (
        ((df.if_return) & ((df.plannum - df.return_periods) > 1)).astype(int) * df.loan_amount_).sum()

        return pd.DataFrame({'uid': [uid], 'loan_left': [loan_left], 'loan_return_amount': [return_amount],
                             'loan_to_return_this_month': [loan_to_return_this_month],
                             'loan_num_to_return_this_month': [loan_num_to_return_this_month],
                             'loan_amount_inrecord1': [loan_amount_inrecord1],
                             'loan_amount_inrecord2': [loan_amount_inrecord2]})

    user_loan = user_loan.groupby('uid').apply(user_loan_left)
    print(user_loan.shape)
    return user_loan


# user
def get_age_amount_stat(start_day, end_day):
    age_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('age', \
                                                              as_index=False)['loan_amount_'].agg(
        {'age_loan_amount_mean': 'mean', \
         'age_loan_amount_median': 'median'})
    return age_loan_amount


def get_sex_amount_stat(start_day, end_day):
    sex_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('sex', \
                                                              as_index=False)['loan_amount_'].agg(
        {'sex_loan_amount_mean': 'mean', \
         'sex_loan_amount_median': 'median'})
    return sex_loan_amount


def get_active_year_amount_stat(start_day, end_day):
    active_year_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('active_year', \
                                                                      as_index=False)['loan_amount_'].agg(
        {'active_year_loan_amount_mean': 'mean', \
         'active_year_loan_amount_median': 'median'})
    return active_year_loan_amount


def get_active_month_amount_stat(start_day, end_day):
    active_month_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('active_month', \
                                                                       as_index=False)['loan_amount_'].agg(
        {'active_month_loan_amount_mean': 'mean', \
         'active_month_loan_amount_median': 'median'})
    return active_month_loan_amount


def get_active_year_month_amount_stat(start_day, end_day):
    active_year_month_loan_amount = t_loan[(t_loan.day >= end_day)].groupby(['active_year', 'active_month'], \
                                                                            as_index=False)['loan_amount_'].agg(
        {'active_year_month_loan_amount_mean': 'mean', \
         'active_year_month_loan_amount_median': 'median'})
    return active_year_month_loan_amount


def get_age_sex_amount_stat(start_day, end_day):
    age_sex_loan_amount = t_loan[(t_loan.day >= end_day)].groupby(['age', 'sex'], \
                                                                  as_index=False)['loan_amount_'].agg(
        {'age_sex_loan_amount_mean': 'mean', \
         'age_sex_loan_amount_median': 'median'})
    return age_sex_loan_amount


def get_limit_amount_stat(start_day, end_day):
    limit_loan_amount = t_loan[(t_loan.day >= end_day)].groupby(['limit_'], \
                                                                as_index=False)['loan_amount_'].agg(
        {'limit_loan_amount_mean': 'mean', \
         'limit_loan_amount_median': 'median'})
    return limit_loan_amount


# user
def get_sex_month_amount_stat(start_day, end_day):
    age_loan_amount = t_loan[(t_loan.day >= end_day)]
    age_loan_amount = age_loan_amount.groupby(['sex', 'month'], as_index=False)['loan_amount_'].sum()
    age_loan_amount = age_loan_amount.groupby('sex', as_index=False)['loan_amount_'].agg(
        {'age_month_loan_amount': "mean"})
    return age_loan_amount


def get_sex_amount_stat(start_day, end_day):
    sex_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('sex', \
                                                              as_index=False)['loan_amount_'].agg(
        {'sex_loan_amount_mean': 'mean', \
         'sex_loan_amount_median': 'median'})
    return sex_loan_amount


def get_active_year_amount_stat(start_day, end_day):
    active_year_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('active_year', \
                                                                      as_index=False)['loan_amount_'].agg(
        {'active_year_loan_amount_mean': 'mean', \
         'active_year_loan_amount_median': 'median'})
    return active_year_loan_amount


def get_active_month_amount_stat(start_day, end_day):
    active_month_loan_amount = t_loan[(t_loan.day >= end_day)].groupby('active_month', \
                                                                       as_index=False)['loan_amount_'].agg(
        {'active_month_loan_amount_mean': 'mean', \
         'active_month_loan_amount_median': 'median'})
    return active_month_loan_amount


def get_active_year_month_amount_stat(start_day, end_day):
    active_year_month_loan_amount = t_loan[(t_loan.day >= end_day)].groupby(['active_year', 'active_month'], \
                                                                            as_index=False)['loan_amount_'].agg(
        {'active_year_month_loan_amount_mean': 'mean', \
         'active_year_month_loan_amount_median': 'median'})
    return active_year_month_loan_amount


def get_age_sex_amount_stat(start_day, end_day):
    age_sex_loan_amount = t_loan[(t_loan.day >= end_day)].groupby(['age', 'sex'], \
                                                                  as_index=False)['loan_amount_'].agg(
        {'age_sex_loan_amount_mean': 'mean', \
         'age_sex_loan_amount_median': 'median'})
    return age_sex_loan_amount


def get_limit_amount_stat(start_day, end_day):
    limit_loan_amount = t_loan[(t_loan.day >= end_day)].groupby(['limit_'], \
                                                                as_index=False)['loan_amount_'].agg(
        {'limit_loan_amount_mean': 'mean', \
         'limit_loan_amount_median': 'median'})
    return limit_loan_amount


def get_kind_amount_static_stat(kind):
    kind_loan_amount = t_loan[(t_loan.day >= 85)].groupby([kind], \
                                                          as_index=False)['loan_amount_'].agg(
        {'%s_loan_amount_mean' % kind: 'mean', \
         '%s_loan_amount_median' % kind: 'median'})
    return kind_loan_amount


# click、
@dump_feature
def get_pid_count(start_day, end_day, window_size):
    pid_count = t_click[(t_click.day >= end_day) & (t_click.day < (end_day + window_size))]
    pid_count = pid_count.groupby(['uid', 'pid'])['uid'].agg({'pid_count': 'count'}).unstack().fillna(0)
    pid_count.columns = ['pid_count_%s_%s' % (window_size, i) for i in range(10)]
    pid_count.reset_index(inplace=True)
    return pid_count


@dump_feature
def get_param_count(start_day, end_day, window_size):
    param_count = t_click[(t_click.day >= end_day) & (t_click.day < (end_day + window_size))]
    param_count = param_count.groupby(['uid', 'param'])['uid'].agg({'param_count': 'count'}).unstack().fillna(0)[
        'param_count']
    param_count.columns = ['param_count_%s_%s' % (col, window_size) for col in param_count.columns]
    param_count.reset_index(inplace=True)
    return param_count


@dump_feature
def get_pidparam_count(start_day, end_day, window_size):
    pidparam_count = t_click[(t_click.day >= end_day) & (t_click.day < (end_day + window_size))]
    pidparam_count = \
    pidparam_count.groupby(['uid', 'pidparam'])['uid'].agg({'pidparam_count': 'count'}).unstack().fillna(0)[
        'pidparam_count']
    pidparam_count.columns = ['pidparam_count_%s_%s' % (col, window_size) for col in pidparam_count.columns]
    pidparam_count.reset_index(inplace=True)
    return pidparam_count


def get_loan_timediff_stat(start_day, end_day, window_size):
    user_lasttime = t_loan[(t_loan.day >= end_day) & (t_loan.day < end_day + window_size)][['uid', "loan_time"]]
    user_lasttime.sort_values(by=['uid', "loan_time"], inplace=True)

    user_lasttime['loan_diff'] = (pd.to_datetime(user_lasttime.loan_time) - \
                                  pd.to_datetime(user_lasttime.loan_time.shift(1))).dt.total_seconds()

    user_lasttime['last_uid'] = user_lasttime.uid.shift(1)

    user_lasttime = user_lasttime[user_lasttime.uid == user_lasttime.last_uid]

    user_lasttime = user_lasttime.groupby('uid', as_index=False)['loan_diff'].mean()[['uid', 'loan_diff']]
    user_lasttime.columns = ['uid', 'loan_diff_%s' % window_size]
    return user_lasttime


from dateutil.parser import parse
from collections import defaultdict


# 相差的日期数
def diff_of_days(day1, day2):
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return abs(days)


# 最大借款额度
def get_user_max_limit(start_day, end_day, window_size):
    result_path = cache_path + 'user_max_limit_{0}_{1}.hdf'.format(start_day, end_day)

    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        loan_temp = t_loan[(t_loan['day'] >= end_day) & (t_loan['day'] < end_day + window_size)]
        loan_temp['loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_days('2016-01-01', x))
        loan_temp.set_index('uid', drop=False, inplace=True)
        loan_temp['0'] = 0
        min_time = loan_temp['loan_time'].min()
        max_time = loan_temp['loan_time'].max()
        user_own = loan_temp[['uid', 'loan_time', '0']].drop_duplicates().set_index(['uid', 'loan_time'])['0'].unstack()

        user_own = user_own[[f for f in user_own.columns if f > min_time + 30]].reset_index().fillna(0)

        for i in range(min_time, max_time + 1):
            loan_sub = loan_temp[loan_temp['loan_time'] == i].copy()
            for j in [1, 3, 6, 12]:
                start_day = i
                end_day = i + j * 30
                user_loan_dict = defaultdict(lambda: 0)
                user_loan_dict.update(loan_sub[loan_sub['plannum'] == j].groupby('uid')['loan_amount'].sum())
                for k in user_own.drop('uid', axis=1).columns:
                    if (k >= start_day) & (k < end_day):
                        user_own[k] += user_own['uid'].map(user_loan_dict)
        user_own.set_index('uid', inplace=True)
        user_own = user_own[[f for f in user_own.columns if f > min_time + 30]]
        result = t_user[['uid', 'limit_']]
        result['mean_own'] = result['uid'].map(user_own.mean(axis=1))
        result['max_own'] = result['uid'].map(user_own.max(axis=1))
        result['std_own'] = result['uid'].map(user_own.std(axis=1))
        result['std2_own'] = (result['uid'].map(user_own.std(axis=1)) / result['mean_own'] + 0.01).fillna(-1)
        result['dfii_limit'] = result['limit_'] - result['max_own']
        result.fillna(0, inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# loan
def get_user_bt_loan_count(start_day, end_day, window_size):
    user_loan_count = t_bt_loan[(t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size))].groupby('uid', \
                                                                                                                as_index=False)[
        'uid'].agg({'user_bt_loan_count_%s' % window_size: 'count'})
    return user_loan_count


def get_user_bt_loan_day_count(start_day, end_day, window_size):
    user_loan_day_count = t_bt_loan[(t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size))][
        ['uid', 'day']].drop_duplicates()
    user_loan_day_count = user_loan_day_count.groupby('uid', \
                                                      as_index=False)['uid'].agg(
        {'user_bt_loan_day_count_%s' % window_size: 'count'})
    return user_loan_day_count


def get_user_bt_loan_hour_count(start_day, end_day, window_size):
    user_loan_day_count = t_bt_loan[(t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size))]
    user_loan_day_count['hour'] = pd.to_datetime(user_loan_day_count.loan_time).dt.hour
    user_loan_day_count = user_loan_day_count[['uid', 'day', 'hour']].drop_duplicates()
    user_loan_day_count = user_loan_day_count.groupby('uid', \
                                                      as_index=False)['uid'].agg(
        {'user_bt_loan_hour_count_%s' % window_size: 'count'})
    return user_loan_day_count


def get_user_bt_plannum_stat(start_day, end_day, window_size):
    user_plannum_count = \
    t_bt_loan[(t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size))].groupby(['uid', 'plannum'], \
                                                                                              )[
        'uid'].count().unstack().fillna(0)
    user_plannum_count.columns = ["plannum_bt_%s_count_%s" % (col, window_size) for col in user_plannum_count.columns]
    user_plannum_count.reset_index(inplace=True)
    return user_plannum_count


def get_user_bt_plannum2_stat(start_day, end_day, window_size):
    user_plannum_count = \
    t_bt_loan[(t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size))].groupby(['uid'], as_index=False \
                                                                                              )['plannum'].agg(
        {'user_bt_plannum_sum_%s' % window_size: 'sum', \
         'user_bt_plannum_mean_%s' % window_size: 'mean', \
         'user_bt_plannum_max_%s' % window_size: 'max'})
    return user_plannum_count


def get_user_bt_plannum3_stat(start_day, end_day, window_size):
    user_plannum_count = t_bt_loan[
        (t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size)) & (t_bt_loan.is_free == 0)].groupby(
        ['uid'], as_index=False \
        )['plannum'].agg({'user_bt_free0_plannum_sum_%s' % window_size: 'sum', \
                          'user_bt_free0_plannum_mean_%s' % window_size: 'mean', \
                          'user_bt_free0_plannum_max_%s' % window_size: 'max'})
    return user_plannum_count


def get_user_bt_plannum4_stat(start_day, end_day, window_size):
    user_plannum_count = t_bt_loan[
        (t_bt_loan.day >= end_day) & (t_bt_loan.day < (end_day + window_size)) & (t_bt_loan.is_free == 1)].groupby(
        ['uid'], as_index=False \
        )['plannum'].agg({'user_bt_free1_plannum_sum_%s' % window_size: 'sum', \
                          'user_bt_free1_plannum_mean_%s' % window_size: 'mean', \
                          'user_bt_free1_plannum_max_%s' % window_size: 'max'})
    return user_plannum_count


def make_set(start_day, end_day):
    dump_path = cache_path + 'dataset_%s_%s.pickle' % (start_day, end_day)

    if not os.path.exists(dump_path):
        current_time = pd.to_datetime(t_loan[t_loan.day == (end_day)]["loan_time"]).max()
        samples = t_user[pd.to_datetime(t_user.active_date) <= current_time][['uid']]

        samples = samples.merge(get_user_loan_day_count(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_user_loan_day_count(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_user_loan_hour_count(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_user_loan_count(start_day, end_day, 7), how='left', on='uid')
        samples = samples.merge(get_user_loan_count(start_day, end_day, 14), how='left', on='uid')
        samples = samples.merge(get_user_loan_count(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_user_loan_count(start_day, end_day, 60), how='left', on='uid')
        samples = samples.merge(get_user_loan_count(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_user_loan_amount_stat(start_day, end_day, 7), how='left', on='uid')
        samples = samples.merge(get_user_loan_amount_stat(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_user_loan_amount_stat(start_day, end_day, 60), how='left', on='uid')
        samples = samples.merge(get_user_loan_amount_stat(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_user_plannum_stat(start_day, end_day, 7), how='left', on='uid')
        samples = samples.merge(get_user_plannum_stat(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_user_plannum_stat(start_day, end_day, 60), how='left', on='uid')
        samples = samples.merge(get_user_plannum_stat(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_user_plannum2_stat(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_loan_amount_one_plan_stat(start_day, end_day, 7), how='left', on='uid')
        samples = samples.merge(get_loan_amount_one_plan_stat(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_loan_amount_one_plan_stat(start_day, end_day, 60), how='left', on='uid')
        samples = samples.merge(get_loan_amount_one_plan_stat(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_loan_lasttime(start_day, end_day, 2), how='left', on='uid')

        samples = samples.merge(t_user[["uid", "age", "sex", "limit_", 'active_date', 'active_year', 'active_month']],
                                on='uid', how='left')
        samples = samples.merge(get_age_amount_stat(start_day, end_day), on='age', how='left')
        samples = samples.merge(get_sex_amount_stat(start_day, end_day), on='sex', how='left')
        samples = samples.merge(get_active_year_amount_stat(start_day, end_day), on='active_year', how='left')
        samples = samples.merge(get_active_month_amount_stat(start_day, end_day), on='active_month', how='left')

        samples['limit_user_loan_amount_sum_%s' % 7] = samples['limit_'] - samples['user_loan_amount_sum_%s' % 7]
        samples['limit_user_loan_amount_sum_%s' % 21] = samples['limit_'] - samples['user_loan_amount_sum_%s' % 21]
        samples['limit_user_loan_amount_sum_%s' % 60] = samples['limit_'] - samples['user_loan_amount_sum_%s' % 60]
        samples['limit_user_loan_amount_sum_%s' % 150] = samples['limit_'] - samples['user_loan_amount_sum_%s' % 150]

        samples['active_days'] = (pd.to_datetime(samples.active_date) - pd.to_datetime('2016-11-01')).dt.days
        samples = samples.drop('active_date', axis=1)

        samples = samples.merge(get_user_loan_left(start_day, end_day), how='left', on='uid')

        samples = samples.merge(get_pid_count(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_pid_count(start_day, end_day, 150), how='left', on='uid')

        samples = samples.merge(get_param_count(start_day, end_day, 21), how='left', on='uid')
        samples = samples.merge(get_param_count(start_day, end_day, 150), how='left', on='uid')

        if start_day >= 0:
            label = get_label(start_day, end_day)
            samples = samples.merge(label, how='left', on='uid').fillna(0)
        samples.to_pickle(dump_path)

    else:
        samples = pd.read_pickle(dump_path)

    samples['limit_left1'] = samples['limit_'] - samples['loan_amount_inrecord1']
    samples['limit_left1_ratio'] = samples['loan_amount_inrecord1'].values / samples['limit_'].values

    samples['limit_left2'] = samples['limit_'] - samples['loan_amount_inrecord2']
    samples['limit_left2_ratio'] = samples['loan_amount_inrecord2'].values / samples['limit_'].values

    samples.drop('active_days', axis=1, inplace=True)

    samples = samples.merge(get_pidparam_count(start_day, end_day, 150), how='left', on='uid')
    samples = samples.merge(get_pidparam_count(start_day, end_day, 21), how='left', on='uid')
    samples = samples.merge(get_pidparam_count(start_day, end_day, 14), how='left', on='uid')
    samples = samples.merge(get_pidparam_count(start_day, end_day, 7), how='left', on='uid')

    samples = samples.merge(get_limit_amount_stat(start_day, end_day), how='left', on='limit_')

    samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 7), how='left', on='uid')
    samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 4), how='left', on='uid')
    samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 2), how='left', on='uid')
    samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 1), how='left', on='uid').fillna(-1000)

    samples = samples.merge(get_user_max_limit(start_day, end_day, 150).drop('limit_', axis=1), how='left', on='uid')

    samples = samples.merge(get_user_bt_plannum2_stat(start_day, end_day, 150), how='left', on='uid')
    samples = samples.merge(get_user_bt_plannum2_stat(start_day, end_day, 60), how='left', on='uid').fillna(-1000)

    return samples


train1 = make_set(0, 30)
train2 = make_set(30, 61)

train = pd.concat([train2], axis=0).drop_duplicates()

no_predictors = ['uid', 'label']
train_feat = train.drop(no_predictors, axis=1).astype(float)
training_label = train.label.values

val = train1
val_submit = val[['uid', 'label']]
val_feat = val.drop(no_predictors, axis=1).astype(float).fillna(-999)
val_submit['pred'] = 0

useful_columns = train_feat.columns
train_feat = train_feat[list(set(useful_columns) & set(val_feat.columns.tolist()))]
val_feat = val_feat[list(set(useful_columns) & set(val_feat.columns.tolist()))]

print(train_feat.shape, val_feat.shape)

pca_num = 1
pca1 = PCA(n_components=pca_num)
pidparam_cols = [col for col in train_feat.columns if 'pidparam' in col and '_150' in col]

train_pid_param_pca = pca1.fit_transform(train_feat[pidparam_cols].fillna(0))
val_pid_param_pca = pca1.transform(val_feat[pidparam_cols].fillna(0))

train_feat = pd.concat([train_feat.drop(pidparam_cols, axis=1),
                        pd.DataFrame(train_pid_param_pca, columns=['pid_param_pca1_%s' % i for i in range(pca_num)])],
                       axis=1)
val_feat = pd.concat([val_feat.drop(pidparam_cols, axis=1),
                      pd.DataFrame(val_pid_param_pca, columns=['pid_param_pca1_%s' % i for i in range(pca_num)])],
                     axis=1)

pca_num = 1
pca1 = PCA(n_components=pca_num)
pidparam_cols = [col for col in train_feat.columns if 'pidparam' in col and '_21' in col]

train_pid_param_pca = pca1.fit_transform(train_feat[pidparam_cols].fillna(0))
val_pid_param_pca = pca1.transform(val_feat[pidparam_cols].fillna(0))

train_feat = pd.concat([train_feat.drop(pidparam_cols, axis=1),
                        pd.DataFrame(train_pid_param_pca, columns=['pid_param_pca2_%s' % i for i in range(pca_num)])],
                       axis=1)
val_feat = pd.concat([val_feat.drop(pidparam_cols, axis=1),
                      pd.DataFrame(val_pid_param_pca, columns=['pid_param_pca2_%s' % i for i in range(pca_num)])],
                     axis=1)

pca_num = 1
pca1 = PCA(n_components=pca_num)
pidparam_cols = [col for col in train_feat.columns if 'pidparam' in col and '_14' in col]

train_pid_param_pca = pca1.fit_transform(train_feat[pidparam_cols].fillna(0))
val_pid_param_pca = pca1.transform(val_feat[pidparam_cols].fillna(0))

train_feat = pd.concat([train_feat.drop(pidparam_cols, axis=1),
                        pd.DataFrame(train_pid_param_pca, columns=['pid_param_pca3_%s' % i for i in range(pca_num)])],
                       axis=1)
val_feat = pd.concat([val_feat.drop(pidparam_cols, axis=1),
                      pd.DataFrame(val_pid_param_pca, columns=['pid_param_pca3_%s' % i for i in range(pca_num)])],
                     axis=1)

pca_num = 1
pca1 = PCA(n_components=pca_num)
pidparam_cols = [col for col in train_feat.columns if 'pidparam' in col and '_7' in col]

train_pid_param_pca = pca1.fit_transform(train_feat[pidparam_cols].fillna(0))
val_pid_param_pca = pca1.transform(val_feat[pidparam_cols].fillna(0))

train_feat = pd.concat([train_feat.drop(pidparam_cols, axis=1),
                        pd.DataFrame(train_pid_param_pca, columns=['pid_param_pca4_%s' % i for i in range(pca_num)])],
                       axis=1)
val_feat = pd.concat([val_feat.drop(pidparam_cols, axis=1),
                      pd.DataFrame(val_pid_param_pca, columns=['pid_param_pca4_%s' % i for i in range(pca_num)])],
                     axis=1)

pca_num = 1
pca1 = PCA(n_components=pca_num)
pidparam_cols = [col for col in train_feat.columns if 'pid_count_21' in col]

train_pid_param_pca = pca1.fit_transform(train_feat[pidparam_cols].fillna(0))
val_pid_param_pca = pca1.transform(val_feat[pidparam_cols].fillna(0))

train_feat = pd.concat([train_feat.drop(pidparam_cols, axis=1),
                        pd.DataFrame(train_pid_param_pca, columns=['pid_param_pca5_%s' % i for i in range(pca_num)])],
                       axis=1)
val_feat = pd.concat([val_feat.drop(pidparam_cols, axis=1),
                      pd.DataFrame(val_pid_param_pca, columns=['pid_param_pca5_%s' % i for i in range(pca_num)])],
                     axis=1)

val = train1
val_submit = val[['uid', 'label']]
val_submit['pred'] = 0

fold_num = 5
skf = KFold(n_splits=fold_num)
for idx, (train_index, test_index) in enumerate(skf.split(train_feat, training_label)):
    X_train, X_test = train_feat.iloc[train_index], train_feat.iloc[test_index]
    y_train, y_test = training_label[train_index], training_label[test_index]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    np.random.seed(idx)
    params = {
        'boosting_type': 'gbrt',
        'objective': 'regression',
        'metric': ['rmse'],
        'num_leaves': np.random.choice(np.arange(30, 50, 2)),
        'learning_rate': np.random.choice(np.arange(0.05, 0.07, 0.001)),
        'feature_fraction': np.random.choice(np.arange(0.82, 0.9, 0.01)),
        'bagging_fraction': np.random.choice(np.arange(0.82, 0.88, 0.01)),
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 32,
        'is_unbalance': True,
        'min_data_in_leaf': np.random.choice(np.arange(160, 240, 10)),
        'seed': 2017
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=400,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=True,
                    early_stopping_rounds=20)

    del lgb_train, lgb_eval;
    gc.collect()
    val_submit['pred'] += gbm.predict(val_feat)

val_submit['pred'] /= fold_num

names, importances = zip(*(sorted(zip(gbm.feature_name(), gbm.feature_importance()), key=lambda x: x[1])))
for name, importance in zip(names, importances):
    print(name, importance)

print(train_feat.shape)

val_submit.loc[val_submit['pred'] < 0, 'pred'] = 0
val_submit = t_user[['uid']].merge(val_submit, how='left', on='uid')

print(train_feat.columns)
print(train_feat.shape, val_feat.shape)
print(val_submit.shape, val_submit.label.mean(), val_submit.pred.mean(), val_submit[val_submit.label > 0].label.mean(),
      val_submit[val_submit.pred > 0].pred.mean())

val_submit['pred'] = val_submit['pred'].values / val_submit['pred'].mean() * val_submit.label.mean()
print(val_submit.shape)
# val_submit.to_csv('xwc_val.csv', index=None, header=None, encoding='utf8')
val_submit_lgb = val_submit.copy()
print(mean_squared_error(val_submit.label.values, val_submit.pred.values) ** 0.5)
print('一共用时{}秒'.format(time.time() - t0))
print(datetime.now())














samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 7), how='left', on='uid')
samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 4), how='left', on='uid')
samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 2), how='left', on='uid')
samples = samples.merge(get_loan_timediff_stat(start_day, end_day, 1), how='left', on='uid').fillna(-1000)














