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
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

settings = EfficientFCParameters()

cache_path = 'cache/'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)

t0 = time.time()
t_user = pd.read_csv('data/t_user.csv')
t_click = pd.read_csv('data/t_click.csv')
t_loan = pd.read_csv('data/t_loan.csv')
t_order = pd.read_csv('data/t_order.csv')
t_loan_sum = pd.read_csv('data/t_loan_sum.csv')

t_user['active_year'] = pd.to_datetime(t_user['active_date']).dt.year
t_user['active_month'] = pd.to_datetime(t_user['active_date']).dt.month
t_user['limit_'] = np.power(5, t_user['limit']) - 1
t_user['weekday'] = pd.to_datetime(t_user['active_date']).dt.weekday
t_user['dayofmonth'] = pd.to_datetime(t_user['active_date']).dt.days_in_month
t_user = t_user.merge(t_user.groupby('active_date', as_index=False)['uid'].agg({'date_n_user': 'count'}), how='left',
                      on='active_date')

t_loan['day'] = pd.to_datetime(t_loan.loan_time).dt.dayofyear
max_day = t_loan['day'].max()
t_loan['day'] = max_day - t_loan['day']
t_loan['month'] = pd.to_datetime(t_loan.loan_time).dt.month
t_loan['loan_amount_'] = np.power(5, t_loan['loan_amount']) - 1
t_loan['loan_amount_one_plan'] = t_loan['loan_amount_'].values / t_loan.plannum.values
t_loan = t_loan.merge(t_user, how='left', on='uid')

t_click['day'] = pd.to_datetime(t_click.click_time).dt.dayofyear
t_click['day'] = max_day - t_click['day']
t_click['pidparam'] = t_click['pid'].astype(str) + "_" + t_click['param'].astype(str)

t_order['price_'] = 5 ** t_order['price'] - 1
t_order['day'] = max_day - pd.to_datetime(t_order.buy_time).dt.dayofyear


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


def get_loan_count(start_day, end_day, window_size):
    loan_count = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))].shape[0]
    return loan_count


def get_loan_count_mean(start_day, end_day, window_size):
    loan_count = t_loan[(t_loan.day >= end_day) & (t_loan.day < (end_day + window_size))]
    loan_count = loan_count.shape[0] * 1.0 / len(loan_count.day.unique())
    return loan_count


@dump_feature
def get_user_loan_left(start_day, end_day):
    user_loan = t_loan[t_loan.day >= end_day]
    month = user_loan[t_loan.day == end_day].month.values[0] + 1

    def user_loan_left(df):
        uid = df.uid.values[0]
        loan_amount = df['loan_amount_'].sum()
        df['return_periods'] = month - df.month - 1
        df['if_return'] = df['plannum'] > df['return_periods']
        df.loc[df['plannum'] < df['return_periods'], 'return_periods'] = df.loc[
            df['plannum'] < df['return_periods'], 'plannum']
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


def get_pidparam_count(start_day, end_day, window_size):
    pidparam_count = t_click[(t_click.day >= end_day) & (t_click.day < (end_day + window_size))]
    pidparam_count = \
    pidparam_count.groupby(['uid', 'pidparam'])['uid'].agg({'pidparam_count': 'count'}).unstack().fillna(0)[
        'pidparam_count']
    pidparam_count.columns = ['pidparam_count_%s_%s' % (col, window_size) for col in pidparam_count.columns]
    pidparam_count.reset_index(inplace=True)
    return pidparam_count


# day
def get_day_loan_amount_stat(start_day, end_day):
    user_day_loan = t_loan[t_loan.day >= end_day]
    user_day_loan = user_day_loan.groupby(['uid', 'day'], as_index=False)['loan_amount_'].sum()
    user_day_loan = user_day_loan.groupby(['uid'], as_index=False)['loan_amount_'].agg({'day_loan_amount_mean': 'mean'})
    return user_day_loan


def get_day_loan_count_stat(start_day, end_day):
    user_day_loan = t_loan[t_loan.day >= end_day]
    user_day_loan = user_day_loan.groupby(['uid', 'day'], as_index=False)['loan_amount_'].count()
    user_day_loan = user_day_loan.groupby(['uid'], as_index=False)['loan_amount_'].agg({'day_loan_count_mean': 'mean'})
    return user_day_loan


# month
def get_month_loan_amount_stat(start_day, end_day):
    user_month_loan = t_loan[t_loan.day >= end_day]
    user_month_loan = user_month_loan.groupby(['uid', 'month'], as_index=False)['loan_amount_'].sum()
    user_month_loan = user_month_loan.groupby(['uid'], as_index=False)['loan_amount_'].agg(
        {'month_loan_amount_mean': 'mean'})
    return user_month_loan


def get_month_loan_count_stat(start_day, end_day):
    user_month_loan = t_loan[t_loan.day >= end_day]
    user_month_loan = user_month_loan.groupby(['uid', 'month'], as_index=False)['loan_amount_'].count()
    user_month_loan = user_month_loan.groupby(['uid'], as_index=False)['loan_amount_'].agg(
        {'month_loan_count_mean': 'mean'})
    return user_month_loan


@dump_feature
def get_user_tffresh_feature(start_day, end_day):
    user_loan = t_loan[t_loan.day >= end_day]
    user_loan = extract_features(user_loan[['uid', "loan_time", 'loan_amount_', 'plannum']], column_id="uid",
                                 column_sort="loan_time")
    return user_loan


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

    samples = samples.merge(get_month_loan_amount_stat(start_day, end_day), how='left', on='uid')
    samples = samples.merge(get_day_loan_amount_stat(start_day, end_day), how='left', on='uid')

    samples = samples.merge(get_user_tffresh_feature(start_day, end_day).reset_index(), how='left', on='uid')

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
                    early_stopping_rounds=15)

    del lgb_train, lgb_eval;
    gc.collect()
    val_submit['pred'] += gbm.predict(val_feat)

val_submit['pred'] /= fold_num

print
train_feat.shape

val_submit.loc[val_submit['pred'] < 0, 'pred'] = 0
val_submit = t_user[['uid']].merge(val_submit, how='left', on='uid')

print(train_feat.shape, val_feat.shape)
print(val_submit.head(10))
print(val_submit.shape, val_submit.label.mean(), val_submit.pred.mean(), val_submit[val_submit.label > 0].label.mean(),
      val_submit[val_submit.pred > 0].pred.mean())

val_submit['pred'] = val_submit['pred'].values / val_submit['pred'].mean() * val_submit.label.mean()

print(mean_squared_error(val_submit.label.values, val_submit.pred.values) ** 0.5)
print('一共用时{}秒'.format(time.time() - t0))
print(datetime.now())