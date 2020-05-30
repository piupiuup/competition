import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from dateutil.parser import parse
from collections import defaultdict
from joblib import Parallel, delayed
from datetime import datetime,timedelta

cache_path = 'F:/xindai_cache/'
data_path = 'C:/Users/csw/Desktop/python/JD/xindai/data/'
user_path = data_path + 't_user.csv'
order_path = data_path + 't_order.csv'
click_path = data_path + 't_click.csv'
loan_path = data_path + 't_loan.csv'
loan_sum_path = data_path + 't_loan_sum.csv'
flag = True

user = pd.read_csv(user_path)
order = pd.read_csv(order_path)
click = pd.read_csv(click_path)
loan = pd.read_csv(loan_path)
loan_sum = pd.read_csv(loan_sum_path)

# 对金额进行处理
user['limit'] = np.round(5**(user['limit'])-1,2)
order['price'] = np.round(5**(order['price'])-1,2)
order['discount'] = np.round(5**(order['discount'])-1,2)
loan['loan_amount'] = np.round(5**(loan['loan_amount'])-1,2)
loan_sum['loan_sum'] = np.round(5**(loan_sum['loan_sum'])-1,2)

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
    return abs(days)

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

# 获取标签
def get_label(result, start_date,n_days=30):
    label_start_date = date_add_days(start_date, n_days)
    label = loan[(loan['loan_time']<label_start_date) & (loan['loan_time']>=start_date)]
    label = label.groupby('uid',as_index=False)['loan_amount'].agg({'loan_sum':'sum'})
    label['loan_sum'] = np.log1p(label['loan_sum']) / np.log(5)
    result = result.merge(label[['uid', 'loan_sum']], how='left')
    result['loan_sum'].fillna(0,inplace=True)
    return result

# 获取转化率
def get_rate(data,name,label='loan_sum'):
    rate = data.groupby(name)[label].agg({'count': 'count', 'sum': 'sum'})
    rate['rate'] = rate['sum'] / rate['count']
    return rate['rate']


def make_set(start_date):
    data = pd.DataFrame()
    for i in range(2):
        label_end_days = date_add_days(start_date, i*(0 - 30))
        label_start_days = date_add_days(start_date, (i+1)*(0 - 30))
        label = loan[(loan['loan_time'] < label_end_days) & (loan['loan_time'] >= label_start_days)]
        label = label.groupby('uid', as_index=False)['loan_amount'].agg({'loan_sum': 'sum'})
        label['loan_sum'] = np.log1p(label['loan_sum']) / np.log(5)
        data_sub = user.merge(label[['uid', 'loan_sum']], how='left').fillna(0)
        data_sub['label'] = data_sub['loan_sum'].apply(lambda x: 1 if x > 0 else 0)
        data = pd.concat([data,data_sub])
    return data


###################################################
#..................... 构造特征 ....................
###################################################
# 用户基本特征
def get_user_feat(start_date):
    data = make_set(start_date).fillna(-1)
    data['label'] = data['loan_sum'].apply(lambda x: 1 if x > 0 else 0)
    data['age_rate'] = data['age'].map(get_rate(data, 'age', label='loan_sum'))
    data['week'] = pd.to_datetime(data['active_date']).dt.weekday
    data['week_rate'] = data['week'].map(get_rate(data, 'week', label='loan_sum'))
    data['month'] = pd.to_datetime(data['active_date']).dt.month
    data['month_rate'] = data['month'].map(get_rate(data, 'month', label='loan_sum'))
    date_rate = get_rate(data, 'active_date', label='loan_sum').reset_index()
    date_rate1 = date_rate.rename(columns={'rate':'date_rate1'})
    date_rate1['active_date'] = date_rate1['active_date'].apply(lambda x:str(date_add_days(x,+1))[:10])
    date_rate2 = date_rate.rename(columns={'rate': 'date_rate2'})
    date_rate2['active_date'] = date_rate2['active_date'].apply(lambda x:str(date_add_days(x,-1)[:10]))
    date_rate = date_rate.merge(date_rate1,on='active_date', how='left')
    date_rate = date_rate.merge(date_rate2, on='active_date', how='left')
    date_rate['date_rate_mean'] = date_rate[['rate','date_rate1','date_rate2']].mean(axis=1)
    data = data.merge(date_rate[['active_date','date_rate_mean']],on='active_date',how='left')
    data['date_rate'] = data['active_date'].map(get_rate(data, 'active_date', label='loan_sum'))
    data['date_n_people'] = data['active_date'].map(data['active_date'].value_counts())
    data['limit_rate'] = data['limit'].map(get_rate(data, 'limit', label='loan_sum'))
    data['active_date'] = data['active_date'].apply(lambda x: diff_of_days('2016-11-01', x))
    data['user_pred'] = data['age_rate'] * (data['week_rate'] * data['month_rate'] +
                                       data['date_rate'] * 32) * data['limit_rate'] ** 0.4
    data = data.drop_duplicates('uid').drop(['loan_sum','label'],axis=1)
    return data

# 借贷概率
def get_loan_pred(start_date):
    loan_temp = loan[loan['loan_time'] < start_date]
    loan_temp = rank(loan_temp, 'uid', 'loan_time', ascending=False)
    def last_time_rate(x):
        return 0.55 / np.exp(x * 0.03) + 0.2
    def not_loan_rate(x):
        return 0.156/np.exp(90*0.036)+0.054
    last_plannum_rate = {1: 0.707, 3: 0.5666, 6: 0.5492, 12: 0.3862}
    def last_amount_rate(x):
        return -0.1636 * (4.36 - np.log(x + 1) / np.log(5)) ** 2 + 0.689
    loan_temp['last_time_rate'] = loan_temp['loan_time'].apply(
        lambda x: last_time_rate(diff_of_days(start_date, x)))
    loan_temp['last_plannum_rate'] = loan_temp['plannum'].apply(lambda x: last_plannum_rate[x])
    loan_temp['last_amount_rate'] = loan_temp['loan_amount'].apply(lambda x: last_amount_rate(x))
    loan_temp['loan_pred1'] = loan_temp.apply(
        lambda x: x.last_time_rate * (x.last_plannum_rate * 0.85 + x.last_amount_rate * 0.15) ** 0.9, axis=1)
    loan_temp['loan_pred2'] = loan_temp['loan_pred1'] / np.exp(loan_temp['rank'] * 0.3)
    loan_temp['loan_pred2'] = loan_temp['uid'].map(loan_temp.groupby('uid')['loan_pred2'].sum())
    max_diff_days = diff_of_days(start_date, loan_temp['loan_time'].min())
    nan_values = not_loan_rate(max_diff_days)
    loan_pred = loan_temp[loan_temp['rank'] == 0].copy().set_index('uid', drop=False)
    result = user[['uid']].merge(loan_pred[['uid', 'last_time_rate', 'last_plannum_rate', 'last_amount_rate',
                                            'loan_pred1','loan_pred2']], on='uid', how='left')
    result['loan_pred2'].fillna(nan_values ** 1.9, inplace=True)
    result['loan_pred1'].fillna(nan_values ** 1.9, inplace=True)
    return result


# 30天前是否有借款几率
def get_loan_history(start_date,n_day):
    result_path = cache_path + 'loan_history_{0}_{1}.hdf'.format(start_date,n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date,(0 - n_day))
        data_end_date = start_date
        loan_temp = loan[(loan['loan_time']>=data_start_date) & (loan['loan_time']<data_end_date)]
        loan_temp['return_permonth'] = loan_temp['loan_amount'] / loan_temp['plannum']
        loan_stat = loan_temp.groupby('uid',as_index=False)['loan_amount'].agg({
            'sum_loan_{}day'.format(n_day): 'sum',
            'sum2_loan_{}day'.format(n_day): lambda x: sum(np.log(x+1)),
            'sum_loan_{}day_inverse'.format(n_day): lambda x: sum(1/np.log(x + 1)),
            'count_loan_{}'.format(n_day): 'count',
            'mean_loan_{}'.format(n_day): 'mean',
            'mean2_loan_{}'.format(n_day):  lambda x: np.mean(1/np.log(x + 1)),
            'median_loan_{}'.format(n_day): 'median',
            'max_loan_{}'.format(n_day): 'max',
            'min_loan_{}'.format(n_day): 'min',
            'std_loan_{}'.format(n_day): 'std',
            'std2_loan_{}'.format(n_day): lambda x: np.std(np.log(x + 1))
        })
        loan_stat['sum2_loan_{}day'.format(n_day)] = loan_stat['sum2_loan_{}day'.format(n_day)] / n_day
        loan_stat['sum_loan_{0}day_inverse'.format(n_day)] = loan_stat['sum_loan_{0}day_inverse'.format(n_day)] / n_day
        loan_temp['loan_amount'] = np.log(loan_temp['loan_amount']+1)/np.log(5)
        loan_temp['loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date, x))
        for i in [0.999,0.995,0.99,0.95,0.9]:
            loan_temp['weight'] = i**loan_temp['loan_time']
            loan_temp['weight'] = loan_temp['weight'] * loan_temp['loan_amount']
            loan_stat['sum_loan_{0}day_{1}w'.format(n_day,i)] = loan_stat['uid'].map(loan_temp.groupby('uid')['weight'].sum())/((1-i**n_day)/(1-i))
        loan_stat['user_nunique_plannum{}'.format(n_day)] = loan_stat['uid'].map(loan_temp.groupby('uid')['plannum'].nunique()).fillna(0)
        loan_stat['user_sum_plannum{}'.format(n_day)] = loan_stat['uid'].map(loan_temp.groupby('uid')['plannum'].sum()).fillna(0)
        loan_stat['user_loan_nday{}'.format(n_day)] = loan_stat['uid'].map(loan_temp.groupby('uid')['loan_time'].nunique()).fillna(0)/n_day
        loan_stat['user_loan_plannum>1{}'.format(n_day)] = loan_stat['uid'].map(
            loan_temp[loan_temp['plannum']>1].groupby('uid').size()).fillna(0) / n_day
        loan_stat['user_return_pernum{}'.format(n_day)] = loan_stat['sum_loan_{0}day'.format(n_day)]/loan_stat['user_sum_plannum{}'.format(n_day)]
        loan_stat['user_return_2pernum{}'.format(n_day)] = loan_stat['sum2_loan_{0}day'.format(n_day)] / loan_stat[ 'user_sum_plannum{}'.format(n_day)]
        result = user[['uid']].merge(loan_stat,on='uid',how='left').fillna(0)
        result = result.drop('uid',axis=1).fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 本月需还款 以及 剩余还款
def get_loan_now(start_date,n_month):
    user_new = pd.DataFrame({'uid':user['uid'].values,'user_owe{}'.format(n_month):0,'user_return{}'.format(n_month):0,
                             'user_return_count{}'.format(n_month):0,'user_owe_count{}'.format(n_month):0})
    loan_temp = loan.copy()
    loan_temp['return_permonth'] = loan_temp['loan_amount'] / loan_temp['plannum'] #每月还款
    for i in range(n_month):
        data_start_date = date_add_days(start_date, (0 - i - 1) * 30)
        data_end_date = date_add_days(start_date, (0 - i) * 30)
        loan_temp = loan_temp[(loan_temp['loan_time'] >= data_start_date) & (loan_temp['loan_time'] < data_end_date)]
        loan_temp = loan_temp[loan_temp['plannum'] > i]
        loan_temp['owe_permonth'] = loan_temp['loan_amount'] - (i*loan_temp['return_permonth'])
        user_return_permonth = loan_temp.groupby('uid')['return_permonth'].sum()
        user_new['user_return{}'.format(n_month)] += user_new['uid'].map(user_return_permonth).fillna(0)
        user_new['user_return_count{}'.format(n_month)] += (~user_new['uid'].map(user_return_permonth).isnull())
        user_own_permonth = loan_temp.groupby('uid')['loan_amount'].sum()
        user_new['user_owe{}'.format(n_month)] += user_new['uid'].map(user_own_permonth).fillna(0)
        user_new['user_owe_count{}'.format(n_month)] += (~user_new['uid'].map(user_own_permonth).fillna(0).isnull())
    user_new['user_return_rate{}'.format(n_month)] = user_new['user_return{}'.format(n_month)] / user_new['user_owe{}'.format(n_month)]
    result = user_new.drop('uid', axis=1)
    return result


# 用户最近一次借款
def get_loan_last(start_date):
    data_end_date = start_date
    loan_temp = loan[(loan['loan_time'] < data_end_date)].copy().sort_values('loan_time')
    loan_temp = rank(loan_temp,'uid','loan_time',False)
    loan_temp['diff_loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_minutes(start_date, x))
    loan_temp['loan_hour'] = loan_temp['loan_time'].str[11:13].astype(int)
    loan_last = loan_temp[loan_temp['rank'] == 0].copy()
    loan_last2 = loan_temp[loan_temp['rank'] == 1].copy()
    loan_last3 = loan_temp[loan_temp['rank'] == 2].copy()
    loan_last = loan_last.rename(columns={'diff_loan_time':'loan_last_time','loan_hour':'loan_last_hour',
                              'loan_amount':'last_loan_amount','plannum':'last_plannum'}).drop(['rank','loan_time'],axis=1)
    loan_last2 = loan_last2.rename(columns={'diff_loan_time': 'loan_last_time2', 'loan_hour': 'loan_last_hour2',
                              'loan_amount': 'last_loan_amount2', 'plannum': 'last_plannum2'}).drop(['rank','loan_time'], axis=1)
    loan_last3 = loan_last3.rename(columns={'diff_loan_time': 'loan_last_time3', 'loan_hour': 'loan_last_hour3',
                                            'loan_amount': 'last_loan_amount3', 'plannum': 'last_plannum3'}).drop( ['rank', 'loan_time'], axis=1)
    # user_plannum的最后一次
    last_loan_temp = loan_temp.drop_duplicates(['uid','plannum'],keep='first')
    last_loan_temp = last_loan_temp[['uid','plannum','diff_loan_time']].set_index(['uid','plannum']).unstack()
    last_loan_temp.columns = last_loan_temp.columns.droplevel(0)
    last_loan_temp = last_loan_temp.add_prefix('last_plannum_date_').reset_index()
    result = user[['uid']].merge(loan_last, on='uid', how='left')
    result = result.merge(loan_last2, on='uid', how='left')
    result = result.merge(loan_last3, on='uid', how='left')
    result = result.merge(last_loan_temp, on='uid', how='left')
    result = result.drop('uid', axis=1)
    return result


# 用户最大每月还款额 以及 用户最多的plannum 最活跃的时间
def get_loan_max(start_date):
    data_end_date = start_date
    loan_temp = loan[(loan['loan_time'] < data_end_date)].copy().sort_values('loan_time')
    loan_temp['return_permonth'] = loan_temp['loan_amount']/loan_temp['plannum']
    result = user[['uid']].copy()
    result['user_max_return'] = result['uid'].map(loan_temp.groupby('uid')['return_permonth'].max())
    uid_plannum_count = loan_temp.groupby(['uid','plannum'],as_index=False)['uid'].agg({'uid_plannum_count':'count'})
    uid_plannum_count = uid_plannum_count.sort_values('plannum',ascending=False).sort_values('uid_plannum_count',ascending=False)
    most_freq_uid_plannum = uid_plannum_count.drop_duplicates('uid',keep='first')
    uid_plannum_count = uid_plannum_count.set_index(['uid','plannum']).unstack()
    uid_plannum_count.columns = uid_plannum_count.columns.droplevel(0)
    uid_plannum_count = uid_plannum_count.add_prefix('uid_plannum_count_').reset_index()
    result = result.merge(most_freq_uid_plannum[['uid','plannum']].rename(columns={'plannum':'freq_plannum'}),on='uid',how='left')
    result = result.merge(uid_plannum_count,on='uid',how='left')
    return result

# 每隔n天统计一次3个月
def get_loan_window(start_date, n_day):
    loan_window_sum = user[['uid']].copy()
    n_window = 90//n_day
    for i in range(n_window):
        data_start_date = date_add_days(start_date, (0 - i - 1) * n_day)
        data_end_date = date_add_days(start_date, (0 - i) * n_day)
        loan_temp = loan[(loan['loan_time'] >= data_start_date) & (loan['loan_time'] < data_end_date)]
        loan_window_sum_sub = loan_temp.groupby('uid',as_index=False)['loan_amount'].agg({'loan_window_sum{0}_{1}'.format(n_day,i):'sum'})
        loan_window_sum = loan_window_sum.merge(loan_window_sum_sub,on='uid',how='left')
    result = loan_window_sum.drop('uid', axis=1)
    return result

# 一个月借款多少天
def get_loan_nday_count(start_date, n_day, period):
    data_start_date = date_add_days(start_date, (0 - period))
    data_end_date = start_date
    loan_temp = loan[(loan['loan_time'] >= data_start_date) & (loan['loan_time'] < data_end_date)].copy()
    loan_temp['loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date,x[:10])//n_day)
    loan_day_count = loan_temp.groupby('uid',as_index=False)['loan_time'].agg({'loan_{0}day_count{1}'.format(n_day, period):'nunique'})
    result  = user[['uid']].merge(loan_day_count,on='uid',how='left')
    result = result.drop('uid', axis=1)
    return result

# 最近一次交易
def get_order_last(start_date):
    result_path = cache_path + 'order_last_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_end_date = start_date
        order_temp = order[(order['buy_time'] < data_end_date)].copy().sort_values('buy_time')
        order_temp['order_date'] = order_temp['buy_time'].apply(lambda x: diff_of_days(start_date, x))
        loan_last = order_temp.drop_duplicates('uid', keep='last').drop('buy_time', axis=1)
        loan_last.rename(columns={'order_date': 'order_last_date'}, inplace=True)
        result = user[['uid']].merge(loan_last, on='uid', how='left')
        result = result.drop('uid', axis=1)[['order_last_date']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 最近一次交易
def get_click_last(start_date):
    result_path = cache_path + 'click_last_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_end_date = start_date
        click_temp = click[(click['click_time'] < data_end_date)].copy().sort_values('click_time')
        click_temp['click_date'] = click_temp['click_time'].apply(lambda x: diff_of_days(start_date, x[:10]))
        click_temp['click_hour'] = click_temp['click_time'].str[11:13].astype(int)
        click_last = click_temp.drop_duplicates('uid', keep='last').drop('click_time', axis=1)
        click_last.rename(columns={'click_date': 'click_last_date', 'click_hour': 'click_last_hour',
                                  'pid':'last_pid','param':'last_param'}, inplace=True)
        click_first = click_temp.drop_duplicates('uid', keep='first').drop('click_time', axis=1)
        click_first.rename(columns={'click_date': 'click_first_date', 'click_hour': 'click_first_hour',
                                   'pid': 'first_pid', 'param': 'first_param'}, inplace=True)
        result = user[['uid']].merge(click_last, on='uid', how='left')
        result = result.merge(click_first, on='uid', how='left')
        result = result.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 90天前是否有交易记录
def get_order_history(start_date, n_day):
    result_path = cache_path + 'order_history_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        order_temp = order[(order['buy_time'] >= data_start_date) & (order['buy_time'] < start_date)]
        x_day = diff_of_days(start_date,order['buy_time'].min())+1
        order_temp['money'] = order_temp['price'] * order_temp['qty']
        order_temp = order_temp.groupby(['uid','buy_time'],as_index=False).sum()
        order_temp['money'] = (order_temp['money'] - order_temp['discount']).apply(lambda x:1 if x<1 else x)
        # 每次消费的金额统计
        result = order_temp.groupby('uid', as_index=False)['money'].agg({'sum_order_{}'.format(n_day): lambda x: np.sum(x)/x_day,
                                                                         'sum2_order_{}'.format(n_day): lambda x: np.sum(np.log(x+1))/x_day,
                                                                         'count_order_{}'.format(n_day): lambda x: len(x)/x_day,
                                                                         'mean_order_{}'.format(n_day): 'mean',
                                                                         'mean2_order_{}'.format(n_day): lambda x: np.mean(np.log(x+1)),
                                                                         'median_order_{}'.format(n_day): 'median',
                                                                         'max_order_{}'.format(n_day): 'max',
                                                                         'min_order_{}'.format(n_day): 'min'})
        result2 = order_temp.groupby('uid', as_index=False)['discount'].agg({'sum_discount_{}'.format(n_day): lambda x: np.sum(x)/x_day,
                                                                             'count_discount_{}'.format(n_day): lambda x: len(x)/x_day})
        # 单个商品统计
        result3 = order_temp.groupby('uid', as_index=False)['qty'].agg({'sum_order_qty':lambda x: np.sum(x)/x_day,
                                                                        'max_order_qty': 'max',
                                                                        'mean_order_qty': 'mean'})
        result = user[['uid']].merge(result, on='uid', how='left')
        result = result.merge(result2, on='uid', how='left')
        result = result.merge(result3, on='uid', how='left').drop('uid', axis=1)
        result['discount_rate1'] = result['sum_discount_{}'.format(n_day)] / result['sum_order_{}'.format(n_day)]
        result['discount_rate2'] = result['count_discount_{}'.format(n_day)] / result['count_order_{}'.format(n_day)]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 是否购买过某类商品
def get_cate_count(start_date,n_day):
    result_path = cache_path + 'cate_count_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        order_temp = order[(order['buy_time'] >= data_start_date) & (order['buy_time'] < start_date)]
        cate_count = order_temp.groupby(['uid','cate_id']).size().unstack()
        cate_count = cate_count.add_prefix('{}day_cate_count_'.format(n_day)).reset_index()
        result = user[['uid']].merge(cate_count, on='uid', how='left').fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 50天前是否有浏览记录
def get_click_history(start_date, n_day):
    result_path = cache_path + 'click_history_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        click_temp = click[(click['click_time'] >= data_start_date) & (click['click_time'] < start_date)]
        result = click_temp.groupby('uid', as_index=False)['pid'].agg({'count_click_{}'.format(n_day): 'count',
                                                                       'nunique_pid_{}'.format(n_day): 'nunique'})
        pid = pd.get_dummies(click_temp['pid'], prefix='{}day_pid_'.format(n_day))
        pid = pd.concat([click_temp['uid'], pid], axis=1)
        user_pid_count = pid.groupby(['uid'], as_index=False).sum()
        result = user[['uid']].merge(result, on='uid', how='left')
        result = result.merge(user_pid_count, on='uid', how='left').drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

def get_pid_param_count(start_date,n_day):
    result_path = cache_path + 'pid_param_count_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        click_temp = click[(click['click_time'] >= data_start_date) & (click['click_time'] < start_date)]
        click_temp['pid_param'] = click_temp.apply(lambda x: str(x.pid)+'_'+str(x.param), axis=1)
        user_pid_param_count = click_temp.groupby(['uid','pid_param']).size().unstack()
        user_pid_param_count = user_pid_param_count.add_prefix('{}day_pid_param_'.format(n_day)).reset_index()
        result = user[['uid']].merge(user_pid_param_count, on='uid', how='left').fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户param的次数
def get_param_count(start_date, n_day):
    result_path = cache_path + 'param_count_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        click_temp = click[(click['click_time'] >= data_start_date) & (click['click_time'] < start_date)]
        user_param_count = click_temp.groupby(['uid', 'param']).size().unstack()
        user_param_count = user_param_count.add_prefix('{}day_param_'.format(n_day)).reset_index()
        result = user[['uid']].merge(user_param_count, on='uid', how='left').fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 利用order 计算用户的消费档次
def get_order_level(start_date, n_day):
    result_path = cache_path + 'order_level_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        order_temp = order[(order['buy_time'] >= data_start_date) & (order['buy_time'] < start_date)]
        order_temp = rank(order_temp,'cate_id','price')
        order_temp['max_rank'] = order_temp['cate_id'].map(order_temp.groupby('cate_id')['rank'].max())
        order_temp['level'] = order_temp['rank']/order_temp['max_rank']
        order_temp['level'] = order_temp['level'] * order_temp['price']
        order_temp = order_temp[['uid','level','price']].groupby('uid').sum().reset_index()
        order_temp['level'] = order_temp['level']/order_temp['price']
        order_temp['level'] = 1.35 - (0.8-order_temp['level'])**2*0.7
        result = user[['uid']].merge(order_temp[['uid','level']], on='uid', how='left').fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 二次处理特征
def second_feat(result):
    result['sum_loan_rate_1_2'] = result['sum_loan_{0}day'.format(30)] / result['sum_loan_{0}day'.format(60)]
    result['sum_loan_rate_1_3'] = result['sum_loan_{0}day'.format(30)] / result['sum_loan_{0}day'.format(90)]
    result['return_sum_loan_rate'] = result['user_return{}'.format(3)] / result['sum_loan_{0}day'.format(90)]
    result['return_sum_loan_rate'] = result['user_return{}'.format(3)] / result['sum_loan_{0}day'.format(60)]
    result['order_sum_loan_rate'] = result['sum_order_120'] / result['sum_loan_{0}day'.format(60)]
    result['user_loan_pred'] = result['loan_pred2']**0.89*result['user_pred']**0.11

    return result

# 制作训练集
def make_feats(start_date,n_days):
    t0 = time.time()
    start_date = start_date
    print('数据key为：{}'.format(start_date))
    result_path = cache_path + 'train_set_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        print('开始构造特征...')
        user_feat = get_user_feat(start_date)                       # 用户基本特征
        loan_pred = get_loan_pred(start_date)                       # 借贷概率
        loan_history_120 = get_loan_history(start_date, 120)        # 120天前是否有借款
        loan_history_90 = get_loan_history(start_date, 90)          # 90天前是否有借款
        loan_history_60 = get_loan_history(start_date, 60)          # 60天前是否有借款
        loan_history_30 = get_loan_history(start_date, 30)          # 30天前是否有借款
        # loan_now_1 = get_loan_now(start_date, 1)                  # 本月需还款 以及 剩余还款
        # loan_now_2 = get_loan_now(start_date, 2)                  # 本月需还款 以及 剩余还款
        loan_now_3 = get_loan_now(start_date, 3)                    # 本月需还款 以及 剩余还款
        loan_max = get_loan_max(start_date)                         # 用户最大每月还款额 以及 用户最多的plannum 最活跃的时间
        # loan_window1 = get_loan_window(start_date, 1)             # 每隔n天统计一次3个月
        # loan_window3 = get_loan_window(start_date, 3)             # 每隔n天统计一次3个月
        # loan_window5 = get_loan_window(start_date, 5)             # 每隔n天统计一次3个月
        # loan_window7 = get_loan_window(start_date, 7)             # 每隔n天统计一次3个月
        # loan_window10 = get_loan_window(start_date, 10)           # 每隔n天统计一次3个月
        # loan_window15 = get_loan_window(start_date, 15)           # 每隔n天统计一次3个月
        loan_window30 = get_loan_window(start_date, 30)             # 每隔n天统计一次3个月
        loan_1day_count30 = get_loan_nday_count(start_date, 1, 30)  # 一个月借款多少天
        loan_1day_count60 = get_loan_nday_count(start_date, 1, 60)  # 一个月借款多少天
        loan_1day_count90 = get_loan_nday_count(start_date, 1, 90)  # 一个月借款多少天
        # loan_3day_count30 = get_loan_nday_count(start_date, 3, 30)# 一个月借款多少天
        # loan_3day_count60 = get_loan_nday_count(start_date, 3, 60)# 一个月借款多少天
        # loan_3day_count90 = get_loan_nday_count(start_date, 3, 90)# 一个月借款多少天
        loan_7day_count28 = get_loan_nday_count(start_date, 7, 28)  # 一个月借款多少天
        loan_7day_count56 = get_loan_nday_count(start_date, 7, 56)  # 一个月借款多少天
        loan_7day_count84 = get_loan_nday_count(start_date, 7, 84)  # 一个月借款多少天
        loan_15day_count90 = get_loan_nday_count(start_date, 15, 90)# 一个月借款多少天
        loan_30day_count90 = get_loan_nday_count(start_date, 30, 90)# 一个月借款多少天
        loan_last = get_loan_last(start_date)                       # 最近一次借款，最近一次借款金额，借款类型
        order_last = get_order_last(start_date)                     # 最近一次交易
        click_last = get_click_last(start_date)                     # 最近一次点击
        order_history_120 = get_order_history(start_date, 120)      # 50天前是否有交易记录
        # order_history_60 = get_order_history(start_date, 60)      # 50天前是否有交易记录
        # order_history_30 = get_order_history(start_date, 30)      # 50天前是否有交易记录
        # order_history_15 = get_order_history(start_date, 15)      # 50天前是否有交易记录
        # order_history_7 = get_order_history(start_date, 7)        # 50天前是否有交易记录
        cate_count = get_cate_count(start_date,120)                 # 是否购买过某类商品
        click_history_90 = get_click_history(start_date, 90)        # 30天前是否有浏览记录
        click_history_60 = get_click_history(start_date, 60)        # 30天前是否有浏览记录
        click_history_30 = get_click_history(start_date, 30)        # 30天前是否有浏览记录
        # click_history_7 = get_click_history(start_date, 7)        # 30天前是否有浏览记录
        pid_param_count120 = get_pid_param_count(start_date, 120)   # 用户pid_param次数
        pid_param_count30 = get_pid_param_count(start_date, 30)     # 用户pid_param次数
        # param_count60 = get_param_count(start_date, 60)           # 用户param的次数
        order_level = get_order_level(start_date, 120)              # 利用order 计算用户的消费档次
        # 用户过去一个月的消费是否激增
        # 距离上次贷款之后的点击行为

        print('开始合并特征...')
        result = concat([user_feat,loan_pred,loan_history_120,loan_history_60,loan_history_30,
                         loan_now_3,loan_window30,loan_last,loan_max,loan_15day_count90,loan_30day_count90,
                         click_history_30,click_history_90,click_history_60,order_history_120,
                         loan_1day_count30,loan_1day_count60,loan_1day_count90,order_last,
                         loan_7day_count28,loan_7day_count56,loan_7day_count84,click_last,
                         loan_now_3,
                         pid_param_count120,pid_param_count30,cate_count,order_level,loan_history_90])
        result = second_feat(result)

        print('添加label')
        result = get_label(result,start_date,n_days)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result





delect_id = ['count_plannum6_7','min_plannum3_7','count_discount_60','median_loan_1', 'count_loan_1',
 'sum_plannum3_7','count_plannum3_30','loan_num_sum_1','count_plannum1_120','max_plannum1_120',
 'min_plannum3_1','sum_plannum1_120', 'count_plannum6_1', 'sum_plannum6_1','max_plannum3_1',
 'max_plannum6_1', 'min_plannum6_1','sum_plannum12_1','count_plannum12_1','max_plannum12_1',
 'min_plannum12_1','loan_window_sum30_2','discount_rate2','min_plannum1_120','sum_plannum1_1',
 'count_plannum3_1','sum_plannum3_1', 'count_plannum1_30', 'count_plannum6_60', 'max_plannum1_30',
 'min_plannum1_30','sum_plannum1_7','min_plannum1_60','max_plannum1_60','count_plannum1_60',
 'sum_plannum1_60','count_plannum1_7', 'max_plannum1_7','min_plannum1_7','count_plannum3_7',
 'count_plannum12_7','sum_plannum1_30', 'count_plannum1_1','max_plannum1_1','min_plannum1_1',
'120day_pid_param__10_34','120day_pid_param__10_35',
'120day_pid_param__6_10','120day_pid_param__6_11','120day_pid_param__6_6',
'120day_pid_param__6_7','120day_pid_param__6_8','120day_pid_param__6_9',
'120day_pid_param__7_17','120day_pid_param__7_18','120day_pid_param__7_19',
'120day_pid_param__7_20','120day_pid_param__7_21','120day_pid_param__7_22',
'120day_pid_param__8_8','120day_cate_count_24','120day_pid_param_1_39',
'120day_pid_param_10_34','120day_pid_param_10_35','120day_pid_param_6_10','120day_pid_param_6_11',
'120day_pid_param_6_6','120day_pid_param_6_7','120day_pid_param_6_8','120day_pid_param_6_9',
'120day_pid_param_7_17','120day_pid_param_7_18','120day_pid_param_7_19','120day_pid_param_7_20',
'120day_pid_param_7_21','120day_pid_param_7_22','120day_pid_param_8_8','30day_pid_param_10_34',
'30day_pid_param_10_35','30day_pid_param_1_41','30day_pid_param_6_10','30day_pid_param_6_11',
'30day_pid_param_6_6','30day_pid_param_6_7','30day_pid_param_6_8','30day_pid_param_6_9',
'30day_pid_param_7_17','30day_pid_param_7_18','30day_pid_param_7_19','30day_pid_param_7_20',
'30day_pid_param_7_21','30day_pid_param_7_22','30day_pid_param_8_5','30day_pid_param_8_6',
'30day_pid_param_8_8', '120day_cate_count_4','120day_cate_count_27',
 '120day_pid_param_1_28',
 '30day_pid_param_10_20',
 '120day_pid_param_1_31',
 '120day_pid_param_1_5',
 '120day_pid_param_1_7',
 '120day_pid_param_10_17',
 '120day_pid_param_3_1',
 '120day_pid_param_3_7',
 'count_plannum6_30',
 'count_plannum12_60',
 '30day_pid_param_3_9',
 '120day_pid_param_1_1',
 '30day_pid_param_7_11',
 '120day_pid_param_10_10',
 '30day_pid_param_7_6',
 'max_plannum3_7',
 '120day_pid_param_1_2',
 '120day_pid_param_1_37',
 '120day_pid_param_3_2',
 '120day_pid_param_7_12',
 '30day_pid_param_10_18',
 '30day_pid_param_1_26',
 '30day_pid_param_1_7',
 '30day_pid_param_3_14',
 '30day_pid_param_3_6',
 '120day_pid_param_3_8',
 '120day_pid_param_1_6',
 '30day_pid_param_3_7',
 '120day_pid_param_9_4',
 '30day_pid_param_10_32',
 '120day_pid_param_7_5',
 '120day_pid_param_1_10',
 '120day_pid_param_1_46',
 '120day_pid_param_1_38',
 '120day_pid_param_1_9',
 '120day_pid_param_3_14',
 '120day_pid_param_1_17',
 '120day_cate_count_43',
 '120day_pid_param_8_4',
 '120day_pid_param_2_1',
 '30day_pid_param_1_16',
 '120day_pid_param_10_18',
 '30day_pid_param_2_2',
 '30day_pid_param_1_20',
 '30day_pid_param_1_9',
 '120day_pid_param_10_21',
 '120day_pid_param_2_3',
 '120day_cate_count_17',
 '120day_pid_param_5_11',
 '30day_pid_param_8_4',
 '120day_pid_param_1_20',
 '120day_pid_param_3_11',
 '120day_pid_param_1_15',
 '30day_pid_param_1_38',
 '30day_pid_param_1_29',
 '30day_pid_param_1_23',
 '30day_pid_param_1_22',
 '30day_pid_param_1_33',
 '30day_pid_param_1_32',
 '30day_pid_param_1_31',
 '30day_pid_param_10_3',
 '30day_pid_param_1_24',
 '30day_pid_param_1_25',
 '120day_cate_count_8',
 '30day_pid_param_1_3',
 '30day_pid_param_10_37',
 '30day_pid_param_1_21',
 '30day_pid_param_1_10',
 '30day_pid_param_10_4',
 '30day_pid_param_10_36',
 '30day_pid_param_10_5',
 '30day_pid_param_10_6',
 '30day_pid_param_10_7',
 '30day_pid_param_10_8',
 '30day_pid_param_10_9',
 '30day_pid_param_1_1',
 '30day_pid_param_1_11',
 '30day_pid_param_1_2',
 '30day_pid_param_1_12',
 '30day_pid_param_1_13',
 '30day_pid_param_1_15',
 '30day_pid_param_1_35',
 '30day_pid_param_5_9',
 '30day_pid_param_10_38',
 '30day_pid_param_1_17',
 '30day_pid_param_1_18',
 '120day_pid_param_1_3',
 '30day_pid_param_1_42',
 '30day_pid_param_1_36',
 '30day_pid_param_1_37',
 '30day_pid_param_4_3',
 '30day_pid_param_5_1',
 '30day_pid_param_5_11',
 '30day_pid_param_5_12',
 '30day_pid_param_5_2',
 '120day_cate_count_28',
 '120day_cate_count_29',
 '120day_cate_count_30',
 '120day_cate_count_31',
 '120day_cate_count_32',
 '120day_cate_count_34',
 '30day_pid_param_5_3',
 '120day_cate_count_37',
 '30day_pid_param_5_4',
 '120day_cate_count_39',
 '30day_pid_param_5_5',
 '120day_cate_count_41',
 '120day_cate_count_42',
 '30day_pid_param_5_6',
 '30day_pid_param_4_2',
 '30day_pid_param_3_8',
 '120day_cate_count_20',
 '30day_pid_param_1_5',
 '120day_cate_count_16',
 '30day_pid_param_1_39',
 '30day_pid_param_1_4',
 '30day_pid_param_6_2',
 '30day_pid_param_1_43',
 '30day_pid_param_1_46',
 '30day_pid_param_1_47',
 '30day_pid_param_1_48',
 '30day_pid_param_1_6',
 '30day_pid_param_3_4',
 '30day_pid_param_1_8',
 '30day_pid_param_2_1',
 '30day_pid_param_2_3',
 '30day_pid_param_2_7',
 '30day_pid_param_2_9',
 '30day_pid_param_3_1',
 '30day_pid_param_3_2',
 '30day_pid_param_3_3',
 '30day_pid_param_10_27',
 '30day_pid_param_6_3',
 '120day_cate_count_5',
 '120day_pid_param_10_12',
 '120day_pid_param_10_38',
 '120day_pid_param_10_37',
 '120day_pid_param_10_36',
 '120day_pid_param_1_43',
 '120day_pid_param_1_47',
 '30day_pid_param_7_10',
 '120day_pid_param_10_3',
 '120day_pid_param_1_8',
 '120day_pid_param_10_27',
 '30day_pid_param_7_12',
 '120day_pid_param_10_25',
 '120day_pid_param_10_24',
 '120day_pid_param_10_23',
 '120day_pid_param_10_22',
 '120day_pid_param_10_20',
 '120day_pid_param_10_2',
 '120day_pid_param_10_19',
 '120day_pid_param_10_15',
 '120day_pid_param_10_14',
 '120day_pid_param_10_4',
 '120day_pid_param_10_5',
 '120day_pid_param_10_6',
 '120day_pid_param_1_35',
 '120day_pid_param_1_32',
 '120day_pid_param_1_27',
 '120day_pid_param_1_33',
 '120day_pid_param_1_25',
 '120day_pid_param_1_24',
 '120day_pid_param_1_23',
 '120day_pid_param_1_22',
 '120day_pid_param_1_21',
 '120day_pid_param_1_36',
 '120day_pid_param_10_7',
 '120day_pid_param_1_18',
 '30day_pid_param_5_7',
 '120day_pid_param_1_41',
 '120day_pid_param_1_12',
 '120day_pid_param_1_11',
 '120day_pid_param_1_42',
 '120day_pid_param_10_9',
 '120day_pid_param_10_8',
 '120day_pid_param_10_13',
 '120day_pid_param_10_11',
 '30day_pid_param_10_25',
 '120day_pid_param_2_9',
 '120day_pid_param_7_10',
 '30day_pid_param_6_4',
 '120day_pid_param_8_5',
 '120day_pid_param_8_6',
 '120day_pid_param_8_7',
 '30day_pid_param_10_1',
 '30day_pid_param_10_10',
 '30day_pid_param_10_13',
 '30day_pid_param_10_14',
 '30day_pid_param_10_15',
 '120day_pid_param_1_29',
 '30day_pid_param_10_17',
 '30day_pid_param_10_19',
 '30day_pid_param_10_2',
 '30day_pid_param_10_21',
 '30day_pid_param_10_22',
 '30day_pid_param_9_4',
 '30day_pid_param_10_23',
 '30day_pid_param_10_24',
 '120day_pid_param_6_4',
 '120day_pid_param_6_3',
 '120day_pid_param_6_2',
 '120day_pid_param_4_3',
 '30day_pid_param_7_14',
 '30day_pid_param_7_15',
 '120day_pid_param_3_15',
 '120day_pid_param_3_17',
 '30day_pid_param_7_5',
 '120day_pid_param_3_3',
 '120day_pid_param_3_4',
 '120day_pid_param_4_2',
 '120day_pid_param_5_1',
 '120day_pid_param_5_9',
 '120day_pid_param_5_12',
 '120day_pid_param_5_2',
 '120day_pid_param_5_3',
 '120day_pid_param_5_4',
 '120day_pid_param_5_5',
 '120day_pid_param_5_6',
 '120day_pid_param_5_7',
 '120day_pid_param_5_8',
 '30day_pid_param_5_8']












