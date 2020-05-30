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
load = 1

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
    result = result.merge(label[['uid', 'loan_sum']], on='uid', how='left')
    result['loan_sum'].fillna(0,inplace=True)
    return result

def make_last_month_label(start_date):
    last_month_start_days = date_add_days(start_date, - 30)
    loan_temp = loan[(loan['loan_time'] < start_date) & (loan['loan_time'] >= last_month_start_days)]
    loan_temp['loan_amount'] = np.log1p(loan_temp['loan_amount']) / np.log(5)
    last_month_label = loan_temp.groupby('uid',as_index=False)['loan_amount'].agg({'last_month_loan_sum':'sum'})
    last_month_label = user.merge(last_month_label,on='uid', how='left').fillna(0)
    last_month_label['week'] = pd.to_datetime(last_month_label['active_date']).dt.dayofweek
    return last_month_label

# 获取转化率
def get_rate(data,name,label='loan_sum'):
    rate = data.groupby(name)[label].agg({'count': 'count', 'sum': 'sum'})
    rate['mean'] = rate['sum'] / rate['count']
    return rate['mean']

###################################################
#..................... 构造特征 ....................
###################################################
# 用户基本特征
def get_user_feat(start_date):
    result_path = cache_path + 'user_feat_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        last_month_label = make_last_month_label('2016-09-03')
        result = user.copy()
        result['week'] = pd.to_datetime(result['active_date']).dt.dayofweek
        result['month'] = pd.to_datetime(result['active_date']).dt.month
        result['date_rate'] = result['active_date'].map(get_rate(last_month_label,'active_date',label='last_month_loan_sum'))
        result['week_rate'] = result['week'].map(get_rate(last_month_label, 'week', label='last_month_loan_sum'))
        result['limit_rate'] = result['limit'].map(get_rate(last_month_label, 'limit', label='last_month_loan_sum'))
        result['date_n_people'] = result['active_date'].map(result['active_date'].value_counts())
        result['age_rate'] = result['age'].map(get_rate(last_month_label, 'age', label='last_month_loan_sum'))
        result['active_date'] = result['active_date'].apply(lambda x: diff_of_days(x, '2016-11-01'))
        result['user_pred'] = result['age_rate'] * (result['week_rate'] + result['date_rate'] * 25) * result['limit_rate'] ** 0.4
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 30天前是否有借款几率
def get_loan_history(start_date,n_day):
    result_path = cache_path + 'loan_history_{0}_{1}.hdf'.format(start_date,n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date,(0 - n_day))
        data_end_date = start_date
        loan_temp = loan[(loan['loan_time']>=data_start_date) & (loan['loan_time']<data_end_date)]
        x_day = diff_of_days(start_date,loan_temp['loan_time'].min())
        loan_temp['return_permonth'] = loan_temp['loan_amount'] / loan_temp['plannum']
        loan_stat = loan_temp.groupby('uid',as_index=False)['loan_amount'].agg({
            'sum_loan_{}day'.format(n_day): 'sum',
            'sum2_loan_{}day'.format(n_day): lambda x: sum(np.log(x+1))/x_day,
            'sum_loan_{}day_inverse'.format(n_day): lambda x: sum(1/np.log(x + 1))/x_day,
            'count_loan_{}'.format(n_day): lambda x: len(x)/x_day,
            'mean_loan_{}'.format(n_day): 'mean',
            'mean2_loan_{}'.format(n_day):  lambda x: np.mean(1/np.log(x + 1)),
            'median_loan_{}'.format(n_day): 'median',
            'max_loan_{}'.format(n_day): 'max',
            'min_loan_{}'.format(n_day): 'min',
            'std_loan_{}'.format(n_day): 'std',
            'std2_loan_{}'.format(n_day): lambda x: np.std(np.log(x + 1))
        })
        loan_stat['user_nunique_plannum{}'.format(n_day)] = loan_stat['uid'].map(loan_temp.groupby('uid')['plannum'].nunique()).fillna(0)
        loan_stat['user_sum_plannum{}'.format(n_day)] = loan_stat['uid'].map(loan_temp.groupby('uid')['plannum'].sum()).fillna(0)
        loan_stat['user_loan_nday{}'.format(n_day)] = loan_stat['uid'].map(loan_temp.groupby('uid')['loan_time'].nunique()).fillna(0)/x_day
        loan_stat['user_loan_plannum>1{}'.format(n_day)] = loan_stat['uid'].map(loan_temp[loan_temp['plannum']>1].groupby('uid').size()).fillna(0) / x_day
        loan_stat['user_return_pernum{}'.format(n_day)] = loan_stat['sum_loan_{0}day'.format(n_day)]/loan_stat['user_sum_plannum{}'.format(n_day)]
        loan_stat['user_return_2pernum{}'.format(n_day)] = loan_stat['sum2_loan_{0}day'.format(n_day)] / loan_stat[ 'user_sum_plannum{}'.format(n_day)]
        result = user[['uid']].merge(loan_stat,on='uid',how='left').fillna(0)
        result = result.drop('uid',axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户前期借款总额的指数和
def get_sum_loan_exp(start_date, n_day):
    result_path = cache_path + 'sum_loan_exp_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        data_end_date = start_date
        loan_temp = loan[(loan['loan_time'] >= data_start_date) & (loan['loan_time'] < data_end_date)]
        x_day = diff_of_days(start_date, loan_temp['loan_time'].min())
        result = user[['uid']].copy()
        loan_temp['loan_amount'] = np.log(loan_temp['loan_amount']+1)/np.log(5)
        loan_temp['loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date, x))
        for i in [0.999,0.995,0.99,0.95,0.9,0.85,0.7]:
            loan_temp['weight'] = i**loan_temp['loan_time']
            loan_temp['weight'] = loan_temp['weight'] * loan_temp['loan_amount']
            result['sum_loan_{0}day_{1}w'.format(n_day,i)] = result['uid'].map(loan_temp.groupby('uid')['weight'].sum())/((1-i**n_day)/(1-i))
        result = result.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户最近一次借款
def get_loan_last(start_date):
    result_path = cache_path + 'loan_last_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
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
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 一个月借款多少天
def get_loan_nday_count(start_date, n_day, period):
    result_path = cache_path + 'loan_nday_count_{0}_{1}_{2}.hdf'.format(start_date,n_day,period)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - period))
        data_end_date = start_date
        loan_temp = loan[(loan['loan_time'] >= data_start_date) & (loan['loan_time'] < data_end_date)].copy()
        x_day = diff_of_days(start_date, loan_temp['loan_time'].min())
        loan_temp['loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date,x[:10])//n_day)
        loan_day_count = loan_temp.groupby('uid',as_index=False)['loan_time'].agg({
            'loan_{0}day_count{1}'.format(n_day, period):lambda x: x.nunique()/x_day})
        result  = user[['uid']].merge(loan_day_count,on='uid',how='left')
        result = result.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户最大每月还款额 以及 用户最多的plannum 最活跃的时间
def get_loan_max(start_date):
    result_path = cache_path + 'loan_nday_count_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        loan_temp = loan[(loan['loan_time'] < start_date)].copy().sort_values('loan_time')
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

# 本月需还款 以及 剩余还款
def get_loan_now(start_date,n_month):
    result_path = cache_path + 'pid_loan_now_{0}_{1}.hdf'.format(start_date, n_month)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_new = pd.DataFrame({'uid':user['uid'].values,'user_1owe{}'.format(n_month):0,'user_return{}'.format(n_month):0,
                                 'user_return_count{}'.format(n_month):0,'user_owe_count{}'.format(n_month):0,
                                 'user_2owe{}'.format(n_month): 0,'user_new_loan{}'.format(n_month): 0,
                                 'user_new_loan_count{}'.format(n_month):0})
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

            user_own_permonth1 = loan_temp.groupby('uid')['loan_amount'].sum()
            user_new['user_1owe{}'.format(n_month)] += user_new['uid'].map(user_own_permonth1).fillna(0)

            user_own_permonth2 = loan_temp.groupby('uid')['owe_permonth'].sum()
            user_new['user_2owe{}'.format(n_month)] += user_new['uid'].map(user_own_permonth2).fillna(0)
            user_new['user_owe_count{}'.format(n_month)] += (~user_new['uid'].map(user_own_permonth2).fillna(0).isnull())

            loan_temp = loan_temp[loan_temp['plannum'] == (i+1)]
            user_new_loan = loan_temp.groupby('uid')['loan_amount'].sum()
            user_new['user_new_loan{}'.format(n_month)] += user_new['uid'].map(user_new_loan).fillna(0)
            user_new['user_new_loan_count{}'.format(n_month)] += (~user_new['uid'].map(user_new_loan).fillna(0).isnull())

        user_new['user_return_rate{}'.format(n_month)] = user_new['user_return{}'.format(n_month)] / user_new['user_1owe{}'.format(n_month)]
        result = user_new.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 是否购买过某类商品
def get_cate_count(start_date,n_day):
    result_path = cache_path + 'cate_count_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        order_temp = order[(order['buy_time'] >= data_start_date) & (order['buy_time'] < start_date)]
        x_day = diff_of_days(start_date, order_temp['buy_time'].min())
        cate_count = order_temp.groupby(['uid','cate_id']).size().unstack()/x_day
        cate_count = cate_count.add_prefix('{}day_cate_count_'.format(n_day)).reset_index()
        result = user[['uid']].merge(cate_count, on='uid', how='left').fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 50天前是否有浏览记录
def get_click_history(start_date, n_day):
    result_path = cache_path + 'click_history_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        click_temp = click[(click['click_time'] >= data_start_date) & (click['click_time'] < start_date)]
        x_day = diff_of_days(start_date, click_temp['click_time'].min())
        result = click_temp.groupby('uid', as_index=False)['pid'].agg({'count_click_{}'.format(n_day): lambda x: len(x)/x_day,
                                                                       'nunique_pid_{}'.format(n_day): 'nunique'})
        pid = pd.get_dummies(click_temp['pid'], prefix='{}day_pid_'.format(n_day))
        pid = pd.concat([click_temp['uid'], pid], axis=1)
        user_pid_count = pid.groupby(['uid'], as_index=False).sum()
        result = user[['uid']].merge(result, on='uid', how='left')
        result = result.merge(user_pid_count, on='uid', how='left').drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户pid_param次数
def get_pid_param_count(start_date,n_day):
    result_path = cache_path + 'pid_param_count_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        click_temp = click[(click['click_time'] >= data_start_date) & (click['click_time'] < start_date)]
        x_day = diff_of_days(start_date, click_temp['click_time'].min())
        click_temp['pid_param'] = click_temp['pid'].astype(str) + '_' + click_temp['param'].astype(str)
        user_pid_param_count = click_temp.groupby(['uid','pid_param']).size().unstack()/x_day
        user_pid_param_count = user_pid_param_count.add_prefix('{}day_pid_param_'.format(n_day)).reset_index()
        result = user[['uid']].merge(user_pid_param_count, on='uid', how='left').fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 90天前是否有交易记录
def get_order_history(start_date, n_day):
    result_path = cache_path + 'order_history_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & load:
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

# 二次处理特征
def second_feat(result):

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
        loan_history_120 = get_loan_history(start_date, 120)        # 120天前是否有借款
        loan_history_60 = get_loan_history(start_date, 60)          # 120天前是否有借款
        loan_history_30 = get_loan_history(start_date, 30)          # 120天前是否有借款
        loan_history_7 = get_loan_history(start_date, 7)            # 120天前是否有借款
        sum_loan_exp = get_sum_loan_exp(start_date, 120)            # 用户前期借款总额的指数和
        loan_last = get_loan_last(start_date)                       # 最近一次借款，最近一次借款金额，借款类型
        loan_1day_count120 = get_loan_nday_count(start_date, 1, 120)# 一个月借款多少天
        loan_30day_count90 = get_loan_nday_count(start_date, 30, 90)# 一个月借款多少天
        loan_max = get_loan_max(start_date)                         # 用户最大每月还款额 以及 用户最多的plannum 最活跃的时间
        loan_now_3 = get_loan_now(start_date, 3)                    # 本月需还款 以及 剩余还款
        cate_count = get_cate_count(start_date, 120)                # 是否购买过某类商品
        click_history_120 = get_click_history('2016-10-02', 120)    # 30天前是否有浏览记录
        pid_param_count120 = get_pid_param_count(start_date, 120)   # 用户pid_param次数
        pid_param_count30 = get_pid_param_count(start_date, 30)     # 用户pid_param次数
        order_history_120 = get_order_history(start_date, 120)      # 120天前是否有交易记录

        print('开始合并特征...')
        result = concat([user_feat,loan_history_120,loan_history_60,loan_history_30,
                         loan_history_7,loan_last,loan_1day_count120,pid_param_count120,
                         loan_30day_count90,loan_max,loan_now_3,pid_param_count30,
                         click_history_120,sum_loan_exp,cate_count,order_history_120])
        result = second_feat(result)

        print('添加label')
        result = get_label(result,start_date,n_days)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result










delect_id = ['120day_pid_param_8_8','120day_pid_param_7_22','120day_pid_param_7_21',
             '120day_pid_param_7_20','120day_pid_param_7_19','120day_pid_param_7_18',
             '120day_pid_param_7_17','120day_pid_param_6_9','120day_pid_param_6_8',
             '120day_pid_param_6_7','120day_pid_param_6_6','120day_pid_param_6_11',
             '120day_pid_param_6_10','120day_pid_param_10_35','120day_pid_param_10_34',
             '60day_pid_param_8_8' '60day_pid_param_8_6' '60day_pid_param_8_5',
             '60day_pid_param_7_22','60day_pid_param_7_21','60day_pid_param_7_20',
             '60day_pid_param_7_19','60day_pid_param_7_18','60day_pid_param_7_17',
             '60day_pid_param_6_9','60day_pid_param_6_8','60day_pid_param_6_7',
             '60day_pid_param_6_6','60day_pid_param_6_11','60day_pid_param_6_10',
             '60day_pid_param_8_8','60day_pid_param_8_6','60day_pid_param_8_5',
             '60day_pid_param_10_35','60day_pid_param_10_34','30day_pid_param_8_8',
             '30day_pid_param_8_6','30day_pid_param_8_5','30day_pid_param_7_22',
             '30day_pid_param_7_21','30day_pid_param_7_20','30day_pid_param_7_19',
             '30day_pid_param_7_18','30day_pid_param_7_17','30day_pid_param_6_9',
             '30day_pid_param_6_8','30day_pid_param_6_7','30day_pid_param_6_6',
             '30day_pid_param_6_11','30day_pid_param_6_10','30day_pid_param_1_41',
             '30day_pid_param_10_35','30day_pid_param_10_34','30day_pid_param_10_5',
             '30day_pid_param_2_9', '30day_pid_param_5_2','120day_cate_count_24',
             '120day_pid_param_5_7', '120day_pid_param_5_8', '120day_pid_param_5_9',
            '120day_pid_param_6_2', '120day_pid_param_10_7', '120day_pid_param_8_5',
           '30day_pid_param_1_5', '120day_pid_param_8_6', '30day_pid_param_10_1',
           '30day_pid_param_10_10', '120day_pid_param_10_5',
           '120day_pid_param_10_25', '120day_pid_param_10_24',
           '120day_pid_param_5_3', '30day_pid_param_5_3', '30day_pid_param_5_5',
           '30day_pid_param_5_6', '120day_pid_param_5_2', '30day_pid_param_5_7',
           '30day_pid_param_5_8', '30day_pid_param_5_9', '30day_pid_param_6_2',
           '120day_pid_param_10_15', '30day_pid_param_6_4',
           '120day_pid_param_10_13', '120day_pid_param_10_12',
           '120day_pid_param_10_11', '30day_pid_param_7_10',
           '30day_pid_param_10_13', '120day_pid_param_10_8',
           '30day_pid_param_10_6', '120day_pid_param_1_41', '30day_pid_param_1_21',
           '30day_pid_param_10_25', '30day_pid_param_10_3',
           '120day_pid_param_1_43', '120day_pid_param_1_42',
           '30day_pid_param_1_13', '30day_pid_param_1_11', '30day_pid_param_1_43',
           '30day_pid_param_1_10', '120day_pid_param_1_33', '30day_pid_param_10_9',
           '30day_pid_param_10_8', '120day_pid_param_1_36', '30day_pid_param_10_7',
           '120day_pid_param_1_24', '30day_pid_param_10_24',
           '120day_pid_param_3_15', '30day_pid_param_1_22', '30day_pid_param_1_23',
           '30day_pid_param_1_24', '30day_pid_param_10_23', '30day_pid_param_1_29',
           '30day_pid_param_1_3', '30day_pid_param_10_2', '30day_pid_param_1_33',
           '30day_pid_param_1_35', '30day_pid_param_1_36', '30day_pid_param_10_15',
           '120day_pid_param_1_11', '30day_pid_param_10_14',
           '30day_pid_param_1_42', '120day_pid_param_3_17',
             '30day_pid_param_3_3', '30day_pid_param_3_4', '120day_pid_param_1_29',
             'count_loan_60', '120day_pid_param_1_23', '30day_pid_param_3_14',
             '120day_pid_param_1_35', 'user_nunique_plannum7', '30day_pid_param_4_2',
             '30day_pid_param_7_5', '120day_pid_param_5_12', '30day_pid_param_8_4',
             '120day_pid_param_5_11', '120day_pid_param_5_6', '120day_pid_param_5_1',
             '120day_pid_param_4_3', '120day_pid_param_6_4', '30day_pid_param_4_3',
             '120day_pid_param_2_9', '120day_pid_param_7_10', '30day_pid_param_6_3',
             '30day_pid_param_5_4', '30day_pid_param_5_12', '30day_pid_param_5_1',
             '120day_pid_param_1_22', '30day_pid_param_2_3', '120day_pid_param_1_18',
             '30day_pid_param_1_31', '120day_pid_param_10_14',
             'user_new_loan_count3', 'user_owe_count3', 'user_return_count3',
             '30day_pid_param_10_19', '30day_pid_param_1_2', 'count_loan_7',
             '120day_pid_param_5_4', '30day_pid_param_10_21',
             '30day_pid_param_10_22', '30day_pid_param_1_17',
             '30day_pid_param_10_27', '30day_pid_param_10_36',
             '30day_pid_param_10_37', '30day_pid_param_10_38',
             '30day_pid_param_1_15', '30day_pid_param_10_4', '120day_pid_param_10_2',
             '30day_pid_param_1_37', '120day_pid_param_1_15', '30day_pid_param_1_38',
             '120day_pid_param_1_12', '120day_pid_param_10_9', '30day_pid_param_2_1',
             '120day_pid_param_10_6', '120day_pid_param_10_4',
             '120day_pid_param_10_38', '120day_pid_param_10_37',
             '30day_pid_param_1_8', '120day_pid_param_10_36', '30day_pid_param_1_6',
             '120day_pid_param_8_7', '30day_pid_param_1_47', '120day_pid_param_10_3',
             '30day_pid_param_1_46', '120day_pid_param_10_27',
             '120day_pid_param_10_23', '30day_pid_param_1_4',
             '120day_pid_param_5_5','120day_pid_param_3_4',
             '120day_cate_count_8', '120day_cate_count_20',
           '120day_pid_param_4_2', '120day_cate_count_16', '30day_pid_param_3_7',
           '120day_cate_count_28', '30day_pid_param_1_1', '120day_cate_count_29',
           '120day_cate_count_30', '120day_cate_count_31', '120day_cate_count_32',
           '120day_cate_count_34', '120day_cate_count_37', '120day_pid_param_6_3',
           '120day_cate_count_39', '120day_cate_count_41', '120day_cate_count_42',
           '120day_pid_param_7_5', '120day_pid_param_3_3', '30day_pid_param_9_4',
           'user_loan_nday120', '120day_cate_count_5', '30day_pid_param_5_11',
           '30day_pid_param_3_2', 'user_loan_nday7', '30day_pid_param_3_1',
           '30day_pid_param_2_7', '30day_pid_param_1_9', '30day_pid_param_1_48',
           '30day_pid_param_1_39', '120day_pid_param_10_10',
           '30day_pid_param_7_14', '30day_pid_param_1_32', '120day_pid_param_9_4',
           '30day_pid_param_1_25', '120day_pid_param_1_20',
           '120day_pid_param_1_21', '120day_pid_param_1_3', '30day_pid_param_1_18',
           '120day_pid_param_1_46', '120day_pid_param_1_47',
           '120day_pid_param_1_8', '120day_pid_param_2_1', 'discount_rate2',
            '120day_cate_count_27', '120day_cate_count_4'
             ]



import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2016-11-01'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),30).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)
# train_feat = make_feats('2016-08-15').fillna(-1)
# test_feat = make_feats('2016-11-03').fillna(-1)

# train_feat['date_rate'] = train_feat['date_rate']/train_feat['date_rate'].mean()*1.3
# test_feat['date_rate'] = test_feat['date_rate']/test_feat['date_rate'].mean()
predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum']+delect_id)]
predictors = list(reversed(predictors))

def evalerror(pred, df):
    label = df.get_label().values.copy()
    label_mean = np.mean(label)
    label = label - label_mean
    pred_temp = np.array(pred.copy())
    pred_mean = np.mean(pred_temp)
    pred_temp = pred_temp - pred_mean
    rmse = mean_squared_error(label,pred_temp)**0.5
    return ('rmse',rmse,False)

params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_preds = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    gbm = lgb.train(params, lgb_train, 750)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds += test_preds_sub

test_preds = test_preds/5
print('CV训练用时{}秒'.format(time.time() - t0))
pred_mean = np.mean(test_preds)
print(pred_mean)
submission = pd.DataFrame({'uid':test_feat.uid.values,'pred':test_preds/pred_mean*1.2575})[['uid','pred']]
submission['pred'] = submission['pred'].apply(lambda x: x if x>0.1 else 0.1)
submission.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, header=None, float_format='%.4f')