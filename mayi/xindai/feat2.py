import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
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
# order['price'] = np.round(5**(order['price'])-1,2)
# order['discount'] = np.round(5**(order['discount'])-1,2)
loan['loan_amount'] = np.round(5**(loan['loan_amount'])-1,2)
loan_sum['loan_sum'] = np.round(5**(loan_sum['loan_sum'])-1,2)

# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 日期的加减
def date_add_days(start_date, days):
    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

# 相差的日期数
def diff_of_days(day1, day2):
    days = (datetime.strptime(day1, '%Y-%m-%d') - datetime.strptime(day2, '%Y-%m-%d')).days
    return abs(days)

# 相差的分钟数
def diff_of_minutes(time1,time2):
    minutes = (datetime.strptime(time1, '%Y-%m-%d %H:%M:%S') - datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')).total_seconds()//60
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

###################################################
#..................... 构造特征 ....................
###################################################
# 用户基本特征
def get_user_feat(start_date):
    result = user.copy()
    result['week'] =  pd.to_datetime(user['active_date']).dt.weekday
    result['dayofmonth'] = pd.to_datetime(user['active_date']).dt.days_in_month
    result['month'] = pd.to_datetime(user['active_date']).dt.month
    date_n_user = result.groupby('active_date', as_index=False)['uid'].agg({'date_n_user': 'count'})
    result = result.merge(date_n_user,on='active_date',how='left')
    result['active_date'] = result['active_date'].apply(lambda x:diff_of_days('2016-11-01',x))
    return result

# 不同注册日期用户的转化率
def get_date_rate(start_date):
    result_path = cache_path + 'date_rate_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_end_date = start_date
        data_start_date = date_add_days(start_date, (0 - 30))
        loan_temp = loan[(loan['loan_time'] < data_end_date) & (loan['loan_time'] > data_start_date)]
        pos_id = loan_temp['uid'].unique().tolist()
        user_temp = user[['uid','active_date']].copy()
        user_temp['label'] = user_temp['uid'].apply(lambda x:1 if x in pos_id else 0)
        date_rate = user_temp.groupby('active_date')['label'].agg({'user_count':'count',
                                                                   'neg_count':'sum'})
        date_rate['date_rate'] = date_rate['neg_count'] / date_rate['user_count']
        user_temp['date_rate'] = user_temp['active_date'].map(date_rate['date_rate'])
        result = user_temp[['date_rate']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 30天前是否有借款几率
def get_loan_history(start_date,n_day):
    data_start_date = date_add_days(start_date,(0 - n_day))
    data_end_date = start_date
    loan_temp = loan[(loan['loan_time']>=data_start_date) & (loan['loan_time']<data_end_date)]
    result = loan_temp.groupby('uid',as_index=False)['loan_amount'].agg({'sum_loan_{}'.format(n_day):'sum',
                                                                         'count_loan_{}'.format(n_day):'count',
                                                                         'mean_loan_{}'.format(n_day): 'mean',
                                                                         'median_loan_{}'.format(n_day): 'median',
                                                                         'max_loan_{}'.format(n_day): 'max',
                                                                         'min_loan_{}'.format(n_day): 'min'})
    result = user[['uid']].merge(result, on='uid', how='left')
    loan_num_sum = loan_temp.groupby('uid',as_index=False)['plannum'].agg({'loan_num_sum_{}'.format(n_day):'sum'})
    result = result.merge(loan_num_sum, on='uid', how='left')
    for i in [1,3,6,12]:
        loan_temp_sum = loan_temp[loan_temp['plannum']==i]
        result_sub = loan_temp_sum[loan_temp['plannum'] > 1].groupby(['uid'], as_index=False)['loan_amount'].agg(
            {'sum_plannum{}_{}'.format(i,n_day): 'sum',
             'count_plannum{}_{}'.format(i,n_day): 'count',
             'max_plannum{}_{}'.format(i,n_day): 'max',
             'min_plannum{}_{}'.format(i,n_day): 'min'})
        result = result.merge(result_sub, on='uid', how='left')
    result = result.drop('uid',axis=1).fillna(0)
    return result

# 本月需还款 以及 剩余还款
def get_loan_now(start_date,n_month):
    user_new = pd.DataFrame({'uid':user['uid'].values,'user_owe':0,'user_return':0})
    loan_temp = loan.copy()
    loan_temp['return_permonth'] = loan_temp['loan_amount'] / loan_temp['plannum'] #每月还款
    for i in range(n_month):
        data_start_date = date_add_days(start_date, (0 - i - 1) * 30)
        data_end_date = date_add_days(start_date, (0 - i) * 30)
        loan_temp = loan_temp[(loan_temp['loan_time'] >= data_start_date) & (loan_temp['loan_time'] < data_end_date)]
        loan_temp = loan_temp[loan_temp['plannum'] > i]
        loan_temp['owe_permonth'] = loan_temp['loan_amount'] - (i*loan_temp['return_permonth'])
        user_return_permonth = loan_temp.groupby('uid')['return_permonth'].sum().to_dict()
        user_new['user_return_permonth'] = user_new['uid'].map(user_return_permonth).fillna(0)
        user_new['user_return'] = user_new['user_return'] + user_new['user_return_permonth']
        user_loan_amount = loan_temp.groupby('uid')['loan_amount'].sum().to_dict()
        user_new['user_loan_amount'] = user_new['uid'].map(user_loan_amount).fillna(0)
        user_new['user_owe'] = user_new['user_owe'] + user_new['user_loan_amount']
        user_new.drop(['user_return_permonth','user_loan_amount'],axis=1,inplace=True)
    user_new.rename(columns={'user_owe':'user_owe{}'.format(n_month),
                             'user_return':'user_return{}'.format(n_month)},inplace=True)
    user_new['user_return_rate{}'.format(n_month)] = user_new['user_return{}'.format(n_month)] / user_new['user_owe{}'.format(n_month)]
    result = user_new.drop('uid', axis=1)
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

# 用户最近一次借款
def get_loan_last(start_date):
    data_end_date = start_date
    loan_temp = loan[(loan['loan_time'] < data_end_date)].copy().sort_values('loan_time')
    loan_temp = rank(loan_temp,'uid','loan_time',False)
    loan_temp['loan_date'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date, x[:10]))
    loan_temp['loan_hour'] = loan_temp['loan_time'].str[11:13].astype(int)
    loan_last = loan_temp[loan_temp['rank'] == 0].copy()
    loan_last2 = loan_temp[loan_temp['rank'] == 1].copy()
    loan_last = loan_last.rename(columns={'loan_date':'loan_last_date','loan_hour':'loan_last_hour',
                              'loan_amount':'last_loan_amount','plannum':'last_plannum'}).drop(['rank','loan_time'],axis=1)
    loan_last2 = loan_last2.rename(columns={'loan_date': 'loan_last_date2', 'loan_hour': 'loan_last_hour2',
                              'loan_amount': 'last_loan_amount2', 'plannum': 'last_plannum2'}).drop(['rank','loan_time'], axis=1)
    result = user[['uid']].merge(loan_last, on='uid', how='left')
    result = result.merge(loan_last2, on='uid', how='left')
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
        order_temp['money'] = order_temp['price'] * order_temp['qty']
        order_temp = order_temp.groupby(['uid','buy_time'],as_index=False).sum()
        order_temp['money'] = order_temp['money'] - order_temp['discount']
        result = order_temp.groupby('uid', as_index=False)['money'].agg({'sum_order_{}'.format(n_day): 'sum',
                                                                         'count_order_{}'.format(n_day): 'count',
                                                                         'mean_order_{}'.format(n_day): 'mean',
                                                                         'median_order_{}'.format(n_day): 'median',
                                                                         'max_order_{}'.format(n_day): 'max',
                                                                         'min_order_{}'.format(n_day): 'min'})
        result2 = order_temp.groupby('uid', as_index=False)['discount'].agg({'sum_discount_{}'.format(n_day): 'sum',
                                                                              'count_discount_{}'.format(n_day): 'count' })
        result = user[['uid']].merge(result, on='uid', how='left')
        result = result.merge(result2, on='uid', how='left').drop('uid', axis=1)
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
        result = user[['uid']].merge(order_temp[['uid','level']], on='uid', how='left').fillna(0)
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
        date_rate = get_date_rate(start_date)                       # 不同注册日期用户的转化率
        loan_history_120 = get_loan_history(start_date, 120)        # 120天前是否有借款
        # loan_history_90 = get_loan_history(start_date, 90)        # 90天前是否有借款
        loan_history_60 = get_loan_history(start_date, 60)          # 60天前是否有借款
        loan_history_30 = get_loan_history(start_date, 30)          # 30天前是否有借款
        # loan_history_15 = get_loan_history(start_date, 15)        # 15天前是否有借款
        loan_history_7 = get_loan_history(start_date, 7)            # 7天前是否有借款
        # loan_history_4 = get_loan_history(start_date, 4)          # 4天前是否有借款
        loan_history_1 = get_loan_history(start_date, 1)            # 1天前是否有借款
        # loan_now_1 = get_loan_now(start_date, 1)                  # 本月需还款 以及 剩余还款
        # loan_now_2 = get_loan_now(start_date, 2)                  # 本月需还款 以及 剩余还款
        loan_now_3 = get_loan_now(start_date, 3)                    # 本月需还款 以及 剩余还款
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
        loan_last = get_loan_last(start_date)                       # 最近一次借款，最近一次借款金额，借款类型
        # order_last = get_order_last(start_date)                     # 最近一次交易
        click_last = get_click_last(start_date)                     # 最近一次点击
        # order_history_90 = get_order_history(start_date, 90)        # 50天前是否有交易记录
        # order_history_60 = get_order_history(start_date, 60)        # 50天前是否有交易记录
        # order_history_30 = get_order_history(start_date, 30)        # 50天前是否有交易记录
        # order_history_15 = get_order_history(start_date, 15)      # 50天前是否有交易记录
        # order_history_7 = get_order_history(start_date, 7)          # 50天前是否有交易记录
        cate_count = get_cate_count(start_date,120)                 # 是否购买过某类商品
        click_history_90 = get_click_history(start_date, 90)        # 30天前是否有浏览记录
        click_history_60 = get_click_history(start_date, 60)        # 30天前是否有浏览记录
        click_history_30 = get_click_history(start_date, 30)        # 30天前是否有浏览记录
        # click_history_7 = get_click_history(start_date, 7)        # 30天前是否有浏览记录
        pid_param_count120 = get_pid_param_count(start_date, 120)   # 用户pid_param次数
        pid_param_count30 = get_pid_param_count(start_date, 30)     # 用户pid_param次数
        # param_count60 = get_param_count(start_date, 60)               # 用户param的次数
        order_level = get_order_level(start_date, 120)              # 利用order 计算用户的消费档次
        # 用户过去一个月的消费是否激增
        # 距离上次贷款之后的点击行为

        print('开始合并特征...')
        result = concat([user_feat,loan_history_120,loan_history_60,loan_history_30,loan_history_7,
                         loan_history_1,loan_now_3,loan_window30,loan_last,date_rate,
                         click_history_30,click_history_90,click_history_60,
                         loan_1day_count30,loan_1day_count60,loan_1day_count90,
                         loan_7day_count28,loan_7day_count56,loan_7day_count84,click_last,
                         pid_param_count120,pid_param_count30,cate_count,order_level])
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












