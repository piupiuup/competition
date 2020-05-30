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
flag = 0

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


# 获取标签
def get_label(result, start_date):
    result_path = cache_path + 'label_{}.hdf'.format(start_date)
    if os.path.exists(result_path) & flag:
        label = pd.read_hdf(result_path, 'w')
    else:
        label_start_date = date_add_days(start_date, 30)
        label = loan[(loan['loan_time']<label_start_date) & (loan['loan_time']>=start_date)]
        label = label.groupby('uid',as_index=False)['loan_amount'].agg({'loan_sum':'sum'})
        label.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    result = result.merge(label[['uid', 'loan_sum']], how='left')
    result['loan_sum'] = np.log1p(result['loan_sum']) / np.log(5)
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
    date_n_user = result.groupby('active_date', as_index=False)['uid'].agg({'date_n_user': 'count'})
    result = result.merge(date_n_user,on='active_date',how='left')
    result['active_date'] = result['active_date'].apply(lambda x:diff_of_days('2016-11-01',x))
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
    result_path = cache_path + 'loan_new_{0}_{1}.hdf'.format(start_date, n_month)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
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
            user_owe_permonth = loan_temp.groupby('uid')['owe_permonth'].sum().to_dict()
            user_new['user_owe_permonth'] = user_new['uid'].map(user_owe_permonth).fillna(0)
            user_new['user_owe'] = user_new['user_owe'] + user_new['user_owe_permonth']
            user_new.drop(['user_return_permonth','user_owe_permonth'],axis=1,inplace=True)
        user_new.rename(columns={'user_owe':'user_owe{}'.format(n_month),
                                 'user_return':'user_return{}'.format(n_month)},inplace=True)
        user_new['user_return_rate{}'.format(n_month)] = user_new['user_return{}'.format(n_month)] / user_new['user_owe{}'.format(n_month)]
        result = user_new.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 每隔n天统计一次3个月
def get_loan_window(start_date, n_day):
    result_path = cache_path + 'loan_window_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        loan_window_sum = user[['uid']].copy()
        n_window = 90//n_day
        for i in range(n_window):
            data_start_date = date_add_days(start_date, (0 - i - 1) * n_day)
            data_end_date = date_add_days(start_date, (0 - i) * n_day)
            loan_temp = loan[(loan['loan_time'] >= data_start_date) & (loan['loan_time'] < data_end_date)]
            loan_window_sum_sub = loan_temp.groupby('uid',as_index=False)['loan_amount'].agg({'loan_window_sum{0}_{1}'.format(n_day,i):'sum'})
            loan_window_sum = loan_window_sum.merge(loan_window_sum_sub,on='uid',how='left')
        result = loan_window_sum.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 一个月借款多少天
def get_loan_nday_count(start_date, n_day, period):
    result_path = cache_path + 'loan_{1}day_count{2}_{0}.hdf'.format(start_date, n_day, period)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - period))
        data_end_date = start_date
        loan_temp = loan[(loan['loan_time'] >= data_start_date) & (loan['loan_time'] < data_end_date)].copy()
        loan_temp['loan_time'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date,x[:10])//n_day)
        loan_day_count = loan_temp.groupby('uid',as_index=False)['loan_time'].agg({'loan_{0}day_count{1}'.format(n_day, period):'nunique'})
        result  = user[['uid']].merge(loan_day_count,on='uid',how='left')
        result = result.drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户最近一次借款
def get_loan_last(start_date):
    data_end_date = start_date
    loan_temp = loan[(loan['loan_time'] < data_end_date)].copy().sort_values('loan_time')
    loan_temp['loan_date'] = loan_temp['loan_time'].apply(lambda x: diff_of_days(start_date, x[:10]))
    loan_temp['loan_hour'] = loan_temp['loan_time'].str[11:13].astype(int)
    loan_last = loan_temp.drop_duplicates('uid',keep='last').drop('loan_time',axis=1)
    loan_last.rename(columns={'loan_date':'loan_last_date','loan_hour':'loan_last_hour',
                              'loan_amount':'last_loan_amount','plannum':'last_plannum'},inplace=True)
    result = user[['uid']].merge(loan_last, on='uid', how='left')
    result = result.drop('uid', axis=1)
    return result

# 最近一次交易
def get_order_last(start_date):
    data_end_date = start_date
    order_temp = order[(order['buy_time'] < data_end_date)].copy().sort_values('buy_time')
    order_temp['order_date'] = order_temp['buy_time'].apply(lambda x: diff_of_days(start_date, x))
    loan_last = order_temp.drop_duplicates('uid', keep='last').drop('buy_time', axis=1)
    loan_last.rename(columns={'order_date': 'order_last_date'}, inplace=True)
    result = user[['uid']].merge(loan_last, on='uid', how='left')
    result = result.drop('uid', axis=1)[['order_last_date']]
    return result

# 最近一次交易
def get_click_last(start_date):
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

# 50天前是否有浏览记录
def get_click_history(start_date, n_day):
    result_path = cache_path + 'click_history_{0}_{1}.hdf'.format(start_date, n_day)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_start_date = date_add_days(start_date, (0 - n_day))
        click_temp = click[(click['click_time'] >= data_start_date) & (click['click_time'] < start_date)]
        result = click_temp.groupby('uid', as_index=False)['pid'].agg({'count_click_{}'.format(n_day): 'count'})
        result = user[['uid']].merge(result, on='uid', how='left').drop('uid', axis=1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# binary_pred
def get_binary_pred(start_date):
    data_binary_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\data\data_binary_pred.csv')
    data_binary_pred = data_binary_pred[data_binary_pred['date']==start_date]
    result = user[['uid']].merge(data_binary_pred,on='uid',how='left')
    result = result[['binary_pred']]
    return result

# 二次处理特征
def second_feat(result):
    return result

# 制作训练集
def make_feats(start_date):
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
        loan_history_60 = get_loan_history(start_date, 60)          # 60天前是否有借款
        loan_history_30 = get_loan_history(start_date, 30)          # 30天前是否有借款
        loan_history_7 = get_loan_history(start_date, 7)            # 7天前是否有借款
        loan_history_1 = get_loan_history(start_date, 1)            # 1天前是否有借款
        loan_now_3 = get_loan_now(start_date, 3)                    # 本月需还款 以及 剩余还款
        loan_window30 = get_loan_window(start_date, 30)             # 每隔n天统计一次3个月
        loan_1day_count30 = get_loan_nday_count(start_date, 1, 30)  # 一个月借款多少天
        loan_1day_count60 = get_loan_nday_count(start_date, 1, 60)  # 一个月借款多少天
        loan_1day_count90 = get_loan_nday_count(start_date, 1, 90)  # 一个月借款多少天
        loan_7day_count28 = get_loan_nday_count(start_date, 7, 28)  # 一个月借款多少天
        loan_7day_count56 = get_loan_nday_count(start_date, 7, 56)  # 一个月借款多少天
        loan_7day_count84 = get_loan_nday_count(start_date, 7, 84)  # 一个月借款多少天
        loan_last = get_loan_last(start_date)                       # 最近一次借款，最近一次借款金额，借款类型
        order_last = get_order_last(start_date)                     # 最近一次交易
        click_last = get_click_last(start_date)                     # 最近一次点击
        order_history_90 = get_order_history(start_date, 90)        # 50天前是否有交易记录
        order_history_60 = get_order_history(start_date, 60)        # 50天前是否有交易记录
        order_history_30 = get_order_history(start_date, 30)        # 50天前是否有交易记录
        order_history_7 = get_order_history(start_date, 7)          # 50天前是否有交易记录
        click_history_90 = get_click_history(start_date, 90)        # 30天前是否有浏览记录
        click_history_60 = get_click_history(start_date, 60)        # 30天前是否有浏览记录
        click_history_30 = get_click_history(start_date, 30)        # 30天前是否有浏览记录


        print('开始合并特征...')
        result = concat([user_feat,loan_history_120,loan_history_60,loan_history_30,loan_history_7,
                         loan_history_1,loan_now_3,loan_window30,loan_last,order_history_90,click_history_90,
                         order_history_60,click_history_60,order_history_30,click_history_30,
                         loan_1day_count30,loan_1day_count60,loan_1day_count90,order_history_7,
                         loan_7day_count28,loan_7day_count56,loan_7day_count84,order_last,click_last])
        result = second_feat(result)

        print('添加label')
        result = get_label(result,start_date)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result







import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2016-11-01'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7))).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30)).fillna(-1)

predictors = train_feat.columns.drop(['uid','loan_sum'])

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
'sub_feature': 0.5,
'num_leaves': 30,
'min_data':500,
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
    gbm = lgb.train(params, lgb_train, 1100)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds += test_preds_sub

test_preds = test_preds/5
print('CV训练用时{}秒'.format(time.time() - t0))
pred_mean = np.mean(test_preds)
print(pred_mean)
submission = pd.DataFrame({'uid':test_feat.uid.values,'pred':test_preds/pred_mean*1.2575})[['uid','pred']]
submission['pred'] = submission['pred'].apply(lambda x: x if x>0.2 else 0.2)
submission.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, header=None, float_format='%.4f')













