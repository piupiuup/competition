import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime,timedelta
from sklearn.metrics import roc_auc_score

data_path = 'C:/Users/csw/Desktop/python/JD/xindai/data/'
user_path = data_path + 't_user.csv'
order_path = data_path + 't_order.csv'
click_path = data_path + 't_click.csv'
loan_path = data_path + 't_loan.csv'
loan_sum_path = data_path + 't_loan_sum.csv'

user = pd.read_csv(user_path)
order = pd.read_csv(order_path)
click = pd.read_csv(click_path)
loan = pd.read_csv(loan_path)
loan_sum = pd.read_csv(loan_sum_path)

user['limit'] = np.round(5**(user['limit'])-1,2)
order['price'] = np.round(5**(order['price'])-1,2)
order['discount'] = np.round(5**(order['discount'])-1,2)
loan['loan_amount'] = np.round(5**(loan['loan_amount'])-1,2)
loan_sum['loan_sum'] = np.round(5**(loan_sum['loan_sum'])-1,2)

# 日期的加减
def date_add_days(start_date, days):
    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

# 相差的日期数
def diff_of_days(day1, day2):
    days = (datetime.strptime(day1[:10], '%Y-%m-%d') - datetime.strptime(day2[:10], '%Y-%m-%d')).days
    return abs(days)

# 获取标签
def get_label(result, start_date,n_days=30):
    label_start_date = date_add_days(start_date, n_days)
    label = loan[(loan['loan_time']<label_start_date) & (loan['loan_time']>=start_date)]
    label = label.groupby('uid',as_index=False)['loan_amount'].agg({'loan_sum':'sum'})
    label['loan_sum'] = np.log1p(label['loan_sum']) / np.log(5)
    result = result.merge(label[['uid', 'loan_sum']], how='left')
    result['loan_sum'].fillna(0,inplace=True)
    return result

# 分组排序
def rank(data, feat1, feat2, ascending=True):
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 获取转化率
def get_rate(data,name,label='sum_loan'):
    rate = data.groupby(name)[label].agg( {'count': 'count', 'sum': 'sum'})
    rate['rate'] = rate['sum'] / rate['count']
    return rate['rate']

def make_set(start_date):
    data = user.copy()
    label_start_date = date_add_days(start_date, 30)
    label = loan[(loan['loan_time']<label_start_date) & (loan['loan_time']>=start_date)]
    label = label.groupby('uid',as_index=False)['loan_amount'].agg({'loan_sum':'sum'})
    label['loan_sum'] = np.log1p(label['loan_sum']) / np.log(5)
    data = data.merge(label[['uid', 'loan_sum']], how='left').fillna(0)
    data['start_date'] = start_date
    data['label'] = data['loan_sum'].apply(lambda x: 1 if x > 0 else 0)
    return data

train_feat = pd.DataFrame()
start_date = '2016-10-02'
for i in range(3):
    train_feat_sub = make_set(date_add_days(start_date, i*(-30))).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
train_feat['label'] = train_feat['loan_sum'].apply(lambda x:1 if x>0 else 0)
test_start_date = date_add_days(start_date, 30)
test_feat = make_set(test_start_date).fillna(-1)

# 年龄段转化率
train_feat['age_rate'] = train_feat['age'].map(get_rate(train_feat,'age',label='loan_sum'))
# 日期转化率
train_feat['week'] =  pd.to_datetime(train_feat['active_date']).dt.weekday
train_feat['week_rate'] = train_feat['week'].map(get_rate(train_feat,'week',label='loan_sum'))
train_feat['month'] = pd.to_datetime(train_feat['active_date']).dt.month
train_feat['month_rate'] = train_feat['month'].map(get_rate(train_feat, 'month', label='loan_sum'))
train_feat['date_rate'] = train_feat['active_date'].map(get_rate(train_feat, 'active_date', label='loan_sum'))
train_feat['date_n_people'] = train_feat['active_date'].map(train_feat['active_date'].value_counts())
train_feat['limit_rate'] = train_feat['limit'].map(get_rate(train_feat,'limit',label='loan_sum'))

train_feat['user_pred'] = train_feat['age_rate'] * (train_feat['week_rate'] * train_feat['month_rate'] +
                     train_feat['date_rate']*32) * train_feat['limit_rate']**0.4

test_feat = test_feat.merge(train_feat[:90993][['uid','user_pred']])

loan_temp = loan[loan['loan_time']<test_start_date]
loan_temp = rank(loan_temp,'uid','loan_time',ascending=False)
def last_time_rate(x):
    return 0.55/np.exp(x*0.03)+0.2
def not_loan_rate(x):
    return 0.156/np.exp(90*0.036)+0.054
last_plannum_rate = {1:0.707,3:0.5666,6:0.5492,12:0.3862}
def last_amount_rate(x):
    return -0.1636 * (4.36 - np.log(x+1)/np.log(5)) ** 2 + 0.689
def mul_pred(l):
    mul1 = 1; mul2 = 1
    for i in l:
        mul1 = mul1 * i
        mul2 = mul2 * (1-i)
    return mul1/(mul1+mul2)

loan_temp['last_time_rate'] = loan_temp['loan_time'].apply(lambda x:last_time_rate(diff_of_days(test_start_date,x)))
loan_temp['last_plannum_rate'] = loan_temp['plannum'].apply(lambda x:last_plannum_rate[x])
loan_temp['last_amount_rate'] = loan_temp['loan_amount'].apply(lambda x:last_amount_rate(x))
loan_temp['loan_pred'] = loan_temp.apply(lambda x: x.last_time_rate * (x.last_plannum_rate*0.85 + x.last_amount_rate*0.15)**0.9,axis=1)
loan_temp['loan_pred'] = loan_temp['loan_pred']/np.exp(loan_temp['rank']*0.3)
loan_temp['loan_pred'] = loan_temp['uid'].map(loan_temp.groupby('uid')['loan_pred'].sum())

max_diff_days = diff_of_days(test_start_date, loan_temp['loan_time'].min())
nan_values = not_loan_rate(max_diff_days)
loan_pred = loan_temp[loan_temp['rank']==0].copy().set_index('uid',drop=False)

a = test_feat.merge(loan_pred[['uid','last_time_rate','last_plannum_rate','last_amount_rate','loan_pred']],on='uid',how='left')
a['loan_pred'].fillna(nan_values ** 1.9,inplace=True)
print('AUC得分为：{}'.format(roc_auc_score(a['label'].values,a['loan_pred'].values)))












