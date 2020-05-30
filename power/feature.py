import pandas as pd
import numpy as np
import time
import os
import xgboost as xgb
from datetime import datetime
from datetime import timedelta

data_path = r'C:\Users\csw\Desktop\python\Tianchi_power\Tianchi_power.csv'

def predict(train,test):
    result = None
    return result

#日期的加减
def date_add_days(start_date, days):
    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

# 获取ID 和 label
def get_id(power_of_date, start_date, end_date):
    data = power_of_date.copy()
    result = data[(data['record_date']>=start_date) & (data['record_date']<end_date)]
    return result

# 获取前一周的均值
def get_avg_first_week(power_of_date, end_date, n_days):
    data = power_of_date.copy()
    start_date = date_add_days(end_date, -n_days)
    result = data[(data['record_date']>=start_date) & (data['record_date']<end_date)].mean().values[0]
    return result

# 获取日期对应的周几以及平均
def get_week(power_of_date, end_date):
    # 将日期转换为周几
    data = power_of_date.copy()
    data['record_date'] = pd.to_datetime(data['record_date'])
    data['week'] = data['record_date'].dt.dayofweek + 1
    # 统计平均每周几的平均用电量
    result = data[data['record_date']<end_date].groupby('week', as_index=False)['power_of_date'].agg({'power_of_week': 'sum'})
    result = pd.merge(data,result,how='left',on='week')
    result = result[['record_date', 'week', 'power_of_week']]
    result['record_date'] = result['record_date'].astype('str')
    return result

# 获取去年阳历同期的数据
def get_power_last_year(power_of_date, start_date, end_date):
    data = power_of_date.copy()
    data['record_date'] = data['record_date'].apply(lambda x: '%d%s' % (int(x[:4])+1,x[4:]))
    data.rename(columns={'power_of_date':'power_last_year'},inplace=True)
    data = pd.merge(power_of_date,data,on='record_date',how='left')
    result = data[(data['record_date']>=start_date) & (data['record_date']<end_date)]
    result = result[['record_date','power_last_year']]
    return result

# 获取去年阳历同期一个月的均值数据
def get_power_last_year_of_month(power_of_date, start_date, end_date, n_days):
    for i in range(n_days):
        data_sub = power_of_date.copy()
        if (n_days//2-i) != 0:
            data_sub['record_date'] = data_sub['record_date'].apply(lambda x: date_add_days(x, n_days//2-i+366))
            data = pd.merge(power_of_date,data_sub,on='record_date',how='left')
        data['power_last_year_of_month'] = data[[name for name in data.columns if name!='recored_date']].mean(axis=1)
    result = data[(data['record_date']>=start_date) & (data['record_date']<end_date)]
    result = result[['record_date','power_last_year_of_month']]
    return result

# 获取节假日信息
def get_holidays(start_date, end_date):
    holidays = pd.read_csv(r'C:\Users\csw\Desktop\python\Tianchi_power\holiday.csv')
    result = holidays[(holidays['record_date'] >= start_date) & (holidays['record_date'] < end_date)]

    return result

# 构造训练集和测试集
def make_train_set(end_date, n_days):
    set_path = r'C:\Users\csw\Desktop\python\Tianchi_power\cache\train_set_%s_%d_days.hdf' % (end_date, n_days)
    if os.path.exists(set_path) & 0:
        train_set = pd.read_hdf(set_path, 'w')
    else:
        # 统计每日的变化趋势
        data = pd.read_csv(data_path)
        data['record_date'] = pd.to_datetime(data['record_date'])
        power_of_date = data.groupby('record_date', as_index=False)['power_consumption'].agg({'power_of_date': 'sum'})
        power_of_date_temp = pd.DataFrame({'record_date': pd.date_range('20161001', '20161031'), 'power_of_date': np.nan})
        power_of_date = pd.concat([power_of_date, power_of_date_temp])
        power_of_date['day_of_month'] = power_of_date.record_date.dt.day
        power_of_date['time_dayofyear'] = power_of_date.record_date.dt.dayofyear
        power_of_date['time_is_month_end'] = power_of_date.record_date.dt.is_month_end.astype(int)
        power_of_date['record_date'] = power_of_date['record_date'].astype('str')

        # 获取特征
        start_date = date_add_days(end_date, -n_days)
        ID = get_id(power_of_date, start_date, end_date)                            # 获取ID 和 label
        #avg_first_week = get_avg_first_week(power_of_date, start_date, 7)           # 获取前一周的均值
        week = get_week(power_of_date, start_date)                                  # 获取日期对应的周几以及平均
        power_last_year = get_power_last_year(power_of_date, start_date, end_date)  # 获取去年同期的用电量
        power_last_year_of_month = get_power_last_year_of_month(power_of_date, start_date, end_date, 28)# 获取去年同期一个月的均值
        holidays = get_holidays(start_date, end_date)                               # 获取节假日表

        # merge
        train_set = ID
        #train_set['avg_first_week'] = avg_first_week
        train_set = pd.merge(train_set, week,                       how='left', on='record_date')
        train_set = pd.merge(train_set, power_last_year,            how='left', on='record_date')
        train_set = pd.merge(train_set, power_last_year_of_month,   how='left', on='record_date')
        train_set = pd.merge(train_set, holidays,                   how='left', on='record_date')
        train_set['weight'] =  train_set['power_last_year']/train_set['power_last_year_of_month']

        train_set.to_hdf(set_path, 'w', complib='blosc', complevel=5)
    return train_set

'''
dates = pd.date_range('2016-03-01','2016-09-03')
train = None
for i in range(1,180):
    if train is None:
        train = make_train_set(str(dates[-i])[:10], 28)
    else:
        train = pd.concat([train,make_train_set(str(dates[-i])[:10], 28)])

feature_label = ['avg_first_week', 'week', 'power_of_week','power_last_year','power_last_year_of_month']
train1 = train
train2 = make_train_set('2016-10-01', 28)
xgtrain_x = xgb.DMatrix(train2[feature_label], train2['power_of_date'])
xgtrain_y = xgb.DMatrix(train1[feature_label], train1['power_of_date'])

# xgtest = xgb.DMatrix(test[feature_label])

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 4,
          # 'lambda':100,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'min_child_weight': 3,
          'eta': 0.01,
          'seed': 66,
          }
params['silent'] = 1
watchlist = [(xgtrain_x, 'train'), (xgtrain_y, 'eval')]
model = xgb.train(params, xgtrain_x, 5000, watchlist,verbose_eval = 10, early_stopping_rounds=50)
'''

def make_submission():
    feature_label = feature_label = ['week', 'holiday', 'day_of_month', 'time_dayofyear']
    train1 = make_train_set('2016-10-01', 30)
    train2 = make_train_set('2016-09-01', 1000)
    test = make_train_set('2016-11-01', 31)
    train = make_train_set('2016-09-01', 100)
    xgtrain_x = xgb.DMatrix(train2[feature_label], train2['power_of_date'])
    xgtrain_y = xgb.DMatrix(train1[feature_label], train1['power_of_date'])

    # xgtest = xgb.DMatrix(test[feature_label])

    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 7,
              # 'lambda':100,
              'subsample': 0.9,
              'colsample_bytree': 0.8,
              'min_child_weight': 3,
              'eta': 0.005,
              'seed': 66,
              }
    params['silent'] = 1
    watchlist = [(xgtrain_x, 'train'), (xgtrain_y, 'eval')]
    model = xgb.train(params, xgtrain_x, 5000, watchlist, verbose_eval=10, early_stopping_rounds=50)
    best_params = model.best_ntree_limit
    test_xgb = xgb.DMatrix(test[feature_label])
    train_xgb = xgb.DMatrix(train[feature_label], label=train['power_of_date'])
    watchlist = [(train_xgb, 'train')]
    model = xgb.train(params, train_xgb, best_params, watchlist,verbose_eval = 10, early_stopping_rounds=50)
    test_y = model.predict(test_xgb)
    result = pd.DataFrame({'predict_date':test['record_date'].values, 'predict_power_consumption':test_y})
    result['predict_power_consumption'] = result['predict_power_consumption'].astype(int)
    result['predict_date'] = result['predict_date'].apply(lambda x: '%s%s%s' % (x[:4],x[5:7],x[8:10]))
    return result

if __name__ == "__main__":
    date_label = time.strftime('%Y%m%d', time.localtime(time.time()))
    for i in range(1,21):
        submission_path = r'C:\Users\csw\Desktop\python\Tianchi_power\submission\%s(%d).csv' % (date_label,i)
        if not os.path.exists(submission_path):
            break
    submission = make_submission()
    submission.to_csv(submission_path,index=False)



# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:55:48 2017

@author: Administrator
"""

import pandas as pd
import xgboost as xgb
import itertools
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 第一步读取数据，并用笛卡尔积生成预测集
# ==============================================================================
seed = 6666

df = pd.read_csv(data_path)
df.record_date = pd.to_datetime(df.record_date)
dk = itertools.product(df.user_id.unique().tolist(), pd.date_range('20160901', '20160930').tolist())
yc = pd.DataFrame([i for i in dk], columns=['user_id', 'record_date'])
data_concat = pd.concat([df, yc])
sydf = data_concat.groupby(['user_id', 'record_date']).sum().unstack()
yc = sydf.sum().reset_index()
yc = yc[['record_date', 0]]
yc.columns = ['record_date', 'power_consumption']

yc['time_year'] = yc.record_date.dt.year
yc['time_day'] = yc.record_date.dt.day
yc['time_dayofweek'] = yc.record_date.dt.dayofweek
yc['time_dayofyear'] = yc.record_date.dt.dayofyear
yc['time_is_month_end'] = yc.record_date.dt.is_month_end.astype(int)
yc['time_is_month_start'] = yc.record_date.dt.is_month_start.astype(int)

train = yc[(yc.record_date >= '20150401') & (yc.record_date < '20151001')]
test = yc[(yc.record_date >= '20160401') & (yc.record_date < '20160701')]
pred = yc[(yc.record_date >= '20160901') & (yc.record_date < '20161001')]

train_tz = yc[(yc.record_date >= '20150101') & (yc.record_date < '20150401')]
test_tz = yc[(yc.record_date >= '20160101') & (yc.record_date < '20160401')]

train['syzh'] = train_tz['power_consumption'].sum()
test['syzh'] = test_tz['power_consumption'].sum()
pred['syzh'] = test_tz['power_consumption'].sum()

train['syzh'] = train_tz['power_consumption'].mean()
test['syzh'] = test_tz['power_consumption'].mean()
pred['syzh'] = test_tz['power_consumption'].mean()

# ==============================================================================
# 4 5 6 7 8 9 10 11
#
# 4
# ==============================================================================




train.index = range(train.shape[0])
test.index = range(test.shape[0])
pred.index = range(pred.shape[0])

train1 = test[(test.record_date >= '20160401') & (test.record_date < '20160501')]
train = pd.concat([train, train1])
test = test[(test.record_date >= '20160501') & (test.record_date < '20160701')]

featurelist = [i for i in train.columns if i not in df.columns]
gbdt = GradientBoostingRegressor(random_state=seed)
# gbdt = RandomForestRegressor(random_state = seed)
# gbdt = ExtraTreesRegressor(n_estimators=10,random_state = seed)
# gbdt = lgb.LGBMRegressor(max_depth = 2,learning_rate=0.05,n_estimators=3000,reg_alpha=10)
# gbdt = xgb.XGBRegressor(max_depth = 2,learning_rate=0.1,n_estimators=2000,reg_alpha=5,gamma = 10)
# gbdt = xgb.XGBRegressor(max_depth = 7,learning_rate=0.1,n_estimators=200,reg_alpha=5,gamma = 10)
gbdt = gbdt.fit(train[featurelist], train.power_consumption)
test['power_consumptionbk'] = test['power_consumption']
test.power_consumption = gbdt.predict(test[featurelist])
test.power_consumption = test.power_consumption.astype(int)

pred.power_consumption = gbdt.predict(pred[featurelist])

from matplotlib import pyplot as plt

plt.figure()
pred.power_consumption.plot()

pred.power_consumption = pred.power_consumption.astype(int)

jg = pred[['record_date', 'power_consumption']]
jg.to_csv('XGBcs.csv', index=False)

print
np.sqrt(mean_squared_error(test.power_consumptionbk, test.power_consumption))