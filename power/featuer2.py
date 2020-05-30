import pandas as pd
import numpy as np
import time
from datetime import datetime
from datetime import timedelta
import xgboost as xgb

# 获取训练集
def get_train(data, end_date, n_day):
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(n_day)
    start_date = start_date.strftime('%Y-%m-%d')
    result = data[(data['record_date']<end_date) & (data['record_date']>=start_date)]
    return result

# 读取数据
data_path = r'C:\Users\csw\Desktop\python\Tianchi_power\Tianchi_power.csv'
data = pd.read_csv(data_path)
holiday = pd.read_csv(r'C:\Users\csw\Desktop\python\Tianchi_power\holiday.csv')


# 统计每天的用电量
power_of_perday = data.groupby('record_date',as_index=False)['power_consumption'].agg({'power_of_perday':'sum'})
power_of_perday['record_date'] = power_of_perday['record_date'].apply(lambda x: time.strftime('%Y-%m-%d', time.strptime(x,'%Y/%m/%d')))
power_of_perday = pd.merge(holiday,power_of_perday,on='record_date',how='left')
power_of_perday['record_date'] = pd.to_datetime(power_of_perday['record_date'])
power_of_perday['year'] = power_of_perday.record_date.dt.year
power_of_perday['day_of_month'] = power_of_perday.record_date.dt.day
power_of_perday['day_of_week'] = power_of_perday.record_date.dt.dayofweek
power_of_perday['day_of_year'] = power_of_perday.record_date.dt.dayofyear
power_of_perday['record_date'] = power_of_perday['record_date'].astype('str')

# 线下测试机 [2512]  train-rmse:134404       eval-rmse:173698
'''
feature_label = ['holiday', 'year', 'day_of_month', 'day_of_week', 'day_of_year']
train1 = get_train(power_of_perday, '2016-10-01', 30)
train2 = get_train(power_of_perday, '2016-09-01', 1000)
xgtrain_x = xgb.DMatrix(train2[feature_label], train2['power_of_perday'])
xgtrain_y = xgb.DMatrix(train1[feature_label], train1['power_of_perday'])

# xgtest = xgb.DMatrix(test[feature_label])

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 4,
          'subsample': 0.93,
          'colsample_bytree': 0.7,
          'min_child_weight': 4,
          'eta': 0.0045,
          'seed': 2017,
          }
params['silent'] = 1
watchlist = [(xgtrain_x, 'train'), (xgtrain_y, 'eval')]
model = xgb.train(params, xgtrain_x, 5000, watchlist,verbose_eval = 10, early_stopping_rounds=50)
'''

# 生成训练集和预测集
feature_label = ['holiday', 'year', 'day_of_month', 'day_of_week', 'day_of_year']
test = get_train(power_of_perday, '2016-11-01', 31)
train = get_train(power_of_perday, '2016-10-01', 1000)
dtest = xgb.DMatrix(test[feature_label])
dtrain = xgb.DMatrix(train[feature_label], train['power_of_perday'])

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 4,
          'subsample': 0.93,
          'colsample_bytree': 0.7,
          'min_child_weight': 4,
          'eta': 0.0045,
          'seed': 2017,
          }
params['silent'] = 1
model = xgb.train(params, dtrain, 2512)
pred = model.predict(dtest)
result = pd.DataFrame({'predict_date':test['record_date'].values, 'predict_power_consumption':pred})
result['predict_power_consumption'] = result['predict_power_consumption'].astype(int)
result['predict_date'] = result['predict_date'].apply(lambda x: '%s%s%s' % (x[:4],x[5:7],x[8:10]))
result.to_csv('result.csv',index=False)