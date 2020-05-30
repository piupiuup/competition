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
    data = power_of_date.drop('power_of_date')
    for i in range(4):
        data_sub = power_of_date.copy()
        if (n_days-i) != 0:
            data_sub['record_date'] = data_sub['record_date'].apply(lambda x: date_add_days(x,i*7+366))
            data = pd.merge(power_of_date,data_sub,on='record_date',how='left')
        data['power_last_year_of_month'] = data[[name for name in data.columns if name!='recored_date']].median(axis=1)
    result = data[(data['record_date']>=start_date) & (data['record_date']<end_date)]
    result = result[['record_date','power_last_year_of_month']]
    return result

# 获取节假日信息
def get_holidays(start_date, end_date):
    holidays = pd.read_csv(r'C:\Users\csw\Desktop\python\Tianchi_power\Tianchi_power.csv')
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
        power_of_date['record_date'] = power_of_date['record_date'].astype('str')

        # 获取特征
        start_date = date_add_days(end_date, -n_days)
        ID = get_id(power_of_date, start_date, end_date)                            # 获取ID 和 label
        week = get_week(power_of_date, start_date)                                  # 获取日期对应的周几以及平均
        power_last_year = get_power_last_year(power_of_date, start_date, end_date)  # 获取去年同期的用电量
        power_last_year_of_month = get_power_last_year_of_month(power_of_date, start_date, end_date, 7)# 获取去年同期一个月的均值

        # merge
        train_set = ID
        train_set = pd.merge(train_set, week,                       how='left', on='record_date')
        train_set = pd.merge(train_set, power_last_year,            how='left', on='record_date')
        train_set = pd.merge(train_set, power_last_year_of_month,   how='left', on='record_date')
        train_set.to_hdf(set_path, 'w', complib='blosc', complevel=5)
    return train_set



if __name__ == "__main__":
    date_label = time.strftime('%Y%m%d', time.localtime(time.time()))
    for i in range(1,21):
        submission_path = r'C:\Users\csw\Desktop\python\Tianchi_power\submission\%s(%d).csv' % (date_label,i)
        if not os.path.exists(submission_path):
            break
    submission = make_submission()
    submission.to_csv(submission_path,index=False)
