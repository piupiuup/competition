import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv(r'C:\Users\csw\Desktop\python\Tianchi_power\Tianchi_power.csv')
data['record_date'] = pd.to_datetime(data['record_date'])
data['week'] = data['record_date'].dt.dayofweek+1
data['record_date'] = data['record_date'].astype('str')

# 统计每月的变化趋势
data['month'] = data['record_date'].str[:7]
power_of_month = data.groupby('month',as_index=False)['power_consumption'].sum()
power_of_month.set_index('month',inplace=True)
a = power_of_month[:12]
b = power_of_month[12:]
a.plot()
b.plot()

# 统计每日的变化趋势
power_of_date = data.groupby('record_date')['power_consumption'].sum()
power_of_date.plot()

# 统计每个企业的总用电量
power_of_user = data.groupby('user_id')['power_consumption'].sum()
