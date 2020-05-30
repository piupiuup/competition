import numpy as np
import pandas as pd


data_path = 'C:/Users/csw/Desktop/python/Corporacion/data/'

holiday = pd.read_csv(data_path + 'holidays_events.csv')
item = pd.read_csv(data_path + 'items.csv')
oil = pd.read_csv(data_path + 'oil.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
store = pd.read_csv(data_path + 'stores.csv')
test = pd.read_csv(data_path + 'test.csv')
train = pd.read_csv(data_path + 'train.csv')
transaction = pd.read_csv(data_path + 'transactions.csv')


print('训练集大小：{}'.format(train.shape))
print('测试集大小：{}'.format(test.shape))

print('训练集中商店个数：{}'.format(train.store_nbr))
print('测试集中商店个数：{}'.format(test.store_nbr))

print('训练集中商品个数：{}'.format(train.item_nbr))
print('测试集中商品个数：{}'.format(test.item_nbr))

print('训练集中商店-商品个数：{}'.format(train[['store_nbr','item_nbr']].drop_duplicates()[0]))
print('测试集中商店-商品个数：{}'.format(test[['store_nbr','item_nbr']].drop_duplicates()[0]))


print('去年同期8月份的销量')
train_2016_08 = train[(train.date<='2016-08-31') & (train.date>='2016-08-01')]
a = train_2016_08.groupby('date')['unit_sales'].sum()
a.plot()
train_2016_08 = train[(train.date<='2017-08-31') & (train.date>='2017-08-01')]
a = train_2016_08.groupby('date')['unit_sales'].sum()
a.plot()



















