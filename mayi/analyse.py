import numpy as np
import pandas as pd
from collections import Counter

data_path = 'C:/Users/csw/Desktop/python/mayi/data/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)

print('训练集数据个数：{}'.format(train.shape[0]))
print('测试集数据个数：{}'.format(test.shape[0]))
print('商家个数：{}'.format(shop.shape[0]))

print('商家种类数：{}'.format(shop['category_id'].nunique()))
print('商场个数：{}'.format(shop['mall_id'].nunique()))

# 统计信号的大小的分布
wifi_nums = []
for row in train['wifi_infos'].values:
    wifis = row.split(';')
    for wifi in wifis:
        wifi_nums.append(wifi.split('|')[1])
wifi_num_value_count = pd.Series(Counter(wifi_nums))
wifi_num_value_count.sort_index(inplace=True)
wifi_num_value_count.plot(kind='bar')

# 统计每部手机的wifi个数
train_temp = train.copy()
train_temp['n_wifi'] = train_temp['wifi_infos'].apply(lambda x:x.count(';')+1)
train_temp.groupby('n_wifi').size().plot(kind='bar')























