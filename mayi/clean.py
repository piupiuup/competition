import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict

data_path = 'C:/Users/csw/Desktop/python/mayi/data/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train['row_id'] = list(np.arange(len(train)) + 1000000)
train = train[['row_id','user_id', 'shop_id', 'time_stamp', 'longitude', 'latitude','wifi_infos']]

train_wifi_infos = []
for wifi_info in train['wifi_infos'].values:
    train_wifi_infos.append(';'.join(sorted([wifi for wifi in wifi_info.split(';')],key=lambda x:int(x.split('|')[1]),reverse=True)))
train['wifi_infos'] = train_wifi_infos

test_wifi_infos = []
for wifi_info in test['wifi_infos'].values:
    test_wifi_infos.append(';'.join(sorted([wifi for wifi in wifi_info.split(';')],key=lambda x:int(x.split('|')[1]),reverse=True)))
test['wifi_infos'] = test_wifi_infos

test.to_csv(test_path,index=False)
train.to_csv(train_path,index=False)
