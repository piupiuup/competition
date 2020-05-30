import pickle
import numpy as np
import pandas as pd
from collections import Counter

data_path = 'C:/Users/csw/Desktop/python/mayi/data/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

new_data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
new_test_path = new_data_path + 'evaluation_public.csv'
new_shop_path = new_data_path + 'ccf_first_round_shop_info.csv'
new_train_path = new_data_path + 'ccf_first_round_user_shop_behavior.csv'

shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)

new_train = train[train['time_stamp']<'2017-08-25'].copy()
new_test = train[train['time_stamp']>='2017-08-25'].copy()
new_test = new_test.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
true = dict(zip(train['row_id'].values,train['shop_id'].values))
new_test = new_test[['row_id', 'user_id', 'mall_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos']]


new_test.to_csv(new_test_path,index=False)
new_train.to_csv(new_train_path,index=False)
shop.to_csv(new_shop_path)
pickle.dump(true,open(new_data_path+'true.pkl','+wb'))

