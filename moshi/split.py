import pickle
import numpy as np
import pandas as pd
from collections import Counter

data_path = 'C:/Users/csw/Desktop/python/JD/moshi/data/'
eval_path = 'C:/Users/csw/Desktop/python/JD/moshi/data/eval/'
train_path = 't_login.csv'
test_path =  't_login_test.csv'
train_label_path = 't_trade.csv'
test_label_path = 't_trade_test.csv'

train = pd.read_csv(data_path + train_path)
train_label = pd.read_csv(data_path + train_label_path)
true = dict(zip(train_label['rowkey'].values,train_label['is_risk'].values))
test = train[train['time']>='2015-05-00 00:00:00']
train = train[train['time']<'2015-05-00 00:00:00']
test_label = train_label[train_label['time']>='2015-05-00 00:00:00']
train_label = train_label[train_label['time']<'2015-05-00 00:00:00']
test_label.drop('is_risk',axis=1,inplace=True)

train.to_csv(eval_path + train_path, index=False)
test.to_csv(eval_path + test_path, index=False)
train_label.to_csv(eval_path + train_label_path, index=False)
test_label.to_csv(eval_path + test_label_path, index=False)
pickle.dump(true,open(data_path+'true.pkl','+wb'))











