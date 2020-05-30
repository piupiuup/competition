import numpy as np
import pandas as pd
from collections import Counter

data_path = 'C:/Users/csw/Desktop/python/JD/moshi/data/'
train_path = data_path + 't_login.csv'
test_path = data_path + 't_login_test.csv'
train_label_path = data_path + 't_trade.csv'
test_label_path = data_path + 't_trade_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_label = pd.read_csv(train_label_path)
test_label = pd.read_csv(test_label_path)

train.drop(['timestamp','is_scan','is_sec'],axis=1,inplace=True)
test.drop(['timestamp','is_scan','is_sec'],axis=1,inplace=True)
train.rename(columns={'time':'log_time'},inplace=True)
test.rename(columns={'time':'log_time'},inplace=True)
train_label.rename(columns={'is_risk':'label'},inplace=True)
test_label.rename(columns={'is_risk':'label'},inplace=True)

# 让最后一次登录与下一次交易记录对应起来
train_label = train_label.merge(train,on='id',how='left')
train_label = train_label[train_label['time']>=train_label['log_time']]
train_label = train_label[train_label['result']!=31]
train_label.sort_values('log_time',inplace=True)
train_label.drop_duplicates('rowkey',keep='last',inplace=True)















