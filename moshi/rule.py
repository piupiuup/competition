from moshi.feat1 import *
import lightgbm as lgb

data_path = 'C:/Users/csw/Desktop/python/JD/moshi/data/'
train_path = data_path + 't_login.csv'
test_path = data_path + 't_login_test.csv'
train_label_path = data_path + 't_trade.csv'
test_label_path = data_path + 't_trade_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_label = pd.read_csv(train_label_path)
test_label = pd.read_csv(test_label_path)

train.drop(['timestamp', 'is_scan', 'is_sec'], axis=1, inplace=True)
test.drop(['timestamp', 'is_scan', 'is_sec'], axis=1, inplace=True)
train.rename(columns={'time': 'log_time'}, inplace=True)
test.rename(columns={'time': 'log_time'}, inplace=True)
train_label.rename(columns={'is_risk': 'label'}, inplace=True)
test_label.rename(columns={'is_risk': 'label'}, inplace=True)
data = pd.concat([train, test])

data_label = label.merge(data, on='id', how='left')
data_label = data_label[data_label['time'] >= data_label['log_time']]
data_label = data_label[data_label['result'] != 31]
data_label.sort_values('log_time', inplace=True)
data_label.drop_duplicates('rowkey', keep='last', inplace=True)
data_label = label[['rowkey', 'time']].merge(data_label, on=['rowkey', 'time'], how='left')










