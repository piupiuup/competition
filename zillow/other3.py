#encoding=utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc


data_path = 'C:/Users/csw/Desktop/python/zillow/data/'
prop_path = data_path + 'properties_2016.csv'
sample_path = data_path + 'sample_submission.csv'
train_path = data_path + 'train_2016_v2.csv'

train = pd.read_csv(train_path)
prop = pd.read_csv(prop_path)

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')



x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate','propertycountylandusecode','propertyzoningdesc'], axis=1)
y_train = df_train['logerror'].values

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 60000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {
'learning_rate': 0.002,
'boosting_type': 'gbdt',
'objective': 'regression',
'metric': 'mae',
'sub_feature': 0.5,
'num_leaves': 60,
'min_data':500,
'min_hessian': 1,
}

watchlist = [d_valid]
clf = lgb.train(params, d_train, 1500, watchlist,early_stopping_rounds=100)
