import pandas as pd
from datetime import timedelta

data_path = 'C:/Users/csw/Desktop/python/Corporacion/data/'
train = pd.read_csv(data_path + 'train.csv', usecols=[1,2,3,4],
                    dtype={'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'},
                    parse_dates=['date'],
                    skiprows=range(1, 101688779) #Skip dates before 2017-01-01
                    )

train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek

# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
)

del u_dates, u_stores, u_items

# Fill NaNs
train.loc[:, 'unit_sales'].fillna(0, inplace=True)
train.reset_index(inplace=True) # reset index and restoring unique columns
lastdate = train.date,max()

#Load test
test = pd.read_csv(data_path + 'test.csv',
                   usecols=[0,1,2,3,4],
                   dtype={'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'},
                   parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek


# Days of Week Means
# By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train.groupby(['item_nbr','store_nbr','dow'],as_index=False)['unit_sales'].agg({'madw':'mean'})
ma_wk = ma_dw.groupby(['store_nbr', 'item_nbr'],as_index=False)['madw'].agg({'mawk':'mean'})
ma_wk.reset_index(inplace=True)

# Moving Averages
ma_is = train.groupby(['item_nbr','store_nbr'])['unit_sales'].agg({'mais226':'mean'})
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')

del tmp,tmpg

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)

test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])

# Forecasting Test
test['unit_sales'] = test.mais
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']
test.loc[:, "unit_sales"].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values

# 15% more for promotion items
test.loc[test['onpromotion'] == True, 'unit_sales'] = test.loc[test['onpromotion'] == True, 'unit_sales'] * 1.15

# Verify the LB split: Private set starts from 2017-08-21
test.loc[test['id'] >= 126550310, 'unit_sales'] = 0

test[['id','unit_sales']].to_csv('split_verification.csv', index=False, float_format='%.3f')













