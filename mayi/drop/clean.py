import pandas as pd
import numpy as np
import ast
import re

history=pd.read_csv('history.csv',names=['row_id','user_id','shop_id','time_stamp','longitude','latitude','wifi_infos','mall_id'])
shop=pd.read_csv('ccf_first_round_shop_info.csv')
test=pd.read_csv('evaluation_public.csv')

#1周信息用格子
week=1
train=history[(history.time_stamp>='2017-08-22 00:00')&(history.time_stamp<'2017-08-29 00:00')].copy()
history=history[history.time_stamp<='2017-08-22 00:00']
train.index=train.row_id.values

#2周信息用格子
week=2
train=history[history.time_stamp>'2017-08-29 00:00'].copy()
history=history[history.time_stamp<='2017-08-22 00:00']
train.index=train.row_id.values

#3周信息用格子
week=3
train=history[history.time_stamp>'2017-08-22 00:00'].copy()
history=history[history.time_stamp<='2017-08-22 00:00']
train.index=train.row_id.values

#4周信息用格子
week=4
train=test
train.index=train.row_id.values

#block-shop对
history['lo1']=(history.longitude//0.00001).astype(int)
history['la1']=(history.latitude//0.00001).astype(int)
history['lo5']=(history.longitude//0.00005).astype(int)
history['la5']=(history.latitude//0.00005).astype(int)
history['lo10']=(history.longitude//0.0001).astype(int)
history['la10']=(history.latitude//0.0001).astype(int)
for i in ['1','5','10']:
    a=history.groupby(['mall_id','lo%s'%i,'la%s'%i],as_index=False)['time_stamp'].count()
    a.columns=['mall_id','lo%s'%i,'la%s'%i,'block_total%s'%i]
    a.to_csv('data_%s/block_total%s.csv'%(week,i))
    a=history.groupby(['mall_id','lo%s'%i,'la%s'%i,'shop_id'],as_index=False)['time_stamp'].count()
    a.columns=['mall_id','lo%s'%i,'la%s'%i,'shop_id','block_shop%s'%i]
    a.to_csv('data_%s/block_shop%s.csv'%(week,i))

#shop位置历史
location=history.groupby('shop_id',as_index=False)['longitude','latitude'].median()

#shop频率
frequency=history.loc[:,['time_stamp','mall_id','shop_id','user_id']].drop_duplicates()
frequency=frequency.groupby(['mall_id','shop_id'],as_index=False)['user_id'].count()
frequency.columns=['mall_id','shop_id','shop_visit']
frequency['mall_sum']=frequency.groupby('mall_id')['shop_visit'].transform('sum').values
frequency['shop_frequency']=frequency.shop_visit/frequency.mall_sum
frequency.drop(['mall_id','mall_sum'],axis=1,inplace=True)

#user-shop对
user_shop=history.groupby(['user_id','shop_id'],as_index=False)['time_stamp'].count()
user_shop.columns=['user_id','shop_id','user_shop_visit']

#user-mall对
user_mall=history.groupby(['user_id','mall_id'],as_index=False)['time_stamp'].count()
user_mall.columns=['user_id','mall_id','user_mall_visit']

#提取history wifi记录
split_s=re.compile(r';')
split_o=re.compile(r'\|')
signal=re.compile(r'\-\d*')
history['wifi_infos']=history.wifi_infos.apply(lambda x:np.array(split_s.split(x))[(-np.array(signal.findall(x),dtype=int)).argsort()])
for i in range(10):
    history[i]=history.wifi_infos.apply(lambda x:x[i] if len(x)>=i+1 else None)
history_signal=pd.DataFrame(history.iloc[:,-10:].stack(level=0))
history.drop(['wifi_infos',0,1,2,3,4,5,6,7,8,9],axis=1,inplace=True)
history_signal.reset_index(inplace=True)
history_signal.columns=['id','rank','wifi_infos']
history_signal['wifi_infos']=history_signal.wifi_infos.apply(lambda x:split_o.split(x))
history_signal['wifi_id']=history_signal['wifi_infos'].apply(lambda x:x[0])
history_signal['db']=history_signal['wifi_infos'].apply(lambda x:int(x[1]))
history_signal['flag']=history_signal['wifi_infos'].apply(lambda x:x[2])
history_signal.drop('wifi_infos',axis=1,inplace=True)
history_signal=history_signal.groupby(['id','wifi_id'],as_index=False).first()
history_signal=pd.merge(history_signal,history.loc[:,['shop_id','user_id']],how='left',left_on='id',right_index=True)


history_signal['wifi_screen']=history_signal.groupby('wifi_id')['user_id'].transform('count').values
history_signal=history_signal[history_signal.wifi_screen>1].copy()
history_signal.sort_values(by=['id','db'],ascending=False,inplace=True)
history_signal['rank']=1
history_signal['rank']=history_signal.groupby('id')['rank'].cumsum().values-1


#提取train wifi记录
split_s=re.compile(r';')
split_o=re.compile(r'\|')
signal=re.compile(r'\-\d*')
train['wifi_infos']=train.wifi_infos.apply(lambda x:np.array(split_s.split(x))[(-np.array(signal.findall(x),dtype=int)).argsort()])
for i in range(10):
    train[i]=train.wifi_infos.apply(lambda x:x[i] if len(x)>=i+1 else None)
train_signal=pd.DataFrame(train.iloc[:,-10:].stack(level=0))
train.drop(['wifi_infos',0,1,2,3,4,5,6,7,8,9],axis=1,inplace=True)
train_signal.reset_index(inplace=True)
train_signal.columns=['row_id','rank','wifi_infos']
train_signal['wifi_infos']=train_signal.wifi_infos.apply(lambda x:split_o.split(x))
train_signal['wifi_id']=train_signal['wifi_infos'].apply(lambda x:x[0])
train_signal['db']=train_signal['wifi_infos'].apply(lambda x:int(x[1]))
train_signal['flag']=train_signal['wifi_infos'].apply(lambda x:x[2])
train_signal.drop('wifi_infos',axis=1,inplace=True)
train_signal=train_signal.groupby(['row_id','wifi_id'],as_index=False).first()
train_signal=pd.merge(train_signal,train.loc[:,['mall_id']],how='left',left_on='row_id',right_index=True)

train_signal=pd.merge(history_signal.loc[:,['wifi_id']].drop_duplicates(),train_signal,how='inner')
train_signal.sort_values(by=['row_id','db'],ascending=False,inplace=True)
train_signal['rank']=1
train_signal['rank']=train_signal.groupby('row_id')['rank'].cumsum().values-1

# signal-shop对
for i in range(0, 10):
    if i == 0:
        signal_shop = history_signal[history_signal['rank'] <= 2]
    if i == 9:
        signal_shop = history_signal[history_signal['rank'] >= 7]
    else:
        signal_shop = history_signal[(history_signal['rank'] <= i + 1) & (history_signal['rank'] >= i - 1)]

    # signal-shop
    signal_shop = signal_shop.groupby(['shop_id', 'wifi_id'], as_index=False)['id'].count()
    # signal-wifi总数
    signal_wifi_sum = signal_shop.groupby('wifi_id', as_index=False)['id'].sum()

    signal_shop.columns = ['shop_id', 'signal_%s' % i, 'signal_shop_%s' % i]
    signal_wifi_sum.columns = ['signal_%s' % i, 'signal_wifi_sum_%s' % i]

    signal_shop.to_csv('data_%s/signal_shop_%s.csv' % (week, i))
    signal_wifi_sum.to_csv('data_%s/signal_wifi_sum_%s.csv' % (week, i))

#connection-shop对
connection_shop=history_signal[history_signal.flag=='true']
connection_shop=connection_shop.groupby(['shop_id','wifi_id'],as_index=False)['id'].count()
connection_shop.columns=['shop_id','wifi_id','connection_shop']

#connection-wifi总数
connection_wifi_sum=connection_shop.groupby('wifi_id',as_index=False)['connection_shop'].sum()
connection_wifi_sum.columns=['wifi_id','connection_wifi_sum']

#history平均db值
history_db=history_signal.groupby(['shop_id','wifi_id'],as_index=False)['db'].agg(['median','min','max'])
history_db.columns=['db0','dbmin','dbmax']

#history wifi count
history_count=history_signal.groupby(['shop_id','wifi_id'],as_index=False)['id'].count()
history_count.columns=['shop_id','wifi_id','count0']

#生成正负样本
count_distance=pd.merge(shop.loc[:,['shop_id','mall_id']],train_signal.loc[:,['row_id','wifi_id','mall_id']],how='right')
count_distance=pd.merge(count_distance,history_count,how='inner')
count_distance=count_distance.groupby(['row_id','shop_id'])['count0'].agg(['mean','count'])
count_distance.columns=['count_distance','wifi_common']

#并入最强的10个wifi
for i in range(0,10):
    temp=train_signal.loc[train_signal['rank']==i,['row_id','wifi_id','db']]
    temp.columns=['row_id','signal_%s'%(i),'db_%s'%(i)]
    train=pd.merge(train,temp,how='left')

#写文件
train.to_csv('data_%s/train.csv'%week)
count_distance.to_csv('data_%s/count_distance.csv'%week)
location.to_csv('data_%s/location.csv'%week)
frequency.to_csv('data_%s/frequency.csv'%week)
user_shop.to_csv('data_%s/user_shop.csv'%week)
user_mall.to_csv('data_%s/user_mall.csv'%week)
connection_shop.to_csv('data_%s/connection_shop.csv'%week)
connection_wifi_sum.to_csv('data_%s/connection_wifi_sum.csv'%week)
history_db.to_csv('data_%s/history_db.csv'%week)

