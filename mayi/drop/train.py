import pandas as pd
import numpy as np
import pickle

shop=pd.read_csv('ccf_first_round_shop_info.csv')
shop=shop.loc[:,['shop_id','mall_id','longitude','latitude']]
shop.columns=['shop_id','mall_id','longitude0','latitude0']

#读表
week=1
train=pd.read_csv('data_%s/train.csv'%week,index_col=0)
history_db=pd.read_csv('data_%s/history_db.csv'%week)
count_distance=pd.read_csv('data_%s/count_distance.csv'%week)
location=pd.read_csv('data_%s/location.csv'%week,index_col=0)
frequency=pd.read_csv('data_%s/frequency.csv'%week,index_col=0)
user_shop=pd.read_csv('data_%s/user_shop.csv'%week,index_col=0)
user_mall=pd.read_csv('data_%s/user_mall.csv'%week,index_col=0)
connection_shop=pd.read_csv('data_%s/connection_shop.csv'%week,index_col=0)
connection_wifi_sum=pd.read_csv('data_%s/connection_wifi_sum.csv'%week,index_col=0)

train.columns



train.drop(['time_stamp'],inplace=True,axis=1)


train['lo1']=(train.longitude//0.00001).astype(int)
train['la1']=(train.latitude//0.00001).astype(int)
train['lo5']=(train.longitude//0.00005).astype(int)
train['la5']=(train.latitude//0.00005).astype(int)
train['lo10']=(train.longitude//0.0001).astype(int)
train['la10']=(train.latitude//0.0001).astype(int)


#生成正负样本
columns=list(train.columns)
columns[2]='shop_id_true'
train.columns=columns
train=pd.merge(shop,train,how='right')
train['label']=1*(train.shop_id==train.shop_id_true)
train.drop(['shop_id_true'],axis=1,inplace=True)
train=train[(np.random.rand(len(train))+train.label)>0.95]


#并入block_shop
for i in ['1','5','10']:
    a=pd.read_csv('data_%s/block_total%s.csv'%(week,i),index_col=0)
    train=pd.merge(train,a,how='left')
    a=pd.read_csv('data_%s/block_shop%s.csv'%(week,i),index_col=0)
    train=pd.merge(train,a,how='left')
    train['block_%s'%i]=train['block_shop%s'%i]/train['block_total%s'%i]
    train.drop(['lo%s'%i,'la%s'%i,'block_shop%s'%i],inplace=True,axis=1)



#并入wifi距离
train=pd.merge(train,count_distance,how='left')


#计算和采集位置的距离
train['distance0']=np.sqrt(np.square(1000*train.longitude-1000*train.longitude0)+np.square(1000*train.latitude-1000*train.latitude0))
train.drop(['longitude0','latitude0'],axis=1,inplace=True)


#正负样本比例
train.label.mean()


#计算和中位数位置的距离
location.columns=['shop_id','longitude_shop','latitude_shop']
train=pd.merge(train,location,how='left')
train['distance']=np.sqrt(np.square(1000*train.longitude-1000*train.longitude_shop)+np.square(1000*train.latitude-1000*train.latitude_shop))
train.drop(['longitude','latitude','longitude_shop','latitude_shop'],axis=1,inplace=True)


#frequency并表
train=pd.merge(train,frequency,how='left')


#user-shop对
train=pd.merge(train,user_shop,how='left')
train=pd.merge(train,user_mall,how='left')
train['user_shop_over_mall']=train.user_shop_visit/train.user_mall_visit

# signal-shop对
for i in range(0, 10):
    signal_shop = pd.read_csv('data_%s/signal_shop_%s.csv' % (week, i), index_col=0)
    signal_wifi_sum = pd.read_csv('data_%s/signal_wifi_sum_%s.csv' % (week, i), index_col=0)

    train = pd.merge(train, signal_shop, how='left')
    train = pd.merge(train, signal_wifi_sum, how='left')
    train['signal_shop_%s_over_wifi' % i] = train['signal_shop_%s' % i] / train['signal_wifi_sum_%s' % i]

    history_db.columns = ['shop_id', 'signal_%s' % i, 'db0_%s' % i, 'dbmin_%s' % i, 'dbmax_%s' % i]
    train = pd.merge(train, history_db, how='left')
    train['dbd_%s' % i] = train['db_%s' % i] - train['db0_%s' % i]
    train['dbmax_%s' % i] = train['db_%s' % i] - train['dbmax_%s' % i]
    train['dbmin_%s' % i] = train['db_%s' % i] - train['dbmin_%s' % i]

    train.drop(['db_%s' % i, 'db0_%s' % i, 'signal_shop_%s' % i, 'signal_%s' % i], axis=1, inplace=True)


train.fillna(0,inplace=True)
f=5
for i in range(0,10):
    train['mean_encoding_%s'%i]=f/(train['signal_wifi_sum_%s'%i]+f)*train.shop_frequency+train['signal_wifi_sum_%s'%i]/(train['signal_wifi_sum_%s'%i]+f)*train['signal_shop_%s_over_wifi'%i]
    #train.drop(['signal_shop_%s_over_wifi'%i,'signal_wifi_sum_%s'%i],axis=1,inplace=True)


L1,L2,C=0,0,0
for i in range(0,10):
    L1+=np.absolute(train['dbd_%s'%i].fillna(0))
    L2+=np.square(train['dbd_%s'%i].fillna(0))
    C+=train['dbd_%s'%i].notnull()
    if i==9:
        train['L1_%s'%i]=L1/C
        train['L2_%s'%i]=L2/C


#connection-shop对
connection_shop.columns=['shop_id','connection','connection_shop']
connection_wifi_sum.columns=['connection','connection_wifi_sum']
train=pd.merge(train,connection_shop,how='left')
train=pd.merge(train,connection_wifi_sum,how='left')

train['connection_shop_over_wifi']=train.connection_shop/train.connection_wifi_sum
train.drop(['connection','connection_shop'],axis=1,inplace=True)


train.fillna(0,inplace=True)

train['mean_encoding_c']=f/(train.connection_wifi_sum+f)*train.shop_frequency+train.connection_wifi_sum/(train.connection_wifi_sum+f)*train.connection_shop_over_wifi
train.drop(['connection_shop_over_wifi'],axis=1,inplace=True)

multi=pd.read_csv('local_multi.csv',index_col=0)
train=pd.merge(train,multi,how='left')

features=[
        'p_multi',
        'block_1','block_5','block_10',
        'block_total1','block_total5','block_total10',
        'distance0', 'distance', 'shop_visit', 'shop_frequency',
        'user_shop_visit', 'user_mall_visit', 'user_shop_over_mall',
        'signal_shop_0_over_wifi','signal_shop_1_over_wifi','signal_shop_2_over_wifi','signal_shop_3_over_wifi','signal_shop_4_over_wifi','signal_shop_5_over_wifi','signal_shop_6_over_wifi','signal_shop_7_over_wifi','signal_shop_8_over_wifi','signal_shop_9_over_wifi',
        'signal_wifi_sum_0','signal_wifi_sum_1','signal_wifi_sum_2','signal_wifi_sum_3','signal_wifi_sum_4','signal_wifi_sum_5','signal_wifi_sum_6','signal_wifi_sum_7','signal_wifi_sum_8','signal_wifi_sum_9',
        'dbd_0','dbd_1','dbd_2','dbd_3','dbd_4','dbd_5','dbd_6','dbd_7','dbd_8','dbd_9',
        'L1_9','L2_9','count_distance','wifi_common',
        'dbmin_0','dbmin_1','dbmin_2','dbmin_3','dbmin_4','dbmin_5','dbmin_6','dbmin_7','dbmin_8','dbmin_9',
        'dbmax_0','dbmax_1','dbmax_2','dbmax_3','dbmax_4','dbmax_5','dbmax_6','dbmax_7','dbmax_8','dbmax_9',
         ]


from xgboost import XGBClassifier
model=XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=600)
model.fit(train.loc[:,features],train.loc[:,'label'])
f = open('data_%s/model_online_withoutmulti'%week,'wb')
pickle.dump(model,f)
f.close()


model.feature_importances_


