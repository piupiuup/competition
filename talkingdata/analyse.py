import gc
import numpy as np
import pandas as pd

# 对正样本欠采样可能有提升
# 刷过机的手机
#

def int_time(x):
    #'2017-11-06 22:32:21'
    # 以'2017-11-07 00:00:00'为起点 转换为int数据
    try:
        return int(x[8:10])*86400 + int(x[11:13])*3600 + int(x[14:16])*60 + int(x[17:]) - 576000
    except:
        return np.nan
def groupby(data,stat,key,value,func):
    key = key if type(key)==list else [key]
    data_temp = data[key].copy()
    feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
    data_temp = data_temp.merge(feat,on=key,how='left')
    return data_temp['feat'].values
data_path = 'C:/Users/cui/Desktop/python/talkingdata/data/'


train = pd.read_csv(data_path + 'train.csv')
train['attributed_diff_time'] = train['attributed_time'].apply(int_time) - train['click_time'].apply(int_time)
del train['attributed_time']
train['ip'] = train['ip'].astype('int32')
train['app'] = train['app'].astype('int16')
train['device'] = train['device'].astype('int16')
train['os'] = train['os'].astype('int16')
train['channel'] = train['channel'].astype('int16')
train['is_attributed'] = train['is_attributed'].astype('int8')
gc.collect()
train.to_hdf('C:/Users/cui/Desktop/python/talkingdata/data/train.hdf', 'w', complib='blosc', complevel=5)


# data_path = 'C:/Users/cui/Desktop/python/talkingdata/data/'
# train = pd.read_csv(data_path + 'train.csv')
# train.sort_values('click_time',inplace=True)
# train = train[(train['click_time']>'2017-11-06 16') & (train['click_time']<'2017-11-09 16')]
# train['attributed_diff_time'] = train['attributed_time'].apply(f2) - train['click_time'].apply(f2)
# del train['attributed_time']
# train['click_time'] = train['click_time'].apply(f1)
# train['ip'] = train['ip'].astype('int32')
# train['app'] = train['app'].astype('int16')
# train['device'] = train['device'].astype('int16')
# train['os'] = train['os'].astype('int16')
# train['channel'] = train['channel'].astype('int16')
# train['is_attributed'] = train['is_attributed'].astype('int8')
# gc.collect()
# train.drop('attributed_time',axis=1).to_hdf('C:/Users/cui/Desktop/python/talkingdata/data/train.hdf', 'w', complib='blosc', complevel=5)


test = pd.read_csv(data_path + 'test_supplement.csv')
test2 = pd.read_csv(data_path + 'test.csv')
test['click_id'] = np.nan
t = test.append(test2)
t['id_sum'] = groupby(t,t,['ip','app','device','os','channel','click_time'],'click_id',sum)
t['click_id'].fillna(-1,inplace=True)
result = []
for id_sum,click_id in zip(t['id_sum'],t['click_id']):
    if (id_sum>0) & (click_id == -1):
        result.append(0)
    else:
        result.append(1)
t['flag'] = result
t = t[t.flag == 1]
t['click_id'] = t['click_id'].astype('int32')
t['ip'] = t['ip'].astype('int32')
t['app'] = t['app'].astype('int16')
t['device'] = t['device'].astype('int16')
t['os'] = t['os'].astype('int16')
t['channel'] = t['channel'].astype('int16')
t = t[['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time']]
t.to_hdf('C:/Users/cui/Desktop/python/talkingdata/data/test.hdf', 'w', complib='blosc', complevel=5)




train = pd.read_hdf(data_path + 'train.hdf')
test = pd.read_hdf(data_path + 'test.hdf')
data = train.append(test)
del train,test
gc.collect()
data['click_id'].fillna(-1,inplace=True)
data['click_id'] = data['click_id'].astype('int32')
data['ip'] = data['ip'].astype('int32')
data['app'] = data['app'].astype('int16')
data['device'] = data['device'].astype('int16')
data['os'] = data['os'].astype('int16')
data['channel'] = data['channel'].astype('int16')
data['click_time'] = data['click_time'].apply(f3).astype('int32')
data['attributed_diff_time'] = data['attributed_diff_time'].astype('float32')
data['is_attributed'] = data['is_attributed'].astype('float16')
data['date'] = (data['click_time']//86400).astype('int8')
data['hour'] = (data['click_time']%86400//3600).astype('int8')
data['hour2'] = ((data['click_time']+1800)&86400//3600).astype('int8')
data['date_hour'] = (data['click_time']//3600).astype('int8')
data['date_hour_2'] = ((data['click_time']+1800)//3600).astype('int8')
data['date_minute30'] = (data['click_time']//1800).astype('int16')
data['date_minute30_2'] = ((data['click_time']+900)//1800).astype('int16')
data['date_minute15'] = (data['click_time']//900).astype('int16')
data['date_minute15_2'] = ((data['click_time']+300)//900).astype('int16')
data['date_minute15_3'] = ((data['click_time']+600)//900).astype('int16')
data['date_minute10'] = (data['click_time']//600).astype('int16')
data['date_minute10_2'] = ((data['click_time']+300)//600).astype('int16')
data['date_minute5'] = (data['click_time']//300).astype('int16')
data['date_minute4'] = (data['click_time']//240).astype('int16')
data['date_minute4_2'] = ((data['click_time']+120)//240).astype('int16')
data['date_minute3'] = (data['click_time']//180).astype('int16')
data['date_minute2'] = (data['click_time']//120).astype('int16')
data['date_minute'] = (data['click_time']//60).astype('int16')
data['minute15'] = (data['click_time']%86400//900).astype('int16')
data['minute'] = (data['click_time']//900).astype('int16')


# data['attributed_diff_time'] = data['attributed_diff_time'].astype('int32')
data.sort_values('click_time',inplace=True,ascending=True)
# data[(data['click_time']>=0) & (data['click_time']<345600)]
data.reset_index(inplace=True,drop=True)

for c in data.columns:
    print('储存{}...'.format(c))
    result_path = data_path + c + '.hdf'
    data[c].to_hdf(result_path, 'w', complib='blosc', complevel=5)
# data['is_attributed'] = data['is_attributed'].astype('int8')


def read(columns):
    data = pd.DataFrame()
    for c in columns:
        result_path = data_path + c + '.hdf'
        data[c] = pd.read_hdf(result_path)
    return data
columns = ['click_id','ip','os','device','app','channel','click_time','attributed_diff_time','is_attributed']
data = read(columns)

train = pd.read_hdf(data_path + 'train_sub.hdf')
train = train[train['click_time'].str[11:13].astype(int).isin([4,5,9,10,13,14])]



t
























