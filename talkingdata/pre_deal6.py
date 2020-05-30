import gc
import numpy as np
import pandas as pd

# 对正样本欠采样可能有提升
# 刷过机的手机
#

def f1(x):
    try:
        hour = int(x[11:13])+8
        date = int(x[8:10])
        if hour>23:
            hour = hour-24
            date += 1
        return x[:8] + ('%02d' % date) + ' ' + ('%02d' % hour) + x[-6:]
    except:
        return np.nan
def f2(x):
    try:
        return int(x[8:10])*86400 + int(x[11:13])*3600 + int(x[14:16])*60 + int(x[17:19])
    except:
        return np.nan
def f3(x):
    #'2017-11-06 22:32:21'
    # 以'2017-11-07 00:00:00'为起点 转换为int数据
    return int(x[8:10])*86400 + int(x[11:13])*3600 + int(x[14:16])*60 + int(x[17:]) - 576000
def groupby(data,stat,key,value,func):
    key = key if type(key)==list else [key]
    data_temp = data[key].copy()
    feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
    data_temp = data_temp.merge(feat,on=key,how='left')
    return data_temp['feat'].values
data_path = 'D:/talkingdata/data/'


train = pd.read_csv(data_path + 'train.csv')
train['attributed_diff_time'] = train['attributed_time'].apply(f2) - train['click_time'].apply(f2)
del train['attributed_time']
train['ip'] = train['ip'].astype('int32')
gc.collect()



test = pd.read_csv(data_path + 'test_supplement.csv')
data = train.append(test)
# data.sort_values('click_time',inplace=True,ascending=True)
data.reset_index(inplace=True,drop=True)

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



for c in data.columns:
    print('储存{}...'.format(c))
    result_path = data_path + c + '.hdf'
    data[c].to_hdf(result_path, 'w', complib='blosc', complevel=5)
# data['is_attributed'] = data['is_attributed'].astype('int8')





























