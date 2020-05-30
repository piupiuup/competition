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
        return int(x[8:10]) * 86400 + int(x[11:13]) * 3600 + int(x[14:16]) * 60 + int(x[17:]) - 576000
    except:
        return np.nan
def groupby(data,stat,key,value,func):
    key = key if type(key)==list else [key]
    data_temp = data[key].copy()
    feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
    data_temp = data_temp.merge(feat,on=key,how='left')
    return data_temp['feat'].values
data_path = 'D:/talkingdata2/data/'


train = pd.read_csv(data_path + 'train.csv')
train['attributed_diff_time'] = train['attributed_time'].apply(int_time) - train['click_time'].apply(int_time)
train['click_time'] = train['click_time'].apply(int_time)
del train['attributed_time']
gc.collect()



test_old = pd.read_csv(data_path + 'test_supplement.csv')
test_old['click_id'] = -1
test = pd.read_csv(data_path + 'test.csv')
test_old['click_time'] = test_old['click_time'].apply(int_time)
test['click_time'] = test['click_time'].apply(int_time)
t1 = test_old[test_old['click_time']<302400].copy()
t2 = test[(test['click_time']>=302400) & (test['click_time']<=309600)].copy()
t3 = test_old[(test_old['click_time']>309600) & (test_old['click_time']<320400)].copy()
t4 = test[(test['click_time']>=320400) & (test['click_time']<=327600)].copy()
t5 = test_old[(test_old['click_time']>327600) & (test_old['click_time']<334800)].copy()
t6 = test[(test['click_time']>=334800) & (test['click_time']<=342000)].copy()
t7 = test_old[(test_old['click_time']>342000)].copy()
data = train.append(t1).append(t2).append(t3).append(t4).append(t5).append(t6).append(t7)
data.reset_index(inplace=True,drop=True)
data.reset_index(inplace=True)
data.sort_values(['click_time','index'],inplace=True,ascending=True)
data.reset_index(inplace=True,drop=True)
del data['index']

del train,test
gc.collect()
data['click_id'].fillna(-1,inplace=True)
data['click_id'] = data['click_id'].astype('int32')
data['ip'] = data['ip'].astype('int32')
data['app'] = data['app'].astype('int16')
data['device'] = data['device'].astype('int16')
data['os'] = data['os'].astype('int16')
data['channel'] = data['channel'].astype('int16')
data['click_time'] = data['click_time'].astype('int32')
data['attributed_diff_time'] = data['attributed_diff_time'].astype('float32')
data['is_attributed'] = data['is_attributed'].astype('float16')
data['date'] = (data['click_time']//86400).astype('int8')
data['hour'] = (data['click_time']%86400//3600).astype('int8')
data['hour_2'] = ((data['click_time']%86400+1800)//3600).astype('int8')
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





























