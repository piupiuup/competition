import pandas as pd
import numpy as np
import Geohash
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

cache_path = 'F:/mobike_cache2/'
train_path = r'C:\Users\csw\Desktop\python\mobike\data\train.csv'
test_path = r'C:\Users\csw\Desktop\python\mobike\data\test.csv'

arr1 = ['00000','00001','00010','00011','00100','00101','00110','00111',
        '01000','01001','01010','01011','01100','01101','01110','01111',
        '10000','10001','10010','10011','10100','10101','10110','10111',
        '11000','11001','11010','11011','11100','11101','11110','11111']
arr2 = ['0','1','2','3','4','5','6','7','8','9','b','c','d','e','f','g',
        'h','j','k','m','n','p','q','r','s','t','u','v','w','x','y','z']
d = dict(zip(arr2,arr1))
# 解码字典
def get_loc_dict():
    dump_path = cache_path + 'loc_dict.pkl'
    if os.path.exists(dump_path):
        loc_dict = pickle.load(open(dump_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        locs = list(
            set(train['geohashed_start_loc']) | set(train['geohashed_end_loc']) | set(test['geohashed_start_loc']))
        deloc = []
        for loc in locs:
            deloc.append(Geohash.decode_exactly(loc)[:2])
        loc_dict = dict(zip(locs, deloc))
        pickle.dump(loc_dict, open(dump_path, 'wb+'))
    return loc_dict

# 计算两点之间距离
def cal_distance(lat1, lon1, lat2, lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx ** 2 + Ly ** 2) ** 0.5
    return L


# 计算两点之间距离
def cal_mht_distance(lat1, lon1, lat2, lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = Lx + Ly
    return L
# 获取骑行距离
def get_distance(sample, sloc='geohashed_start_loc', eloc='geohashed_end_loc'):
    loc_dict = get_loc_dict()
    geohashed_loc = sample[[sloc, eloc]].values
    distance = []
    mht_distance = []
    for i in geohashed_loc:
        loc1, loc2 = i
        if (loc1 is np.nan) | (loc2 is np.nan):
            distance.append(np.nan)
            mht_distance.append(np.nan)
        else:
            lat1, lon1 = loc_dict[loc1]
            lat2, lon2 = loc_dict[loc2]
            distance.append(cal_distance(lat1, lon1, lat2, lon2))
            mht_distance.append(cal_mht_distance(lat1, lon1, lat2, lon2))
    result = sample.copy()
    result['distance'] = distance
    result['mht_distance'] = mht_distance
    result = result[['distance', 'mht_distance']]
    return result
# 对地点解码
def decode_geohash(data):
    result = data.copy()
    start_loc = np.array(result['geohashed_start_loc'].apply(lambda x:Geohash.decode(x)).tolist())
    result['start_loc_lat'] = start_loc[:, 0]
    result['start_loc_lon'] = start_loc[:, 1]
    if 'geohashed_end_loc' in data.columns:
        end_loc = np.array(result['geohashed_end_loc'].apply(lambda x: Geohash.decode(x)).tolist())
        result['end_loc_lat'] = end_loc[:, 0]
        result['end_loc_lon'] = end_loc[:, 1]
    return result

# 对geohash解码
def decode(geohash):
    result = ''
    for i in list(geohash):
        result += d[i]
    return result



# 读取数据
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

############数据探索部分#############
# 数据
print(train.head())
print(test.head())

# 训练集各种特征的种类数
print(train['orderid'].nunique())
print(train['userid'].nunique())
print(train['bikeid'].nunique())
print(train['biketype'].value_counts())
print(train['geohashed_start_loc'].nunique())
print(train['geohashed_end_loc'].nunique())

# 测试集各种特征的种类数
print(test['orderid'].nunique())
print(test['userid'].nunique())
print(test['bikeid'].nunique())
print(test['biketype'].value_counts())
print(test['geohashed_start_loc'].nunique())
print(test['geohashed_end_loc'].nunique())

#训练集起始时间，终止时间
print('训练集起始时间：%s' %train['starttime'].min())
print('训练集结束时间：%s' %train['starttime'].max())

#测试集起始时间，终止时间
print('测试集起始时间：%s' %test['starttime'].min())
print('测试集结束时间：%s' %test['starttime'].max())


# 训练集中有多少用户
print('训练集中用户个数：%d' %(train['userid'].nunique()))
# 测试集中有多少用户
print('测试集中用户个数：%d' %(test['userid'].nunique()))
# 测试机中有多少用户有记录
print('测试集中用户有记录的比例：%f' %(len(set(test['userid'].unique())&set(train['userid'].unique()))/test['userid'].nunique()))

# 训练集中出发地点个数
print('训练集中出发地点个数：%f' %train['geohashed_start_loc'].nunique())
# 训练集中目的地点个数
print('训练集中目的地点个数：%f' %train['geohashed_end_loc'].nunique())
# 训练集中地点总个数
print('训练集中地点总个数：%f' %len(set(train['geohashed_start_loc'].unique())|set(train['geohashed_end_loc'].unique())))
# 测试集中出发地点个数
print('测试集中出发地点个数：%f' %test['geohashed_start_loc'].nunique())
# 测试集中出发地点个数有记录的比例
train_loc_set =  set(train['geohashed_start_loc'].unique())|set(train['geohashed_end_loc'].unique())
test_loc_set = set(test['geohashed_start_loc'].unique())
print('测试集中出发地点个数有记录的比例：%f' % (len(test_loc_set&train_loc_set)/len(test_loc_set)))
del test_loc_set,train_loc_set

# 训练集中每辆车平均每天被骑次数
print(train.shape[0]/train['bikeid'].nunique()/14)
# 测试集中每辆车平均每天被骑次数
print(test.shape[0]/test['bikeid'].nunique()/7)

# 训练集中每人每天骑车次数
print(train.shape[0]/train['userid'].nunique()/14)
# 测试集中每人每天骑车次数
print(test.shape[0]/test['userid'].nunique()/7)

# 用户从又返回出发地的比例
print('又返回出发地用户的比例：%f' % sum(train['geohashed_start_loc']==train['geohashed_end_loc'])/train.shape[0])

# 骑行距离绘图
train[['distance', 'mht_distance']] = get_distance(train)
plt.figure(figsize=(12,8))
sns.distplot(train.distance.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()

# 用户衔接的比例
def n_link(data):
    return sum(data['flag'].values[:-1])
train.sort_values(['userid','starttime'],inplace=True)
temp = train.copy()
a = list(train['geohashed_start_loc'].values)[1:]
a.append(np.nan)
temp['flag'] = a
temp['flag'] = temp['flag']==temp['geohashed_end_loc']
n = sum(temp.groupby('userid').apply(lambda x:n_link(x)))
print('有%f的目的地可以用后面用户的出发地衔接起来' %(n/train.shape[0]))

# 自行车衔接的比例
def n_link(data):
    return sum(data['flag'].values[:-1])
train.sort_values(['bikeid','starttime'],inplace=True)
temp = train.copy()
a = list(train['geohashed_start_loc'].values)[1:]
a.append(np.nan)
temp['flag'] = a
temp['flag'] = temp['flag']==temp['geohashed_end_loc']
n = sum(temp.groupby('bikeid').apply(lambda x:n_link(x)))
print('有%f的目的地可以用后面用户的出发地衔接起来' %(n/train.shape[0]))

# 对比自行车12行驶距离是否有明显差距
def distance(row):
    lat1 = row['start_loc_lat']
    lon1 = row['start_loc_lon']
    lat2 = row['end_loc_lat']
    lon2 = row['end_loc_lon']
    dx = np.absolute(lon1 - lon2)  # 经度差
    dy = np.absolute(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L
temp = decode_geohash(train)
temp['distance'] = temp.apply(lambda x:distance(x),axis=1)
temp.groupby('biketype')['distance'].mean()

# 绘图
def draw(data, n_sep=300, mult=1, shape=None):
    '''
    :param data: dataframe格式，包含经纬度信息
    :param n_sep: 像素点个数 n_sep * n_sep
    :param mult: 颜色深浅
    :param shape: 显示的区域大小，None时显示全图
    '''
    if shape is not None:
        lon_std = data['start_loc_lon'].std()
        lon_ave = data['start_loc_lon'].mean()
        lat_std = data['start_loc_lat'].std()
        lat_ave = data['start_loc_lat'].mean()
        temp = data[(data['start_loc_lon'] > (lon_ave - shape * lon_std)) & (data['start_loc_lon'] < (lon_ave + shape * lon_std))]
        temp = temp[(temp['start_loc_lat'] > (lat_ave - shape * lat_std)) & (temp['start_loc_lat'] < (lat_ave + shape * lat_std))]
    else:
        temp = data.copy()
    sep = (temp['start_loc_lon'].max()-temp['start_loc_lon'].min())/n_sep
    temp['start_loc_lon'] = temp['start_loc_lon'] // sep
    temp['start_loc_lat'] = temp['start_loc_lat'] // sep
    matrix = temp.groupby(['start_loc_lon','start_loc_lat'])['userid'].count().unstack().fillna(0)

    im = Image.fromarray(matrix.values*mult)
    im = im.convert('RGB')
    im.save(r'C:\Users\csw\Desktop\a.jpg', format='jpeg')

# 根据用户连续构造答案
all = pd.concat([train,test])
all.sort_values(['userid','starttime'],inplace=True)
all['user_rank'] = np.arange(all.shape[0])
min_user_rank = all.groupby('userid',as_index=False)['user_rank'].agg({'min_user_rank':min})
all = pd.merge(all,min_user_rank,on='userid',how='left')
all['user_rank'] = all['user_rank'] - all['min_user_rank'] + 1
del all['min_user_rank']
temp = all[['userid','user_rank','geohashed_start_loc']].copy()
temp.rename(columns={'geohashed_start_loc':'pred_loc'},inplace=True)
temp['user_rank'] = temp['user_rank']-1
all = pd.merge(all,temp,on=['userid','user_rank'],how='left')
output = pd.merge(test[['orderid']],all.loc[:,['orderid','pred_loc']],on='orderid',how='left')
output['pred_loc2'] = np.nan
output['pred_loc3'] = np.nan
output.fillna('wx4gg1x',inplace=True)

# 根据自行车连续构造答案
all = pd.concat([train,test])
all.sort_values(['bikeid','starttime'],inplace=True)
all['bike_rank'] = np.arange(all.shape[0])
min_bike_rank = all.groupby('bikeid',as_index=False)['bike_rank'].agg({'min_bike_rank':min})
all = pd.merge(all,min_bike_rank,on='bikeid',how='left')
all['bike_rank'] = all['bike_rank'] - all['min_bike_rank'] + 1
del all['min_bike_rank']
temp = all[['bikeid','bike_rank','geohashed_start_loc']].copy()
temp.rename(columns={'geohashed_start_loc':'pred_loc'},inplace=True)
temp['bike_rank'] = temp['bike_rank']-1
all = pd.merge(all,temp,on=['bikeid','bike_rank'],how='left')
output = pd.merge(test[['orderid']],all.loc[:,['orderid','pred_loc']],on='orderid',how='left')
output['pred_loc2'] = np.nan
output['pred_loc3'] = np.nan
output.fillna('wx4gg1x',inplace=True)