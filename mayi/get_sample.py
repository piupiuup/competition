from mayi.feat2 import *
import lightgbm as lgb

cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')

# 计算实际距离
def cal_distance(lat1,lon1,lat2,lon2):
    lat1 = float(lat1);lon1 = float(lon1);lat2 = float(lat2);lon2 = float(lon2)
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

def rank(data, key, values, ascending=True):
    data.sort_values([key,values],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(key,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=key,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 用户去过的店铺
def get_user_visit_shop(train,test):
    data = train[['user_id','shop_id','mall_id']].drop_duplicates()
    data.columns=['user_id','pre_shop_id','mall_id']
    t_data = pd.merge(test,data,how='left',on=['user_id','mall_id'])
    return t_data

# 用户gps离商店gps中位数最近的三个
def get_nearest_shop(train,test):
    t_data1 = train.groupby('shop_id')['longitude'].median()
    t_data1 = t_data1.reset_index()
    t_data1.columns=['pre_shop_id','pre_longitude_median']

    t_data2 = train.groupby('shop_id')['latitude'].median()
    t_data2 = t_data2.reset_index()
    t_data2.columns = [ 'pre_shop_id', 'pre_latitude_median']
    t_data = pd.merge(t_data1,t_data2,on='pre_shop_id')

    shop = pd.read_csv(shop_path)
    shop.rename(columns={'shop_id':'pre_shop_id','longitude':'pre_s_longitude','pre_s_latitude':'pre_s_latitude'},inplace=True)
    shop = shop[['pre_shop_id', 'mall_id']]
    data = pd.merge(test,shop,how='left',on='mall_id')
    data = pd.merge(data,t_data,how='left',on='pre_shop_id')
    data = data[~data.pre_latitude_median.isnull()]
    fuck = data[['longitude', 'latitude', 'pre_longitude_median', 'pre_latitude_median']].values
    res = []
    for i in fuck:
        res.append(cal_distance(i[0], i[1], i[2], i[3]))
    data['dis'] = res
    data = data.sort_values(by='dis',ascending=False)
    del data['dis']
    del data['pre_longitude_median']
    del data['pre_latitude_median']
    data = data.groupby('row_id').tail(3)
    return data

# 用户gps knn 前50个，重复的很多
def get_knn_gps(train,test,number):
    clf = neighbors.KNeighborsClassifier(number,p=1)
    train_data = train.reset_index(drop=True).copy()
    clf = clf.fit(train_data[['longitude', 'latitude']], train_data['shop_id'])
    test_data = test.reset_index(drop=True).copy()
    dia,indice = clf.kneighbors(test_data[['longitude', 'latitude']])

    shop_res = list(train_data['shop_id'])
    fuck = test_data[['row_id']].values
    res1=[]
    res2=[]

    for i in range(0,len(fuck)):
        for j in range (0,len(indice[i])):
            res1.append(fuck[i][0])
            res2.append(shop_res[indice[i][j]])

    data = pd.DataFrame()
    data['row_id'] = res1
    data['pre_shop_id'] = res2
    data = pd.merge(test,data,how='left',on='row_id')
    return data

# .//用户当前wifi，和商店中出现次数最多前15个WiFi有交集的所有商店
def get_wifi_nearest(train,test,train_wifi,test_wifi):
    train1 = pd.merge(train, train_wifi, how='left', on='row_id')
    fuck = train1[['shop_id', 'wifi_infos']].values
    res1 = []
    res2 = []
    res3 = []
    for i in fuck:
        wifis = i[1].split(';')
        for j in wifis:
            wifi = j.split('|')
            res1.append(i[0])
            res2.append(wifi[0])
            res3.append((int)(wifi[1]))

    t_data = pd.DataFrame()
    t_data['shop_id'] = res1
    t_data['wifi_id'] = res2
    t_data['sign'] = res3

    data1 = t_data.groupby(['shop_id', 'wifi_id'])['sign'].mean()
    data1 = data1.reset_index()
    data1.columns = ['pre_shop_id', 'wifi_id', 'sign_mean']

    data2 = t_data.groupby(['shop_id', 'wifi_id'])['sign'].count()
    data2 = data2.reset_index()
    data2.columns = ['pre_shop_id', 'wifi_id', 'shop_wifi_count']

    data1 = pd.merge(data1, data2, how='left', on=['pre_shop_id', 'wifi_id'])

    data1 = rank(data1, 'pre_shop_id', 'shop_wifi_count', False)

    data1 = data1[data1['rank'] < 15]

    ma1={}
    fuck =data1[['pre_shop_id','wifi_id','sign_mean']].values
    for i in fuck:
        if(i[0] not in ma1):
                    ma1[i[0]]={}
        ma1[i[0]][i[1]]=i[2]

    data = pd.merge(test, test_wifi, how='left', on='row_id')
    fuck = data[['row_id', 'wifi_infos']].values
    ma2={}
    for i in fuck:
        ma2[i[0]] = {}
        wifis = i[1].split(';')
        for j in wifis:
           wifi = j.split('|')
           ma2[i[0]][wifi[0]] = wifi[1]
    shop = pd.read_csv(shop_path)
    shop.rename(columns={'shop_id': 'pre_shop_id', 'longitude': 'pre_s_longitude', 'pre_s_latitude': 'pre_s_latitude'},
                inplace=True)
    shop = shop[['pre_shop_id', 'mall_id']]
    data = pd.merge(test,shop,how='left',on='mall_id')

    #data = data[~data.pre_shop_id.isnull()]
    fuck=data[['row_id','pre_shop_id']].values
    res1=[]
    res2=[]
    for i in fuck:
        cnt1 = 0
        cnt2 = 0
        if(i[1] not in ma1):
            res1.append(cnt1)
            res2.append(cnt2)
            continue
        for j in ma1[i[1]]:

            if(j in ma2[i[0]]):
                cnt1+=1

                x1 = (float)(ma1[i[1]][j])
                x2 = (float)(ma2[i[0]][j])
               # print(ma1[i[1]][j], ma2[i[0]][j],(x1-x2)*(x1-x2))

                cnt2 +=(x1-x2)*(x1-x2)

        res1.append(cnt1)
        res2.append(cnt2)

    data['count1'] = res1
    data['count2'] = res2
    data = data[data.count1!=0]
    print(data['count2'])
    data['count3'] = data['count2']/data['count1']
    data = data.sort_values('count1',ascending=True)

  #  data = data.groupby(['row_id']).tail(20)
    print(data)
    del data['count1']
    del data['count2']
    del data['count3']
    return data
