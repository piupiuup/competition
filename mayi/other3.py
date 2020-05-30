import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from sklearn import  preprocessing
from collections import defaultdict

t0 = time.time()
cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

# 商店对应的连接wiif
def get_if_connect(wifi_infos):
    if wifi_infos != '':
        for wifi_info in wifi_infos.split(';'):
            bssid,signal,flag = wifi_info.split('|')
            if flag == 'true':
                return 1
    return 0

def get_wifi_signal(wifi_infos):
    result = []
    if wifi_infos != '':
        for wifi_info in wifi_infos.split(';'):
            bssid,signal,flag = wifi_info.split('|')
            result.append(int(signal))
    return result
# 线下测评
def acc(data,name='shop_id'):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:
        print('没有发现真实数据，无法测评')
    return sum(data['row_id'].map(true)==data[name])/data.shape[0]


# 制作训练集
def make_feat(data):
    mall_id = data.mall_id.values[0]
    print('mall_id为：{}'.format(mall_id))
    df_train_path = cache_path + 'multi_train_feat_{}.hdf'.format(mall_id)
    df_test_path = cache_path + 'multi_test_feat_{}.hdf'.format(mall_id)
    lbl_path = cache_path + 'lbl_{}.hdf'.format(mall_id)
    if os.path.exists(df_train_path) & os.path.exists(df_test_path) & 0:
        df_train = pd.read_hdf(df_train_path, 'w')
        df_test = pd.read_hdf(df_test_path, 'w')
        lbl = pickle.load(open(lbl_path,'+rb'))
    else:
        l = []
        wifi_dict = defaultdict(lambda : 0)
        for row in data.itertuples():
            r = {}
            wifi_list = [wifi.split('|') for wifi in row.wifi_infos.split(';')]
            for i in wifi_list:
                r[i[0]] = int(i[1])
                # if row.shop_id is np.nan:
                wifi_dict[i[0]] += 1
            l.append(r)
        delate_wifi = []
        for i in wifi_dict:
            if wifi_dict[i] < 15:
                delate_wifi.append(i)
        m = []
        for row in l:
            new = {}
            for n in row.keys():
                if n not in delate_wifi:
                    new[n] = row[n]
            m.append(new)
        df_data = pd.concat([data, pd.DataFrame(m)], axis=1)
        df_train = df_data[df_data.shop_id.notnull()]
        df_test = df_data[df_data.shop_id.isnull()]
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train['shop_id'].values))
        df_train['label'] = lbl.transform(list(df_train['shop_id'].tolist()))
        df_train.to_hdf(df_train_path, 'w', complib='blosc', complevel=5)
        df_test.to_hdf(df_test_path, 'w', complib='blosc', complevel=5)
        pickle.dump(lbl,open(lbl_path,'+wb'))
    return df_train,df_test,lbl


df=pd.read_csv(train_path)
shop=pd.read_csv(shop_path)
test=pd.read_csv(test_path)
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
train=pd.concat([df,test])

train['time_stamp']=pd.to_datetime(train['time_stamp'])
train['week'] = train['time_stamp'].dt.dayofweek
train['hour'] = train['time_stamp'].dt.hour
train['if_connect'] = train['wifi_infos'].apply(get_if_connect)
train['signals'] = train['wifi_infos'].apply(get_wifi_signal)
train['wifi_signals_mean'] = train['signals'].apply(lambda x: np.mean(x))
train['wifi_signals_median'] = train['signals'].apply(lambda x: np.median(x))
train['wifi_signals_max'] = train['signals'].apply(lambda x: np.max(x))
train['wifi_signals_min'] = train['signals'].apply(lambda x: np.min(x))
train['wifi_signals_std'] = train['signals'].apply(lambda x: np.std(x))
del train['signals']
mall_list = shop.mall_id.unique().tolist()

result = pd.DataFrame()
for mall in tqdm(mall_list[:20]):
    train_sub=train[train.mall_id==mall].reset_index(drop=True)
    df_train,df_test,lbl = make_feat(train_sub)
    print('特征矩阵大小为：{}'.format(df_train.shape))

    num_class=df_train['label'].nunique()
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'eval_metric': ['mlogloss', 'merror'],
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.85,
        'min_child_weight': 0.5,
        'eta': 0.04,
        'missing': -999,
        'num_class': num_class,
        'nthread': 6,
        'seed': 2016,
        'silent': 1
    }
    feature = [x for x in df_train.columns if
               x not in ['user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]

    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    num_rounds = 285
    model = xgb.train(params, xgbtrain, num_rounds)
    pred = model.predict(xgbtest)
    pred = pd.DataFrame(pred, columns=list(lbl.inverse_transform(np.arange(num_class))), index=df_test['row_id'].tolist())
    pred = pred.stack().reset_index()
    pred.columns = ['row_id','shop_id','multi_pred']
    result = pd.concat([result, pred])
    # result['row_id']=result['row_id'].astype('int')
    # result.to_csv(path+'sub.csv',index=False)
# result.to_hdf(r'C:\Users\csw\Desktop\python\mayi\data\multi_pred_25-31.hdf', 'w', complib='blosc', complevel=5)
# result = pd.read_hdf(cache_path+'multi_pred.hdf', 'w')
result.sort_values('multi_pred',inplace=True)
result = result.drop_duplicates('row_id',keep='last')
print('准确率为：{}'.format(acc(result,'shop_id')))
print('用时{}s'.format(t0-time.time()))











