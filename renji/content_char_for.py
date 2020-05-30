import gc
import re
import sys
import time
import jieba
import pickle
import string
import codecs
import hashlib
import os.path
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.stats as ss
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

train_hdf_path = r'C:\Users\csw\Desktop\python\360\data2\train.hdf'
test_hdf_path = r'C:\Users\csw\Desktop\python\360\data2\evaluation_public.hdf'
cache_path = 'F:/360_cache/'
new = True

cmark = '。，、；：？！“”’‘（ ）……—《 》〈 〉·.'
ymark = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'



# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 统计文章长度
def get_content_len(data,data_key):
    result_path = cache_path + 'content_len_{0}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        regex = re.compile('.')
        feat = data['content'].apply(lambda x: len(re.findall(regex, x))).to_frame(name='content_len')
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 统计个数
def get_count(data,regular,data_key,name):
    result_path = cache_path + '{0}_count_{1}_{2}.hdf'.format(name,hashlib.md5(regular.encode()).hexdigest(),data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        regex = re.compile(regular)
        feat = data[name].apply(lambda x: len(re.findall(regex, x))).to_frame(name='{0}_count_{1}'.format(name,regular))
        # c0 = get_content_len(data,data_key)
        # feat['{0}_count_{1}_rate'.format(name,regular)] = feat['{0}_count_{1}'.format(name,regular)] / c0['content_len']
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 各个字符的个数
def get_count_for(data, data_key, name):
    result_path = cache_path + 'content_count_for_{0}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        many = pickle.load(open(r'C:\Users\csw\Desktop\python\360\data\many.pkl', 'rb+'))
        feat = data[['id']]
        for c in many:
            try:
                regex = re.compile(c)
                feat['{0}_count_{1}'.format(name, c)] = data[name].apply(lambda x: len(re.findall(regex, x)))
            except:
                print(c)
        del feat['id']
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 二次处理特征
def second_feat(result):
    return result

# 构造训练集
def make_set(data):
    t0 = time.time()
    data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        result = pd.read_hdf(result_path, 'w')
    else:
        data.index = list(range(len(data.index)))

        print('开始构造特征...')
        c0 = get_content_len(data,data_key)                                 # 正文长短
        c1 = get_count_for(data, data_key, 'content')                       # 各个字符的个数

        print('开始合并特征...')
        result = concat([data[['id','label']],c0,c1])

        result = second_feat(result)
        print('添加label')
        result['label'] = result['label'].map({'POSITIVE':1,'NEGATIVE':0})
        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result



print('读取数据...')
train = pd.read_hdf(train_hdf_path,'w')
test = pd.read_hdf(test_hdf_path,'w')

print('构造特征...')
train_feat = make_set(train)
test_feat = make_set(test)
predictors = train_feat.columns.drop(['id', 'label'])
lgb_eval = lgb.Dataset(test_feat[predictors], test_feat.label)

mean_auc = []
train_pred = pd.DataFrame({'id':train['id'].values,'pred':0})
test_pred = pd.DataFrame({'id':test['id'].values,'pred':0})
kf = KFold(len(train), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):

    print('###################第{}轮训练####################'.format(i+1))
    lgb_path = cache_path + 'lgb_model_{}.model'.format(i)
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat.label.iloc[train_index])

    print('开始训练...')
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 4,
        'num_leaves': 150,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 66,
    }
    gbm = lgb.train(params,lgb_train,10000)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    feat_imp.to_csv(cache_path + 'feat_imp_char.csv')

    print('开始预测...')
    train_pred['pred'].iloc[test_index] = gbm.predict(train_feat[predictors].iloc[test_index])
    test_pred['pred'] += gbm.predict(test_feat[predictors])
test_pred['pred'] = test_pred['pred']/5.0
data_pred = pd.concat([train_pred,test_pred])
data_pred.rename(columns={'pred':'char_pred'},inplace=True)

data_pred.to_csv(r'C:\Users\csw\Desktop\python\360\data\data_pred_char_for.csv',index=False)





