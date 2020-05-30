import pandas as pd
import numpy as np
import lightgbm as lgb

from datetime import datetime, timedelta
from functools import partial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, PCA, TruncatedSVD
import gc
import os
import pickle
from tool.tool import *


cache_path = 'E:/jdata2/'
data_path = 'C:/Users/cui/Desktop/python/jdata2/datab/'

cache_path = 'E:/jdata2/'
data_path = 'C:/Users/cui/Desktop/python/jdata2/datab/'
inplace = False


all_data_path = data_path + 'all_data3.pkl'
if os.path.exists(all_data_path):
    user_action,user_info,user_comment,user_order,sku_info,all_action,all_user_order = pickle.load(open(all_data_path,'+rb'))
else:
    user_action = pd.read_csv(data_path + 'jdata_user_action.csv',dtype={
        'user_id': np.int32,
        'sku_id': np.int32,
        'a_date': str,
        'a_num': np.int16,
        'a_type': np.int8}).rename(columns={'a_date':'time'})
    user_info = pd.read_csv(data_path + 'jdata_user_basic_info.csv',dtype={
        'user_id': np.int32,
        'age': np.int8,
        'sex': np.int8,
        'user_lv_cd': np.int8})
    user_comment = pd.read_csv(data_path + 'jdata_user_comment_score.csv',dtype={
        'user_id': np.int32,
        'comment_create_tm': str,
        'o_id': np.int32,
        'score_level': np.int8}).rename(columns={'comment_create_tm':'time'})
    user_order = pd.read_csv(data_path + 'jdata_user_order.csv',dtype={
        'user_id': np.int32,
        'sku_id': np.int32,
        'o_id': np.int32,
        'o_area': np.int16,
        'o_sku_num':np.int16}).rename(columns={'o_date':'time'})
    sku_info = pd.read_csv(data_path + 'jdata_sku_basic_info.csv',dtype={
        'sku_id': np.int32,
        'price': np.float64,
        'cate': np.int8,
        'para_1': np.float64,
        'para_2':np.int8,
        'para_3':np.int8})
    # 对用户属性onehot
    user_age = pd.get_dummies(user_info['age']).astype(bool)
    user_age.columns = ['age{}'.format(c) for c in user_age.columns]
    user_info = pd.concat([user_info,user_age],axis=1)
    user_sex = pd.get_dummies(user_info['sex']).astype(bool)
    user_sex.columns = ['sex{}'.format(c) for c in user_sex.columns]
    user_info = pd.concat([user_info, user_sex], axis=1)
    user_lv = pd.get_dummies(user_info['user_lv_cd']).astype(bool)
    user_lv.columns = ['user_lv_cd{}'.format(c) for c in user_lv.columns]
    user_info = pd.concat([user_info, user_lv], axis=1)
    user_info.drop(['age','sex','user_lv_cd'],axis=1,inplace=True)

    date_day_map = {str(date)[:10]: diff_of_days(str(date)[:10], '2016-05-01') for date in
                    pd.date_range('2016-05-01', '2017-11-01')}
    user_action['diff_of_days'] = user_action['time'].map(date_day_map).astype(np.int16)
    user_action.sort_values('diff_of_days',inplace=True)
    user_action = user_action.merge(sku_info, on='sku_id', how='left')
    user_action = user_action[user_action['cate'].isin([30,101])]

    user_order['diff_of_days'] = user_order['time'].map(date_day_map).astype(np.int16)
    user_order.sort_values('diff_of_days',inplace=True)
    user_order = user_order.merge(sku_info,on='sku_id',how='left')
    all_user_order = user_order.copy()
    user_order = user_order[user_order['cate'].isin([30,101])]

    user_comment['diff_of_days'] = user_comment['time'].str[:10].map(date_day_map).astype(np.int16)
    user_comment.sort_values('time',inplace=True)
    user_comment = user_comment.merge(user_order[['o_id','sku_id','price', 'cate', 'para_1', 'para_2', 'para_3']],on='o_id',how='left')
    user_comment = user_comment[user_comment['cate'].isin([30,101])]


    all_action1 = user_action[['user_id','sku_id','time','a_num','a_type','cate','diff_of_days']].rename(columns={'a_num':'num','a_type':'type'})
    # all_action1['type'] = 1
    all_action2 = user_order[['user_id','sku_id','time','o_sku_num','cate','diff_of_days']].rename(columns={'o_sku_num':'num'})
    all_action2['type'] = 3
    all_action3 = user_comment[['user_id','sku_id','time','cate','diff_of_days']]
    all_action3['type'] = 4
    all_action = pd.concat([all_action1, all_action2, all_action3],axis=0)
    all_action['num'].fillna(1,inplace=True)
    all_action.sort_values(['diff_of_days','type'],inplace=True)
    pickle.dump((user_action,user_info,user_comment,user_order,sku_info,all_action,all_user_order),open(all_data_path,'+wb'))


############################################# 降维 ######################################################
user_action_ = user_action[user_action['time']<'2017-03-01']

mapping = {}
for sample in user_action_[['user_id', 'sku_id']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_action" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '1.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_action" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '2.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
del cate2_as_matrix;
gc.collect()
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_action" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '3.hdf', 'w')

mapping = {}
for sample in user_action_[['user_id', 'para_2']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_action_param2" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '4.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_action_param2" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '5.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_action_param2" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '6.hdf', 'w')

mapping = {}
for sample in user_action_[['user_id', 'para_3']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_action_param3" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '7.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_action_param3" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '8.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_action_param3" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '9.hdf', 'w')

user_order_ = user_order[user_order['time']<'2017-03-01']

mapping = {}
for sample in user_order_[['user_id', 'sku_id']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1, columns=["%s_%s_lda_order" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '10.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1, columns=["%s_%s_nmf_order" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '11.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1, columns=["%s_%s_svd_order" % ('user_sku', i) for i in range(5)]).astype(
    'float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '12.hdf', 'w')

mapping = {}
for sample in user_order_[['user_id', 'para_2']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_order_param2" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '13.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_order_param2" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '14.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_order_param2" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '15.hdf', 'w')

mapping = {}
for sample in user_order_[['user_id', 'para_3']].values:
    mapping.setdefault(sample[0], []).append(str(sample[1]))
cate1s = list(mapping.keys())
print(len(cate1s))
cate2_as_sentence = [' '.join(mapping[cate_]) for cate_ in cate1s]
cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=2).fit_transform(cate2_as_sentence)

lda = LDA(n_components=5,
          learning_method='online',
          batch_size=1000,
          n_jobs=40,
          random_state=520)
topics_of_cate1 = lda.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_lda_order_param3" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '16.hdf', 'w')

nmf = NMF(n_components=5,
          random_state=520,
          beta_loss='kullback-leibler',
          solver='mu',
          max_iter=1000,
          alpha=.1,
          l1_ratio=.5)
topics_of_cate1 = nmf.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_nmf_order_param3" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '17.hdf', 'w')

pca = TruncatedSVD(5)
topics_of_cate1 = pca.fit_transform(cate2_as_matrix)
topics_of_cate1 = pd.DataFrame(topics_of_cate1,
                               columns=["%s_%s_svd_order_param3" % ('user_sku', i) for i in range(5)]).astype('float32')
topics_of_cate1['user_id'] = cate1s
topics_of_cate1.to_hdf(data_path + '18.hdf', 'w')