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

data_path = 'C:/Users/csw/Desktop/python/toxic/data/'
cache_path = 'F:/toxic_cache'

train = pd.read_csv(data_path + 'train.csv').fillna(' ')
new = True

cmark = '。，、；：？！“”’‘（ ）……—《 》〈 〉·.'
ymark = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
lower = 'abcdefghijklmnopqrstuvwxyz'
upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'



# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 统计个数
def get_count(data,regular,name):
    regex = re.compile(regular)
    feat = data[name].apply(lambda x: len(re.findall(regex, x))).to_frame(name='{0}_count_{1}'.format(name,regular))
    return feat


# 大写字母所占的比重
def get_upper(data):
    data_temp = data.copy()
    data_temp['upper_count'] = get_count(data_temp, '[A-Z]', 'upper_count')
    data_temp['lower_count'] = get_count(data_temp, '[a-z]', 'lower_count')
    data_temp['upper_rate'] = data_temp['upper_count'] / (data_temp['lower_count']+0.1)
    return data_temp[['upper_count','lower_count','upper_rate']]


# 首写字符
def get_head(data):
    data_temp = data.copy()
    data_temp['comment_toxic'] = data_temp['comment_toxic'].str[:1]
    comment_head = pd.get_dummies(train['comment_head'], prefix='comment_head_')
    data_temp['comment_head_count'] = data_temp['comment_head'].map(data_temp['comment_head'].value_counts())
    result = pd.concat([data_temp[['comment_head_count']],comment_head],axis=1)
    return result

# 单词平均长度
def get_word_len(data):
    def word_len(s):
        



# 结尾字符
def get_end(data,data_key,name):
    result_path = cache_path + '{0}_end_{1}.hdf'.format(name,data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        feat = data[name].apply(lambda x: -1 if len(x)==0 else int(x[-1].encode().hex(),16)).to_frame(name='{}_end'.format(name))
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 是否满足某个条件
def get_if(data,regular,data_key,name):
    result_path = cache_path + '{0}_if_{1}_{2}.hdf'.format(name,regular,data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        feat = None
        feat = feat[[regular]]
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 句子特征
def get_sencence(data, data_key):
    result_path = cache_path + 'sentence_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        def f(s):
            splits = re.split('[,。，！;？]', s)
            splits = [len(elem) for elem in splits if len(elem) > 0]
            return splits

        data_temp = data['content'].copy()
        data_temp = data_temp.apply(lambda x: f(x))
        feat = data_temp.apply(lambda x: len(x)).to_frame(name='sentence_count')
        feat['sentence_max'] = data_temp.apply(lambda x: 0 if len(x)==0 else np.max(x))
        feat['sentence_min'] = data_temp.apply(lambda x: 0 if len(x)==0 else np.min(x))
        feat['sentence_mean'] = data_temp.apply(lambda x: 0 if len(x)==0 else np.mean(x))
        feat['sentence_median'] = data_temp.apply(lambda x: 0 if len(x)==0 else np.median(x))
        feat['sentence_std'] = data_temp.apply(lambda x: -1 if len(x)==0 else np.std(x,ddof=1))
        feat['sentence_skew'] = data_temp.apply(lambda x: 0 if len(x)==0 else ss.skew(x))
        feat['sentence_skew'] = data_temp.apply(lambda x: 0 if len(x)==0 else ss.kurtosis(x))
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 模型char概率
def get_char_for(data,data_key):
    result_path = cache_path + 'char_pred_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        char_for = pd.read_csv(r'C:\Users\csw\Desktop\python\360\data\data_pred_char_for.csv')
        char_for.rename(columns={'pred':'char_pred'},inplace=True)
        feat = data.merge(char_for,on='id',how='left')
        feat = feat[['char_pred']]
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 二次处理特征
def second_feat(result):
    result['headline_mark_count'] = result['headline_count_.'] - result['headline_count_[\\u4e00-\\u9fa5]']# 标题中非汉字非英文个数
    result['headline_hanzi_rate'] = result['headline_count_[\\u4e00-\\u9fa5]']/(result['headline_count_.']+0.1)# 标题中汉字占的比例
    result['content_mark_count'] = result['content_count_.'] - result['content_count_[\\u4e00-\\u9fa5]']# 正文中非汉字非英文的个数
    result['content_hanzi_rate'] = result['content_count_[\\u4e00-\\u9fa5]'] / (result['content_count_.'] + 0.1)# 正文中汉字占的比例
    result['content_hanzi_rate'] = result['content_count_[a-zA-Z]'] / (result['content_count_.'] + 0.1)# 英文占全文的比重
    result['content_nunique_rate'] = result['content_nunique'] / (result['content_count_.'] + 0.1)/\
                                     (np.log(result['content_count_.'] + 1)*0.148+1.23)# 种类数字数比
    return result

# 构造训练集
def make_set(data):
    t0 = time.time()
    data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        data = data.reset_index(drop=True)

        print('开始构造特征...')
        feat = [data['id']]
        feat.append(get_upper(data))           # 大写字母比例
        feat.append(get_head(data))            # 开头首字符
        feat.append(get_word_len(data))            # 单词平均长度
        # 标点符号重复个数
        # 句子长度
        # 句子单词的长度
        # 换行符个数
        h0 = get_count(data,'.',data_key,'headline')                        # 标题长短
        h1 = get_count(data,',',data_key,'headline')                        # 标题标点符号个数
        h2 = get_count(data, r'[\u4e00-\u9fa5]', data_key, 'headline')       # 标题文字个数
        h3 = get_head(data, data_key, 'headline')                           # 标题起始字符
        h4 = get_end(data, data_key, 'headline')                            # 标题结尾字符
        h5 = get_count(data,'[a-zA-Z]',data_key,'headline')                 # 英文个数

        c0 = get_count(data, '.', data_key, 'content')                           # 正文长短
        c1 = get_count(data, ',', data_key, 'content')                           # 正文标点符号个数
        c2 = get_count(data, r'[\u4e00-\u9fa5]', data_key, 'content')             # 正文文字个数
        c3 = get_head(data, data_key, 'content')                                 # 正文起始字符
        c4 = get_end(data,data_key,'content')                                    # 正文结尾字符
        c14 = get_count(data, '[a-zA-Z]', data_key, 'content')           # 正文英语个数
        c15 = data['content'].apply(lambda x: 0 if x is np.nan else len(set(x))).to_frame(name='content_nunique')# 种类个数
        c16 = get_sencence(data, data_key)                               # 句子特征
        c17 = get_count(data, ' ', data_key, 'content')                  # 正文空格个数
        c18 = get_char_for(data,data_key)                                # 模型char概率



        print('开始合并特征...')
        result = concat([data[['id','label']]])

        result = second_feat(result)
        print('添加label')
        result['label'] = result['label'].map({'POSITIVE':1,'NEGATIVE':0})
        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result









