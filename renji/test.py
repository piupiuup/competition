import jieba
import re
import sys
from collections import Counter
import pandas as pd



test_path = r'C:\Users\csw\Desktop\python\360\data\evaluation_public.tsv'
train_path = r'C:\Users\csw\Desktop\python\360\data\train.tsv'
cache_path = 'F:/360_cache/'
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

train_hdf_path = r'C:\Users\csw\Desktop\python\360\data\train.hdf'
test_hdf_path = r'C:\Users\csw\Desktop\python\360\data\evaluation_public.hdf'
cache_path = 'F:/360_cache/'
new = True

cmark = '。，、；：？！“”’‘（ ）……—《 》〈 〉·.'
ymark = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


# 获取阈值
def get_threshold(preds,silent=0):
    preds_temp = sorted(preds,reverse=True)
    n = sum(preds) # 实际正例个数
    m = 0   # 提交的正例个数
    e = 0   # 正确个数的期望值
    f1 = 0  # f1的期望得分
    for threshold in preds_temp:
        e += threshold
        m += 1
        f1_temp = e/(m+n)
        if f1>f1_temp:
            break
        else:
            f1 = f1_temp
    if not silent:
        print('阈值为：{}'.format(threshold))
        print('提交正例个数为：{}'.format(m-1))
        print('期望召回率为：{}'.format(e/n))
        print('期望正确率为：{}'.format(e/m))
        print('期望得分为：{}'.format(f1*2))
    return [(1  if (pred>threshold) else 0) for pred in preds]

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

# 常见字符
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

# 开头字符
def get_head(data,data_key,name):
    result_path = cache_path + '{0}_header_{1}.hdf'.format(name,data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        feat = data.apply(lambda x: -1 if len(x)==0 else int(x[0].encode().hex(),16)).to_frame(name='{}_head'.format(name))
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 结尾字符
def get_end(data,data_key,name):
    result_path = cache_path + '{0}_end_{1}.hdf'.format(name,data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        feat = data.apply(lambda x: -1 if len(x)==0 else int(x[-1].encode().hex(),16)).to_frame(name='{}_end'.format(name))
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

# 句子重复的个数
def get_repeat_sentence(data,data_key,i):
    result_path = cache_path + 'repeat_sentence_{0}_{1}.hdf'.format(data_key,i)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        def f(s):
            splits = re.split('[,。，！;、？]', s)
            splits = [elem for elem in splits if len(elem) > i]
            return splits
        def f1(splits):
            t = Counter(splits).most_common(1)
            if len(t) >= 1:
                return t[0][1]
            return 0
        def f2(splits):
            nunique_sentence = len(Counter(splits))
            return nunique_sentence
        def f3(splits):
            repeat_sentence_count = pd.Series(Counter(splits))
            repeat_sentence_count = len(repeat_sentence_count[repeat_sentence_count>1])
            return repeat_sentence_count
        def f4(splits):
            sentence_count = len(splits)
            return sentence_count
        data_temp = data.copy()
        data_temp = data_temp.apply(lambda x: f(x))
        feat = data_temp.apply(lambda x: f1(x)).to_frame(name='repeat_sentence_{}'.format(i))
        feat['nunique_sentence_{}'.format(i)] = data_temp.apply(lambda x: f2(x))
        feat['repeat_sentence_count_{}'.format(i)] = data_temp.apply(lambda x: f3(x))
        feat['sentence_count_{}'.format(i)] = data_temp.apply(lambda x: f4(x))
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

        data_temp = data.copy()
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

# 二次处理特征
def second_feat(result):
    # result['headline_mark_count'] = result['headline_count_.'] - result['headline_count_[\\u4e00-\\u9fa5]']# 标题中非汉字非英文个数
    # result['headline_hanzi_rate'] = result['headline_count_[\\u4e00-\\u9fa5]']/(result['headline_count_.']+0.1)# 标题中汉字占的比例
    # result['content_mark_count'] = result['content_count_.'] - result['content_count_[\\u4e00-\\u9fa5]']# 正文中非汉字非英文的个数
    # result['content_hanzi_rate'] = result['content_count_[\\u4e00-\\u9fa5]'] / (result['content_count_.'] + 0.1)# 正文中汉字占的比例
    # result['content_hanzi_rate'] = result['content_count_[a-zA-Z]'] / (result['content_count_.'] + 0.1)# 英文占全文的比重
    # result['content_nunique_rate'] = result['content_nunique'] / (result['content_count_.'] + 0.1)/\
    #                                  (np.log(result['content_count_.'] + 1)*0.148+1.23)# 种类数字数比
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
        data.index = list(range(len(data.index)))

        print('开始构造特征...')
        h0 = get_count(data,'.',data_key,'headline')                        # 标题长短
        # h1 = get_count(data['headline'],',',data_key,'headline')                        # 标题标点符号个数
        # h2 = get_count(data['headline'], r'[\u4e00-\u9fa5]', data_key, 'headline')       # 标题文字个数
        # h3 = get_head(data['headline'], data_key, 'headline')                           # 标题起始字符
        # h4 = get_end(data['headline'], data_key, 'headline')                            # 标题结尾字符
        # h5 = get_count(data['headline'],'[a-zA-Z]',data_key,'headline')                 # 英文个数

        # c0 = get_content_len(data,data_key)  # 正文长短
        c1 = get_count_for(data, data_key, 'content')  # 正文长短
        # c1 = get_count(data, '，', data_key, 'content')                           # 正文长短
        # c2 = get_count(data, '的', data_key, 'content')  # 正文长短
        # c3 = get_count(data, '。', data_key, 'content')  # 正文长短
        # c4 = get_count(data, '、', data_key, 'content')  # 正文长短
        # c5 = get_count(data, '一', data_key, 'content')  # 正文长短
        # c6 = get_count(data, '是', data_key, 'content')  # 正文长短
        # c7 = get_count(data, '" ', data_key, 'content')  # 正文长短
        # c8 = get_count(data, '[0-9]', data_key, 'content')  # 正文长短
        # c1 = get_count(data['content'], ',', data_key, 'content')                           # 正文标点符号个数
        # c2 = get_count(data['content'], r'[\u4e00-\u9fa5]', data_key, 'content')             # 正文文字个数
        # c3 = get_head(data['content'], data_key, 'content')                                 # 正文起始字符
        # c4 = get_end(data['content'],data_key,'content')                                    # 正文结尾字符
        # c5 = get_repeat_sentence(data['content'], data_key, 0)                              # 正文句子重复个数
        # c6 = get_repeat_sentence(data['content'], data_key, 3)                      # 正文句子重复个数
        # c7 = get_repeat_sentence(data['content'], data_key, 6)                      # 正文句子重复个数
        # c8 = get_repeat_sentence(data['content'], data_key, 9)                      # 正文句子重复个数
        # c9 = get_repeat_sentence(data['content'], data_key, 12)                     # 正文句子重复个数
        # c10 = get_repeat_sentence(data['content'], data_key, 15)                    # 正文句子重复个数
        # c11 = get_repeat_sentence(data['content'], data_key, -1)                    # 正文句子重复个数
        # c12 = get_count(data['content'], '[a-zA-Z]', data_key, 'content')           # 正文英语个数
        # c13 = data['content'].apply(lambda x: 0 if x is np.nan else len(set(x))).to_frame(name='content_nunique')# 种类个数
        # c14 = get_sencence(data['content'], data_key)                               # 句子特征
        # c15 = get_count(data['content'], ' ', data_key, 'content')                  # 正文空格个数
        # c16 = get_count(data['content'], '1、', data_key, 'content')                # 正文1、个数
        # c17 = get_count(data['content'], '2、', data_key, 'content')                 # 正文2、个数
        # c18 = get_count(data['content'], '3、', data_key, 'content')                 # 正文3、个数
        # c19 = get_repeat_sentence(data['content'], data_key, 20)  # 正文句子重复个数


        print('开始合并特征...')
        result = concat([data[['id','label']],h0,c1])

        result = second_feat(result)
        print('添加label')
        result['label'] = result['label'].map({'POSITIVE':1,'NEGATIVE':0})
        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result



print('读取train数据...')
train = pd.read_hdf(train_hdf_path,'w')
print('切分数据...')
select_index = list(train.index[::4])
test = train.iloc[select_index, :]
train = train.iloc[~train.index.isin(select_index), :]

print('构造特征...')
train_feat = make_set(train)
test_feat = make_set(test)
predictors = train_feat.columns.drop(['id', 'label'])

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
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
print('线下auc得分为： {}'.format(roc_auc_score(test_feat.label,preds)))
preds_scatter = get_threshold(preds,silent=0)
print('线下F1得分为： {}'.format(f1_score(test_feat.label,preds_scatter)))





