import gc
import os
import sys
import time
import random
import pickle
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import multiprocessing
import lightgbm as lgb
from scipy import stats
from pypinyin import pinyin
from functools import partial
from multiprocessing import Pool
from dateutil.parser import parse
from lightgbm import LGBMClassifier
from collections import defaultdict
from sklearn.metrics import f1_score
from datetime import date, timedelta
from contextlib import contextmanager
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from joblib import dump, load, Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold


# 求解rmse的均值和标准差
def get_ave_std(c1,c2,f1,f2):
    '''
    :param c1: 提交的常数1
    :param c2: 提交的常数2
    :param f1: 得分1
    :param f2: 得分2
    :return: 均值和标准差
    '''
    f1 = f1**2; f2 = f2**2;
    a = 2; b = 2*(c1+c2); c = c1**2+c2**2-(f1-f2);
    ave = (f1 - f2 + c2 ** 2 - c1 ** 2) / 2 / (c2 - c1)
    std = (f1 - (c1 - ave) ** 2) ** 0.5
    return ave,std

# 求解rmse的均值
def get_sub_ave_std(c1,c2,f1,f2,n1,n2):
    '''
    :param c1: 提交1的常数
    :param c2: 提交2有差异的部分的常数
    :param f1: 提交1的分数
    :param f2: 提交2的分数
    :param n1: 提交总个数
    :param n2: 提交2有差异部分的个数
    :return: 提交2有差异部分的均值
    '''
    result = ((c1+c2)-((f1**2-f2**2)*n1/n2/(c1-c2)))/2
    return result


# 抽样函数
def make_sample(n,n_sub=2,seed=None):
    import random
    if seed is not None:
        random.seed(seed)
    if type(n) is int:
        l = list(range(n))
        s = int(n / n_sub)
    else:
        l = list(n)
        s = int(len(n) / n_sub)
    random.shuffle(l)
    result = []
    for i in range(n_sub):
        if i == n_sub:
            result.append(l[i*s:])
        else:
            result.append(l[i*s: (i+1)*s])
    return result

# 统计list的value_counts
def value_counts(l):
    s = set(l)
    d = dict([(x,0) for x in s])
    for i in l:
        d[i] += 1
    result = pd.Series(d)
    result.sort_values(ascending=False,inplace=True)
    return result

# 分类特征转化率
def analyse(data,name,label='label'):
    result = data.groupby(name)[label].agg({'count':'count',
                                              'sum':'sum'})
    result['rate'] = result['sum']/result['count']
    return result

# 连续特征转化率，等距分隔
def analyse2(data,name='id',label='label', factor=10):
    grouping = pd.cut(data[name],factor)
    rate = data.groupby(grouping)[label].agg({'sum':'sum',
                                              'count':'count'})
    rate['rate'] = rate['sum']/rate['count']
    return rate

# 连续特征转化率，等数分隔
def analyse3(data,name='id',label='label', factor=10):
    grouping = pd.qcut(data[name],factor)
    rate = data.groupby(grouping)[label].agg({'sum':'sum',
                                              'count':'count'})
    rate['rate'] = rate['sum']/rate['count']
    return rate

# 分组标准化
def grp_standard(data,key,names,drop=False):
    key = key if type(key) == list else [key]
    for name in names:
        new_name = '_'.join(key + [name]) + '_normalize'
        mean_std = data.groupby(key, as_index=False)[name].agg({'mean': 'mean',
                                                               'std': 'std'})
        data = data.merge(mean_std, on=key, how='left')
        data[new_name] = ((data[name]-data['mean'])/data['std']).fillna(0).astype(np.float32)
        data[new_name] = data[new_name].replace(-np.inf, 0).fillna(0)
        data.drop(['mean','std'],axis=1,inplace=True)
    return data

# 分组归一化
def grp_normalize(data,key,names,start=0,drop=False):
    key = key if type(key) == list else [key]
    for name in names:
        new_name = '_'.join(key + [name]) + '_normalize'
        max_min = data.groupby(key,as_index=False)[name].agg({'max':'max',
                                                              'min':'min'})
        data = data.merge(max_min, on=key, how='left')
        data[new_name] = (data[name]-data['min'])/(data['max']-data['min'])
        data[new_name] = data[new_name].replace(-np.inf, start).fillna(start).astype(np.float32)
        data.drop(['max','min'],axis=1,inplace=True)
    return data

# 分组排序
def grp_rank(data,key,names,ascending=True):
    for name in names:
        data.sort_values([key, name], inplace=True, ascending=ascending)
        data['rank'] = range(data.shape[0])
        min_rank = data.groupby(key, as_index=False)['rank'].agg({'min_rank': 'min'})
        data = pd.merge(data, min_rank, on=key, how='left')
        data['rank'] = data['rank'] - data['min_rank']
        data[names] = data['rank']
    data.drop(['rank'],axis=1,inplace=True)
    return data


# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 分组排序函数
def group_rank(data, key, values, ascending=True):
    if type(key)==list:
        data_temp = data[key + [values]].copy()
        data_temp.sort_values(key + [values], inplace=True, ascending=ascending)
        data_temp['rank'] = range(data_temp.shape[0])
        min_rank = data_temp.groupby(key,as_index=False)['rank'].agg({'min_rank':'min'})
        index = data_temp.index
        data_temp = data_temp.merge(min_rank,on=key,how='left')
        data_temp.index = index
    else:
        data_temp = data[[key,values]].copy()
        data_temp.sort_values([key,values], inplace=True, ascending=ascending)
        data_temp['rank'] = range(data_temp.shape[0])
        data_temp['min_rank'] = data_temp[key].map(data_temp.groupby(key)['rank'].min())
    data_temp['rank'] = data_temp['rank'] - data_temp['min_rank']
    return data_temp['rank']


def nunique(x):
    return len(set(x))


# 前后时间差的函数：
def group_diff_time(data,key,value,n):
    data_temp = data[key+[value]].copy()
    shift_value = data_temp.groupby(key)[value].shift(n)
    data_temp['shift_value'] = data_temp[value] - shift_value
    return data_temp['shift_value']



# smape
def smape(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_diff = np.abs(y_true-y_pred)
    y_sum = y_true+y_pred
    return np.mean(y_diff/y_sum)*2


# groupby 直接拼接
def groupby(data,stat,key,value,func):
    key = key if type(key)==list else [key]
    data_temp = data[key].copy()
    feat = stat.groupby(key,as_index=False)[value].agg({'feat':func})
    data_temp = data_temp.merge(feat,on=key,how='left')
    return data_temp['feat']



# 计算关系紧密程度指数
def tfidf2(df,key1,key2):
    key = key1 + key2
    tfidf2 = '_'.join(key) + '_tfidf2'
    df1 = df.groupby(key,as_index=False)[key[0]].agg({'key_count': 'size'})
    df2 = df1.groupby(key1,as_index=False)['count'].agg({'key1_count': 'sum'})
    df3 = df1.groupby(key2, as_index=False)['count'].agg({'key2_count': 'sum'})
    df1 = df1.merge(df2,on=key1,how='left').merge(df3,on=key2,how='left')
    df1[tfidf2] = df1['key_count'] / df['key2_count'] / df['key1_count']


# 相差的日期数
def diff_of_days(day1, day2):
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return days

# 相差的分钟数
def diff_of_minutes(time1,time2):
    minutes = (parse(time1) - parse(time2)).total_seconds()//60
    return abs(minutes)

# 相差的小时数
def diff_of_hours(time1,time2):
    hours = (parse(time1) - parse(time2)).total_seconds()//3600
    return abs(hours)

# 日期的加减
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

# 日期的加减
def date_add_hours(start_date, hours):
    end_date = parse(start_date) + timedelta(hours=hours)
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    return end_date

# 获取某个类型里面第n次的值
def get_last_values(data, stat, key, value, shift, sort_value=None):
    key = key if type(key)==list else [key]
    if sort_value is not None:
        stat_temp = stat.sort_values(sort_value, ascending=(shift<0))
    else:
        stat_temp = stat.copy()
    stat_temp['value'] = stat_temp.groupby(key)[value].shift(-abs(shift))
    stat_temp.drop_duplicates(key,keep='first',inplace=True)
    data_temp = data[key].copy()
    data_temp = data_temp.merge(stat_temp,on=key,how='left')
    return data_temp['value'].values

# 获取某个类型下n次的值
def get_next_values(data, key, value, shift, sort_value=None):
    key = key if type(key)==list else [key]
    if sort_value is not None:
        data_temp = data.sort_values(sort_value, ascending=True)
    else:
        data_temp = data.copy()
    data_temp['value'] = data_temp.groupby(key)[value].shift(shift)
    return data_temp['value']

# # 获取某个类型里面第n次的值
# def get_first_values(data, stat, key, sort_value, value, shift, sort=None):
#     key = key if type(key)==list else [key]
#     if sort == 'ascending':
#         stat_temp = stat.sort_values(sort_value, ascending=True)
#     elif sort == 'descending':
#         stat_temp = stat.sort_values(sort_value, ascending=False)
#     else:
#         stat_temp = stat.copy()
#     stat_temp['value'] = stat_temp.groupby(key)[value].shift(-shift)
#     stat_temp.drop_duplicates(key,keep='first',inplace=True)
#     data_temp = data[key].copy()
#     data_temp = data_temp.merge(stat_temp,on=key,how='left')
#     return data_temp['value']

# 或取tf 特征（单个类别占群体总数的比例）
def tf(data,stat,key,value,weight=1,max_features=None):
    '''
    :param data:
    :param stat:
    :param key:
    :param value:
    :param max_features:
    :return: 返回宽表
    '''
    key = key if type(key) == list else [key]
    stat['weight'] = weight
    cate_count = stat.drop_duplicates(key + [value]).groupby(value)[value].count().sort_values(ascending=False)
    if max_features is not None:
        cate_count = cate_count[:max_features]
        stat_temp = stat[stat[value].isin(cate_count.index)][key + [value,'weight']].copy()
    else:
        stat_temp = stat[key + [value,'weight']].copy()
    stat1 = stat_temp.groupby(key + [value],as_index=False)['weight'].agg({'stat1_count':'sum'})
    stat2 = stat_temp.groupby(key,as_index=False)['weight'].agg({'stat2_count': 'sum'})
    stat3 = stat1.merge(stat2,on=key,how='left')
    c_name = '{}_tf'.format('_'.join(key + [value]))
    stat3['count'] = stat3['stat1_count'] / stat3['stat2_count']
    stat3 = stat3.set_index(key + [value])['count'].unstack()
    stat3.columns = [c_name + str(c) for c in stat3.columns]
    data_temp = data[key].copy()
    index = data_temp.index.copy()
    data_temp = data_temp.merge(stat3.reset_index(),on=key ,how='left')
    data_temp.index = index
    return data_temp[stat3.columns]

# 或取tfidf 特征（单个类别占群体总数的比例）
def tfidf(stat,key,value,max_features=None):
    '''
    :param data:
    :param stat:
    :param key:
    :param value:
    :param max_features:
    :return: 返回展开列
    '''
    key = key if type(key) == list else [key]
    cate_count = stat.drop_duplicates(key + [value]).groupby(value)[value].count().sort_values(ascending=False)
    if max_features is not None:
        cate_count = cate_count[:max_features]
        stat_temp = stat[stat[value].isin(cate_count.index)][key + [value]].copy()
    else:
        stat_temp = stat[key + [value]].copy()
    stat1 = stat_temp.groupby(key + [value],as_index=False)[value].agg({'stat1_count':'count'})
    stat2 = stat_temp.groupby(key,as_index=False)[value].agg({'stat2_count': 'count'})
    stat3 = stat_temp[key + [value]].drop_duplicates().groupby(value, as_index=False)[value].agg({'stat3_count': 'count'})
    stat1 = stat1.merge(stat2,on=key,how='left')
    tfidf_name = '{}_tfidf'.format('_'.join(key + [value]))
    tf_name = '{}_tf'.format('_'.join(key + [value]))
    idf_name = '{}_idf'.format('_'.join(key + [value]))

    stat1[tf_name] = stat1['stat1_count'] / stat1['stat2_count']
    stat3[idf_name] = np.log(len(stat_temp.groupby(key)) / (stat3['stat3_count']+1))
    stat1 = stat1.merge(stat3,on=value,how='left')
    stat1[tfidf_name] = stat1[tf_name] * stat1[idf_name]
    return stat1[key + [value,tfidf_name]]


# 压缩数据
def compress(data):
    size = sys.getsizeof(data)/2**20
    def intcp(series):
        ma = max(series)
        mi = min(series)
        if (ma<128) & (mi>=-128):
            return 'int8'
        elif (ma<32768) & (mi>=-32768):
            return 'int16'
        elif (ma<2147483648) & (mi>=-2147483648):
            return 'int32'
        else:
            return None
    def floatcp(series):
        ma = max(series)
        mi = min(series)
        if (ma<32770) & (mi>-32770):
            return 'float16'
        elif (ma<2147483600) & (mi>-2147483600):
            return 'float32'
        else:
            return None
    def astype(c,ctype):
        return data[c].astype(ctype)
    pool = multiprocessing.Pool(processes=7)
    for c in data.columns:
        ctype = None
        dtypes = data[c].dtypes
        if dtypes == np.int64:
            ctype = intcp(data[c])
        if dtypes == np.int32:
            ctype = intcp(data[c])
        if dtypes == np.int16:
            ctype = intcp(data[c])
        if dtypes == np.float64:
            ctype = floatcp(data[c])
        if dtypes == np.float32:
            ctype = floatcp(data[c])
        if ctype is None:
            continue
        try:
            data[c] = pool.apply_async(astype, (c,ctype)).get()
            print('{}   convet to {},     done!   {}'.format(dtypes,ctype,c))
        except:
            print('特征{}的类型为：{}，转化出线问题！！！'.format(c,dtypes))
    pool.close()
    pool.join()
    print('原始数据大小为： {}M'.format(round(size, 2)))
    print('新数据大小为：  {}M'.format(round(sys.getsizeof(data) / 2 ** 20,2)))
    return data



def trend(y):
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    return trend


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))



# 获取阈值
def get_threshold(preds,silent=False):
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
        print('期望得分为：{}'.format(f1*2))
    return [(1  if (pred>threshold) else 0) for pred in preds]

# 多分类F1值
def multi_f1(true,pred,silent=False):
    true_dummy = pd.get_dummies(pd.Series(true))
    pred_dummy = pd.get_dummies(pd.Series(pred))
    scores = []
    for c in true_dummy.columns:
        score = f1_score(true_dummy[c],pred_dummy[c])
        if not silent:
            print('{}       :   {}'.format(c,score))
        scores.append(score)
    return np.mean(scores)


# 多分类f1期望得分
def exp_multi_f1(pred,int_preds, weights=None,silent=True):
    int_preds_dummy = pd.get_dummies(pd.Series(int_preds))
    pred = pd.DataFrame(pred,columns=int_preds_dummy.columns)
    scores = []
    for c in pred.columns:
        n = pred[c].sum()
        m = int_preds_dummy[c].sum()
        r = pred[int_preds_dummy[c]==1][c].sum()
        f1 = 2*r / (m+n)
        if not silent:
            print('{}       :   {}'.format(c, f1))
        scores.append(f1)
    return np.average(scores,weights=weights)


# 多分类f1最佳阈值
def get_multi_f1_threshold(preds,weights=None,n_round=3):
    '''
    :param preds: 二维的概率矩阵
    :param weight: 每个种类所占的比重，默认为1（权重一样）
    :param n_round: 优化循环次数
    :return:
    '''
    if weights is None :
        weights = np.ones(preds.shape[1])
    #差分
    def derivative(arg, p):
        m, n, r = arg
        s = m + n
        return 2 * (p * s - r) / (s + 1) / s

    def get_multi_f1_threshold_di(preds, int_preds, preds_flag):
        int_preds_matrix = pd.get_dummies(int_preds).values
        para_dict = {}
        for i in range(preds.shape[1]):
            m = preds[:, i].sum()                               ##真实个数（估值）
            n = int_preds_matrix[:, i].sum()                    ##预测个数
            r = preds[int_preds_matrix[:, i] == 1, i].sum()     ##正确个数（估值）
            para_dict[i] = [m, n, r]
        for i in range(preds.shape[0]):
            if preds_flag[i]:
                continue
            else:
                temp = np.argmax([derivative(para_dict[j],preds[i,j])*weights[j] for j in range(preds.shape[1])])
                orig = int_preds[i]
                if temp != orig:
                    m, n, r = para_dict[temp]
                    para_dict[temp] = (m,n+1,r+preds[i,temp])
                    m, n, r = para_dict[orig]
                    para_dict[orig] = (m, n - 1, r - preds[i, orig])
                    int_preds[i] = temp
        return int_preds

    int_preds = pd.Series(preds.argmax(axis=1))
    int_preds.iloc[:preds.shape[1]] = list(range(preds.shape[1]))
    preds_flag = list(preds.max(axis=1)>0.9)
    for i in range(n_round):
        int_preds = get_multi_f1_threshold_di(preds,int_preds,preds_flag)
        print('期望的分：    {}'.format(exp_multi_f1(preds,int_preds,weights)))
    return int_preds.values

# count encoding
def count_encoding(li):
    temp = pd.Series(li)
    result = temp.map(temp.value_counts())
    return result

# 众位数
def mode(li):
    if stats.mode(li)[1][0]==1:
        return np.nan
    return stats.mode(li)[0][0]

# 贝叶斯平滑
def bayes_encode(C, I):
    def compute_moment(tries, success):
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i]) / tries[i])
        mean = sum(ctr_list) / len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr - mean, 2)
        return mean, var / (len(ctr_list) - 1)

    def update_from_data_by_moment(tries, success):
        mean, var = compute_moment(tries, success)
        alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        return alpha, beta

    I_temp = list(I)
    C_temp = list(C)
    alpha, beta = update_from_data_by_moment(I_temp, C_temp)
    rate = ((alpha + C) / (alpha + beta + I)).astype('float32')
    return rate


# 交叉验证统计转化率
def mul_cv_convert(data,category,label,cv=5):
    rate = np.zeros((len(data),data[label].nunique()))
    kf = KFold(len(data), n_folds=cv, shuffle=True, random_state=66)
    for i, (train_index, test_index) in enumerate(kf):
        stat1 = data.iloc[train_index]
        stat2 = data.iloc[test_index]
        temp1 = stat1.groupby([category, label], as_index=False).size().unstack().fillna(0)
        temp2 = stat1[~stat1[label].isnull()].groupby([category], as_index=False).size()
        temp3 = (temp1.T / temp2).T
        columns = [category + '_' + str(c) + '_conversion' for c in temp3.columns]
        temp3 = temp3.reset_index()
        temp4 = stat2[[category]].merge(temp3, on=category, how='left')
        rate[test_index,:] = temp4.drop(category,axis=1).values
    rate = pd.DataFrame(rate,columns=columns)
    data = concat([data,rate])
    return data



# 相同的个数
def get_n_min(li,n):
    return sorted(li)[n-1]


# 异常波动检测
def psi(x1,x2,):
    # psi = sum(（实际占比 - 预期占比）*ln(实际占比 / 预期占比))
    s1 = len(x1)
    s2 = len(x2)
    data = pd.DataFrame({'values':list(x1) + list(x2),'set':['x1']*s1 + ['x2']*s2})
    if data['values'].dtype == bool:
        data['values'] = data['values'].astype(int)
    q = 10 if data['values'].fillna('nan').nunique() > 10 else data['values'].fillna('nan').nunique()
    grouping = pd.qcut(data['values'],q,duplicates='drop')
    grouping = grouping.cat.add_categories(['nan'])
    grouping = grouping.fillna('nan', inplace=False)
    data2 = data.groupby([grouping,data['set']]).size().unstack().reset_index()
    data2['w1'] = data2['x1'] / s1
    data2['w2'] = data2['x2'] / s2
    data2['psi'] = (data2['w1'] - data2['w2']) * np.log(data2['w1']/data2['w2'])
    return data2, data2['psi'].sum()

# sigmoid反函数
def sigmoid_inverse(x):
    if (x>0) and (x<1):
        return np.log(x/(1-x))
    elif x==0 :
        return -9999
    elif x==1 :
        return 9999
    else:
        raise ValueError('value {} not in [0,1]!!!'.format(x))


# sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 概率伸缩
def probability_change(ls,p_m):
    diff = (p_m - np.mean(ls))/p_m
    while (diff > 0.001) or (diff < -0.001):
        print(diff)
        wd = sigmoid_inverse(p_m) - sigmoid_inverse(np.mean(ls))
        ls_sigmoid_inverse = ls.apply(sigmoid_inverse)
        ls_sigmoid_inverse = ls_sigmoid_inverse + wd
        ls = ls_sigmoid_inverse.apply(sigmoid)
        diff = (p_m - np.mean(ls))/p_m
    return ls

# 概率期望调整
def probability_adjust(ls,w=0.01):
    p_m = np.mean(ls)
    ls = pd.Series(ls)
    ls_sigmoid_inverse = ls.apply(sigmoid_inverse)
    median = ls_sigmoid_inverse.median()
    ls_sigmoid_inverse = (ls_sigmoid_inverse - median) * w + ls_sigmoid_inverse
    ls = ls_sigmoid_inverse.apply(sigmoid)
    ls = probability_change(ls, p_m)
    return ls


@contextmanager
def timer(title='processor'):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))









