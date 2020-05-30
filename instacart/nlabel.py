# -*-coding:utf-8 -*-
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.stats as scs
from sklearn.cross_validation import KFold


IDIR = '/home/user/Desktop/cuishiwen/instacart/data/'
cache_path = 'F:/cache/instacart_cache/'
cache2_path = 'F:/cache/instacart_cache2/'
pickle_path = 'F:/cache/instacart_cache/pickle/'

# pickle读数据
def load(name):
    result_path = pickle_path + '%s.pkl' % name
    try:
        result = pickle.load(open(result_path,'rb+'))
    except:
        print('地址不存在！')
    return result
# pickle写数据
def dump(var, name):
    result_path = pickle_path + '%s.pkl' % name
    try:
        pickle.dump(var,open(result_path, 'wb+'))
    except:
        print('地址不存在！')

# 对数据进行离散化
def cut(feat, n_cat=100):
    feat_temp = feat.copy()
    feat_temp = feat_temp.replace(-100,np.nan)
    feat_temp = pd.qcut(feat_temp, n_cat).apply(lambda x: x.mid)
    return feat_temp

# 对特征进行标准化
def normalize(feat):
    return feat/feat.std()

def f1(y_true,y_pred):
    TP = len(set(y_true) & set(y_pred))         #预测为a类且正确的数量
    MP = len(y_true)                            #a类实际的数量
    MN = len(y_pred)                            #预测为a类的数量
    return 2*TP/(MP+MN)

def instacart_grade(y_true,y_pred):
    return np.mean([f1(x, y) for x, y in zip(y_true['products'].values, y_pred['products'].values)])

# 第一种按照阈值获取结果
def get_result(data):
    result = data.groupby('order_id',as_index=False)['product_id'].agg({'products':lambda x:list(x)})
    return result

# 第二种按照最佳阀值获取结果
def get_result2(data):
    '''
    :param data: pd.DataFrame格式  包含['order_id','product_id','pred']
    :return: 返回 pd.DataFrame 格式结果  ['order_id','products']
    '''
    # 寻找最佳阀值
    def get_max_exp(pred_list, n_product):
        f1_temp = 0     # 期望f1
        TP = 0          # 期望正确个数
        PNone = (1.0-pred_list).prod()
        for pred in pred_list:
            n_product += 1
            TP += pred
            f1 = TP/n_product
            if f1 < f1_temp:
                if PNone > (f1_temp*1.4):
                    return 1.01
                else:
                    return pred
            else:
                f1_temp = f1

        return 0

    user_n_product = data.groupby('order_id')['pred'].sum()
    user_n_product = dict(user_n_product)
    temp = data.copy()
    temp.sort_values('pred',ascending=False,inplace=True)
    grouped = temp.groupby('order_id')
    result = {}
    for order_id, grouped in grouped:
        TRESHOLD = get_max_exp(grouped['pred'].values,user_n_product[order_id])             #输入概率备选商品的购买概率，获取最佳阀值
        result[order_id] = list(grouped['product_id'].values[grouped['pred'].values>TRESHOLD])  # 根据阀值选择商品
        result[order_id] = [None] if len(result[order_id])==0 else result[order_id]
    result = pd.Series(result).to_frame()
    result.reset_index(inplace=True)
    result.columns = ['order_id','products']
    return result

# 第三种按照最佳阀值获取结果
def get_result3(data):
    '''
    :param data: pd.DataFrame格式  包含['order_id','product_id','pred']
    :return: 返回 pd.DataFrame 格式结果  ['order_id','products']
    '''
    # 寻找最佳阀值
    def get_max_exp(pred_list, n_product):
        flag = True
        f1_temp = 0     # 期望f1
        f2_temp = 0     # 加入none后期望值
        TP = 0          # 期望正确个数
        PNone = (1.0-pred_list).prod()
        for pred in pred_list:
            n_product += 1
            TP += pred
            f1 = 1.4*TP/n_product
            f2 = 1.4*TP/(n_product+1)
            if (f1<f1_temp) and flag:
                f1_result = (pred,f1_temp)
                flag = False
            if f1 < f1_temp:
                f2_result = (pred,f2_temp)
                P1 = f1_result[1]
                P2 = f2_result[1]+PNone/(sum(pred_list)+1)
                arg = np.argmax([PNone,P1,P2])
                result = {0:(1.01,True),1:(f1_result[0],False),2:(f2_result[0],True)}
                return result[arg]
            f1_temp = f1
            f2_temp = f2
        arg = np.argmax([PNone, f1_temp,f2_temp+PNone/(sum(pred_list)+1)])
        result = {0: (1.01, True), 1: (0, False), 2: (0, True)}
        return result[arg]

    user_n_product = data.groupby('order_id')['pred'].sum()
    user_n_product = dict(user_n_product)
    temp = data.copy()
    temp.sort_values('pred',ascending=False,inplace=True)
    grouped = temp.groupby('order_id')
    result = {}
    for order_id, grouped in grouped:
        TRESHOLD = get_max_exp(grouped['pred'].values,user_n_product[order_id])             #输入概率备选商品的购买概率，获取最佳阀值
        result[order_id] = list(grouped['product_id'].values[grouped['pred'].values>TRESHOLD[0]])  # 根据阀值选择商品
        if TRESHOLD[1] is True:
            result[order_id].append(None)
    result = pd.Series(result).to_frame()
    result.reset_index(inplace=True)
    result.columns = ['order_id','products']
    return result

# 第四种按照最佳阀值获取结果
def get_result4(data):
    '''
    :param data: pd.DataFrame格式  包含['order_id','product_id','pred']
    :return: 返回 pd.DataFrame 格式结果  ['order_id','products']
    '''
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    # 寻找最佳阀值
    def get_max_exp(pred_list,PNone=None):
        if PNone is None:
            PNone = (1.0-pred_list).prod()
        expectations = get_expectations(pred_list, PNone)
        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        PNone_set = {0:True,1:False}
        if ix_max[1]==0:
            return (1.01, PNone_set[ix_max[0]])
        else:
            return (pred_list[ix_max[1]-1],PNone_set[ix_max[0]])

    temp = data.copy()
    temp.sort_values('pred',ascending=False,inplace=True)
    grouped = temp.groupby('order_id')
    result = {}
    for order_id, grouped in grouped:
        PNone = order_none[order_id]
        TRESHOLD = get_max_exp(grouped['pred'].values,PNone)             #输入概率备选商品的购买概率，获取最佳阀值
        result[order_id] = list(grouped['product_id'].values[grouped['pred'].values>=TRESHOLD[0]])  # 根据阀值选择商品
        if TRESHOLD[1] is True:
            result[order_id].append(None)
    result = pd.Series(result).to_frame()
    result.reset_index(inplace=True)
    result.columns = ['order_id','products']
    return result

# 将list转换为str
def list_to_str(arr):
    if (type(arr) != list) or (len(arr) == 0):
        return 'None'
    else:
        s = str(arr[0])
        for i in range(len(arr)-1):
            s += ' ' + str(arr[i+1])
        return s

# 基尼系数
def gini(arr):
    arr = list(arr)
    arr = sorted(arr)
    for i in reversed(range(len(arr))):
        arr[i] = sum(arr[:(i + 1)])
    gini = 1+1/len(arr)-2*sum(arr)/arr[-1]/len(arr)
    return gini

# 计算偏度
def skew(arr):
    return scs.skew(arr)

# 分组排序
def rank(data, feat_arr, feat2, ascending=True, name='rank'):
    data.sort_values(feat_arr+[feat2],inplace=True,ascending=ascending)
    data[name] = range(data.shape[0])
    min_rank = data.groupby(feat_arr,as_index=False)[name].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat_arr,how='left')
    data[name] = data[name] - data['min_rank']
    del data['min_rank']
    return data

# 读取order
def get_user_order():
    df_path = cache_path + 'user_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'orders.csv')
        df.sort_values(['user_id', 'order_number'], ascending=False, inplace=True)
        dates = [0]
        date = 0
        for i in df['days_since_prior_order'].values:
            date += i
            if np.isnan(date):
                date = 0
            dates.append(date)
        df['date'] = dates[:-1]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 构造样本集
def get_user_candicate(user_order,eval=None):
    df_path = cache_path + '%s_user_candicate.hdf' % eval
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        if eval is None:
            user_order_temp = user_order[user_order['eval_set'] != 'prior']
        else:
            user_order_temp = user_order[user_order['eval_set'] == eval]
        df = user_order_temp[['user_id','order_id']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 行为基础特征
def get_action_feat(user_order):
    df_path = cache_path + 'action.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = user_order[user_order['eval_set'] != 'prior'][[
            'order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        df.rename(columns={'days_since_prior_order':'user_last_day'},inplace=True)
         #周几，时间，距离上次天数
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df


# 读取prior
def get_prior():
    df_path = cache_path + 'prior.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'order_products__prior.csv')
        user_order = get_user_order()
        df = pd.merge(df,user_order,on='order_id',how='left')
        product = get_product()
        df = pd.merge(df,product[['product_id','aisle_id','department_id']])
        del df['eval_set']
        df.sort_values(['user_id','order_number','product_id'], ascending=True, inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 读取train
def get_train():
    df_path = cache_path + 'train.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'order_products__train.csv')
        user_order = get_user_order()
        df = pd.merge(df, user_order, on='order_id', how='left')
        df['label'] = 1
        del df['eval_set']
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 读取product
def get_product():
    df_path = cache_path + 'product.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'products.csv')
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 构造样本集
def get_candicate(prior=None,user_order=None,eval=None):
    df_path = cache_path + '%s_candicate.hdf' % eval
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior = get_prior() if prior is None else prior
        user_order = get_user_order() if user_order is None else user_order
        user_order_temp = user_order[user_order['eval_set'] != 'prior'] if eval is None else \
            user_order[user_order['eval_set'] == eval]
        df = pd.merge(user_order_temp[['user_id','order_id']],
                      prior[['user_id','product_id']], on='user_id', how='left')
        df = df.drop_duplicates(['user_id', 'product_id'])
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户活跃天数
def get_user_n_day(user_order):
    df_path = cache_path + 'user_n_day.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = user_order.groupby('user_id',as_index=False)['date'].agg({'user_n_day':'max'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买商品个数
def get_user_n_item(prior):
    df_path = cache_path + 'user_n_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('user_id',as_index=False)['product_id'].agg({'user_n_item':'count'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买商品次数
def get_user_n_order(user_order):
    df_path = cache_path + 'user_n_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = user_order.groupby('user_id', as_index=False)['order_number'].agg({'user_n_order': 'max'})
        df['user_n_order'] = df['user_n_order'] - 1
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户每次购买商品个数的中位数
def get_user_median_item(prior):
    df_path = cache_path + 'user_median_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_n_item = prior.groupby(['user_id','order_id'],as_index=False)['user_id'].agg({'order_n_item':'count'})
        df = order_n_item.groupby('user_id',as_index=False)['order_n_item'].agg({'user_median_item':'median'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买商品种类数
def get_user_n_product(prior):
    df_path = cache_path + 'user_n_product.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('user_id', as_index=False)['product_id'].agg({'user_n_product': 'nunique'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买个数的各种特征值
def get_user_order_count(prior):
    df_path = cache_path + 'user_order_count.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order_count = prior.groupby(['user_id','order_number'], as_index=False)['add_to_cart_order'].agg({'user_order_count': 'max'})
        df = user_order_count.groupby('user_id')['user_order_count'].agg({'user_order_avg_count':'mean'})
        df['user_order_max_count'] = user_order_count.groupby('user_id')['user_order_count'].agg({'user_order_max_count': 'max'})
        df['user_order_min_count'] = user_order_count.groupby('user_id')['user_order_count'].agg({'user_order_min_count': 'min'})
        df['user_order_std_count'] = user_order_count.groupby('user_id')['user_order_count'].agg({'user_order_std_count': 'std'})
        df['user_order_skew_count'] = user_order_count.groupby('user_id')['user_order_count'].agg({'user_order_skew_count': skew})
        df.reset_index(inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买个数的重心（相对）
def get_user_barycenter(prior):
    df_path = cache_path + 'user_barycenter.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order_n_item = prior.groupby(['user_id','order_number'],as_index=False)['product_id'].agg({'user_order_n_item':'count'})
        user_order_n_item['user_order_barycenter'] = user_order_n_item['order_number'] * user_order_n_item['user_order_n_item']
        df = user_order_n_item.groupby('user_id').agg({'user_order_n_item':{'user_n_item':'mean'},
                                                                'order_number':{'user_order_sum':'sum'},
                                                                'user_order_barycenter':{'user_barycenter':'sum'}})
        df.columns = df.columns.droplevel(0)
        df['user_barycenter'] = df['user_barycenter'] / df['user_n_item'] / df['user_order_sum']
        df.reset_index(inplace=True)
        df = df[['user_id','user_barycenter']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户多少天购买一次
def get_user_n_day_per_order(prior):
    df_path = cache_path + 'user_n_day_per_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('user_id').agg({'date':{'user_n_day':'max'},
                                                          'order_number':{'user_n_order':'max'}})
        df.columns = df.columns.droplevel(0)
        df['user_n_day_per_order'] = df['user_n_day']/df['user_n_order']
        df.reset_index(inplace=True)
        df = df[['user_id','user_n_day_per_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户每天购买多少个
def get_user_n_item_per_day(prior):
    df_path = cache_path + 'user_n_item_per_day.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('user_id',as_index=False)['date'].agg({'user_n_day':'max','user_n_item':'count'})
        df['user_n_item_per_day'] = df['user_n_item']/(df['user_n_day']+0.01)
        df = df[['user_id','user_n_item_per_day']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户重复购买率
def get_user_rebuy_rate(prior):
    df_path = cache_path + 'user_rebuy_rate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_rebuy_rate = get_user_product_rebuy_rate(prior)
        df = user_product_rebuy_rate.groupby('user_id',as_index=False)['user_product_rebuy_rate'].agg({'user_rebuy_rate':'mean'})
        df = df[['user_id', 'user_rebuy_rate']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买商品的平均时间间隔
def get_user_avg_day_per_item(prior,user_order):
    df_path = cache_path + 'user_avg_day_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_avg_day_per_item = get_user_product_avg_day_per_item(prior,user_order)
        df = user_product_avg_day_per_item.groupby('user_id', as_index=False)[
            'user_product_avg_day_per_item'].agg({'user_avg_day_per_item': 'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户平均多少天购买一次此商品
def get_user_product_avg_day_per_item(prior,user_order):
    df_path = cache_path + 'user_product_avg_day_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_first_day = get_user_product_first(prior,user_order)
        user_product_last_day = get_user_product_last(prior,user_order)
        user_product_n_item = get_user_product_n_item(prior)
        df = user_product_first_day.merge(user_product_last_day,on=['user_id','product_id']
                                           ).merge(user_product_n_item,on=['user_id','product_id'])
        df['user_product_avg_day_per_item'] = (df['user_product_first_day'] - df['user_product_last_day']) / \
                                              (df['user_product_n_item'] - 1).apply(lambda x: np.nan if x == 0 else x)
        df = df[['user_id','product_id','user_product_avg_day_per_item']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户第一次购买此商品的时间间隔和次数间隔
def get_user_product_first(prior,user_order):
    df_path = cache_path + 'user_product_first.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id','product_id'],keep='first')
        df.rename(columns = {'date':'user_product_first_day','order_number':'user_product_first_order'},inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df, user_n_order, on='user_id', how='left')
        df['user_product_first_order'] = df['user_n_order'] - df['user_product_first_order']
        df = df[['user_id','product_id','user_product_first_day','user_product_first_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买商品的平均次数间隔
def get_user_avg_order_per_item(prior,user_order):
    df_path = cache_path + 'user_avg_order_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_avg_order_per_item = get_user_product_avg_order_per_item(prior,user_order)
        df = user_product_avg_order_per_item.groupby('user_id',as_index=False)[
            'user_product_avg_order_per_item'].agg({'user_avg_order_per_item':'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户隔30天购买的次数占总次数的比例
def get_user_percent_30(user_order):
    df_path = cache_path + 'user_percent_30.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_n_30_order = user_order[user_order['days_since_prior_order']==30].groupby(
            'user_id',as_index=False)['user_id'].agg({'user_n_30_order':'count'})
        user_n_order = user_order.groupby(
            'user_id',as_index=False)['user_id'].agg({'user_n_order': 'count'})
        df = pd.merge(user_n_order,user_n_30_order,on='user_id',how='left').fillna(0)
        df['user_percent_30'] = df['user_n_30_order']/df['user_n_order']
        df = df[['user_id','user_percent_30']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户上一次购买个数
def get_user_n_previous_item(prior):
    df_path = cache_path + 'user_n_previous_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_previous_order = prior[['user_id','order_number']].drop_duplicates('user_id',keep='last')
        user_previous_order['order_number'] = user_previous_order['order_number']-1
        user_previous_order = user_previous_order.merge(prior,on=['user_id','order_number'],how='left')
        df = user_previous_order.groupby('user_id',as_index=False)['product_id'].agg({'user_n_previous_item':'count'})
        user_order_avg_count = get_user_order_count(prior)
        df = df.merge(user_order_avg_count,on='user_id',how='left')
        # 用户最后一次购买个数除以平均购买个数
        df['user_percent_previous_item'] = df['user_n_previous_item'] / df['user_order_avg_count']
        df = df[['user_id','user_percent_previous_item','user_n_previous_item']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 获取用户的word特征
def get_user_word2vec(prior):
    import gensim
    from sklearn.decomposition import PCA
    df_path = cache_path + 'user_word2vec.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_products_prior = prior.copy()

        order_products_prior['user_id'] = order_products_prior['user_id'].astype(str)
        user_prior = order_products_prior.groupby('product_id').apply(lambda row: row['user_id'].tolist())

        model = gensim.models.Word2Vec(user_prior.values, size=100, window=5, min_count=2, workers=4)
        model.save('user2vec.model')

        def get_vector_representation(row, pos):
            return model[row.user_id][pos] if row.user_id in model else None

        pca = PCA(n_components=2)
        word2vec_new = pca.fit_transform(model.wv.syn0)
        model.wv.syn0 = word2vec_new
        df = order_products_prior[['user_id']].drop_duplicates()
        df['user_id'] = df['user_id'].astype(str)
        df['user_vector_1'] = df.apply(lambda row: get_vector_representation(row, 0), axis=1)
        df['user_vector_2'] = df.apply(lambda row: get_vector_representation(row, 1), axis=1)
        df['user_id'] = df['user_id'].astype(int)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 根据商品重复购买率判断用户类型
def get_user_rebuy_rate_by_product(prior):
    df_path = cache_path + 'user_rebuy_rate_by_product.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior_temp = prior[['user_id','product_id']].copy()
        product_rebuy_rate = get_product_rebuy_rate(prior)
        prior_temp = pd.merge(prior_temp,product_rebuy_rate,on='product_id',how='left')
        user_rebuy_rate_by_product = prior_temp.groupby('user_id',as_index=False)[
            'product_rebuy_rate'].agg({'user_rebuy_rate_by_product':'mean'})
        user_rebuy_rate = get_user_rebuy_rate(prior)
        user_rebuy_rate_by_product = user_rebuy_rate_by_product.merge(user_rebuy_rate,on='user_id',how='left')
        user_rebuy_rate_by_product['user_rebuy_rate_by_product_rate'] = user_rebuy_rate_by_product[
            'user_rebuy_rate_by_product']/user_rebuy_rate_by_product['user_rebuy_rate']
        df = user_rebuy_rate_by_product[['user_id','user_rebuy_rate_by_product','user_rebuy_rate_by_product_rate']]
        df['user_rebuy_rate_by_product'] = normalize(df['user_rebuy_rate_by_product'])
        df['user_rebuy_rate_by_product_rate'] = normalize(df['user_rebuy_rate_by_product'])
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品重复购买率
def get_product_rebuy_rate(prior):
    df_path = cache_path + 'product_rebuy_rate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_rebuy_rate = get_user_product_rebuy_rate(prior)
        df = user_product_rebuy_rate.groupby('product_id',as_index=False)['user_product_rebuy_rate'].agg({'product_rebuy_rate':'mean'})
        df = df[['product_id', 'product_rebuy_rate']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户购买此商品的重复购买率
def get_user_product_rebuy_rate(prior):
    df_path = cache_path + 'user_product_rebuy_rate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order = get_user_order()
        user_product_n_item = get_user_product_n_item(prior)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(user_product_n_item,user_n_order,on='user_id',how='left')
        df['user_product_rebuy_rate'] = (df['user_product_n_item']-1) / (df['user_n_order']-1)
        df = df[['user_id','product_id','user_product_rebuy_rate']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户购买此商品多少次
def get_user_product_n_item(prior):
    df_path = cache_path + 'user_product_n_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id','product_id'],as_index=False)[
            'user_id'].agg({'user_product_n_item':'count'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户最后一次购买此商品的时间间隔 和次数间隔
def get_user_product_last(prior,user_order):
    df_path = cache_path + 'user_product_last.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id', 'product_id'], keep='last')
        df.rename(columns={'date': 'user_product_last_day', 'order_number': 'user_product_last_order'}, inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df,user_n_order,on='user_id',how='left')
        df['user_product_last_order'] = df['user_n_order'] - df['user_product_last_order'] + 1
        df = df[['user_id', 'product_id', 'user_product_last_day', 'user_product_last_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户平均多少次购买一次此商品
def get_user_product_avg_order_per_item(prior,user_order):
    df_path = cache_path + 'user_product_avg_order_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_first_day = get_user_product_first(prior,user_order)
        user_product_last_day = get_user_product_last(prior,user_order)
        user_product_n_item = get_user_product_n_item(prior)
        df = user_product_first_day.merge(user_product_last_day,on=['user_id','product_id']
                                           ).merge(user_product_n_item,on=['user_id','product_id'])
        df['user_product_avg_order_per_item'] = (df['user_product_first_order'] - df['user_product_last_order']) / \
                                                (df['user_product_n_item'] - 1).apply(lambda x: np.nan if x == 0 else x)
        df = df[['user_id','product_id','user_product_avg_order_per_item']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df


# 添加标签
def get_label(df,prior,train):
    y_candicate = prior[['user_id','product_id']].drop_duplicates()
    y_true = train[['user_id','product_id']].drop_duplicates()
    y_label = pd.merge(y_candicate,y_true,on=['user_id','product_id'],how='inner')
    y_label = y_label.groupby('user_id',as_index=False)['product_id'].agg({'nlabel':'count'})
    df = df.merge(y_label,on='user_id',how='left')
    df['nlabel'] = df['nlabel'].fillna(0)
    df['label'] = df['nlabel'].apply(lambda x:1 if x==0 else 0)
    return df

def get_order_pred(df_pred_path):
    df_pred = load(df_pred_path)
    order_pred = df_pred.groupby('order_id', as_index=False)['pred'].agg({'sum_label': 'sum',
                                                                          'none_label': lambda x: (1 - x).prod(),
                                                                          'max_label': 'max',
                                                                          'min_label': 'min',
                                                                          'mean_label': 'mean',
                                                                          'std_label': 'std',
                                                                          'skew_label': skew})
    return order_pred

# 添加二次特征
def get_second_feat(df):

    # 距离最后一次购买时间/用户多少天购买一次
    df['user_exp_order'] = df['user_last_day'] / df['user_n_day_per_order']
    # 距离最后一次购买时间*用户平均每天购买多少个
    df['user_exp_item'] = df['user_last_day'] * df['user_n_item_per_day']


    return df

#构建用户训练集和测试集
def make_user_train_set(eval):
    df_path = cache_path + '%s_user_set.hdf' % eval
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior = get_prior()
        train = get_train()
        user_order = get_user_order()

        df = get_user_candicate(user_order, eval)                      # 构造样本
        action_feat = get_action_feat(user_order)                       # 构造行为基础特征

        user_n_day = get_user_n_day(user_order)                         # 用户活跃天数
        user_n_item = get_user_n_item(prior)                            # 用户购买商品个数
        user_n_order = get_user_n_order(user_order)                     # 用户购买商品次数
        user_median_item = get_user_median_item(prior)                  # 用户每次购买商品个数的中位数
        user_n_product = get_user_n_product(prior)                      # 用户购买商品种类数
        uesr_order_count = get_user_order_count(prior)                  # 用户购买个数的各种特征值
        user_barycenter = get_user_barycenter(prior)                    # 用户购买个数的重心（相对）
        user_n_day_per_order = get_user_n_day_per_order(prior)          # 用户多少天购买一次
        user_n_item_per_day = get_user_n_item_per_day(prior)            # 用户平均每天购买多少个
        user_rebuy_rate = get_user_rebuy_rate(prior)                    # 用户重复购买率
        user_avg_day_per_item = get_user_avg_day_per_item(prior, user_order)  # 用户购买商品的平均时间间隔
        user_avg_order_per_item = get_user_avg_order_per_item(prior, user_order)  # 用户购买商品的平均次数间隔
        user_percent_30 = get_user_percent_30(user_order)               # 用户隔30天购买的次数占总次数的比例
        user_n_previous_item = get_user_n_previous_item(prior)          # 用户上一次购买个数
        user_word2vec = get_user_word2vec(prior)                        # 添加用户的work2vec
        user_rebuy_rate_by_product = get_user_rebuy_rate_by_product(prior)  # 根据商品重复购买率判断用户类型
        #order_pred = get_order_pred('df_pred')                          # UP模型产生的用户特征
        order_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\instacart\submission\order_pred.csv')


        print('将特征组合到一起')
        df = pd.merge(df, action_feat,                      on='order_id', how='left')

        df = pd.merge(df, user_n_day,                       on='user_id', how='left')
        df = pd.merge(df, user_n_item,                      on='user_id', how='left')
        df = pd.merge(df, user_n_order,                     on='user_id', how='left')
        df = pd.merge(df, user_median_item,                 on='user_id', how='left')
        df = pd.merge(df, user_n_product,                   on='user_id', how='left')
        df = pd.merge(df, uesr_order_count,                 on='user_id', how='left')
        df = pd.merge(df, user_barycenter,                  on='user_id', how='left')
        df = pd.merge(df, user_n_day_per_order,             on='user_id', how='left')
        df = pd.merge(df, user_n_item_per_day,              on='user_id', how='left')
        df = pd.merge(df, user_rebuy_rate,                  on='user_id', how='left')
        df = pd.merge(df, user_avg_day_per_item,            on='user_id', how='left')
        df = pd.merge(df, user_avg_order_per_item,          on='user_id', how='left')
        df = pd.merge(df, user_percent_30,                  on='user_id', how='left')
        df = pd.merge(df, user_n_previous_item,             on='user_id', how='left')
        df = pd.merge(df, user_word2vec,                    on='user_id', how='left')
        df = pd.merge(df, user_rebuy_rate_by_product,       on='user_id', how='left')
        df = pd.merge(df, order_pred,                        on='order_id', how='left')


        df = get_second_feat(df)                            # 添加二次特征
        print('添加label')
        df = get_label(df,prior,train)
        df = df.fillna(-100)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

user_order = get_user_order()
user_train = make_user_train_set('train')
user_test = make_user_train_set('test')

# 线下调参
'''
predictors = user_train.columns.drop(['user_id', 'order_id', 'nlabel', 'label'])
'''
predictors = [ 'order_dow', 'order_hour_of_day',
       'user_last_day', 'user_n_day', 'user_n_item', 'user_n_order',
       'user_median_item', 'user_n_product', 'user_order_avg_count',
       'user_order_max_count', 'user_order_min_count', 'user_order_std_count',
       'user_order_skew_count', 'user_barycenter', 'user_n_day_per_order',
       'user_n_item_per_day', 'user_rebuy_rate', 'user_avg_day_per_item',
       'user_avg_order_per_item', 'user_percent_30',
        'user_rebuy_rate_by_product', 'user_rebuy_rate_by_product_rate',
       'user_percent_previous_item', 'user_n_previous_item', 'user_exp_order',
       'user_exp_item','user_vector_1', 'user_vector_2', 'sum_label',
       'none_label', 'max_label', 'min_label',
       'mean_label', 'std_label', 'skew_label']


# xgb做回归
import xgboost as xgb
eval_train = user_train[:int(user_train.shape[0]*0.7)]
eval_test = user_train[int(user_train.shape[0]*0.7):]
xgb_train = xgb.DMatrix(eval_train[predictors], eval_train['nlabel'])
xgb_eval = xgb.DMatrix(eval_test[predictors], eval_test['nlabel'])

# xgtest = xgb.DMatrix(test[feature_label])
xgb_params = {
        'objective': 'reg:linear',
        'eta': 0.01,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 4,
        'subsample': 0.886,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'rmse',
        'seed': 201703,
        'missing': -1
    }
watchlist = [(xgb_train,'train'), (xgb_eval, 'val')]
model = xgb.train(xgb_params,
                  xgb_train,
                  10000,
                  evals = watchlist,
                  verbose_eval = 50,
                  early_stopping_rounds = 50)
'''
Stopping. Best iteration:
[2614]  train-rmse:3.09517      val-rmse:3.2534
'''


# CV回归
import time
import xgboost as xgb
from sklearn.cross_validation import KFold
def right_test(train,predictors,target,test=None):

    t0 = time.time()

    train_X = train[predictors]
    train_y = train[target]
    pred_train = train[['user_id','order_id']]
    pred_train['pred_nlabel'] = 0

    test_X = test[predictors]
    pred_test = test[['user_id','order_id']]
    pred_test['pred_nlabel'] = 0

    params = {
        'objective': 'reg:linear',
        'eta': 0.01,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 4,
        'subsample': 0.886,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'rmse',
        'seed': 201703,
        'missing': -1
    }


    kf = KFold(len(train_y), n_folds = 5, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):

        sub_train_X = train_X.iloc[train_index]
        sub_test_X = train_X.iloc[test_index]
        sub_train_y = train_y.iloc[train_index]
        sub_test_y = train_y.iloc[test_index]

        ## build xgb
        xgbtrain = xgb.DMatrix(sub_train_X, sub_train_y)
        xgbtest = xgb.DMatrix(sub_test_X, sub_test_y)
        watchlist = [(xgbtrain, 'train'), (xgbtest, 'val')]
        model = xgb.train(xgb_params,
                          xgbtrain,
                          10000,
                          evals=watchlist,
                          verbose_eval=50,
                          early_stopping_rounds=50)
        pred_train['pred_nlabel'].iloc[test_index] = model.predict(xgbtest)
        pred_test['pred_nlabel'] += model.predict(xgb.DMatrix(test_X))


    print ('Done in %.1fs!' % (time.time()-t0))
    pred_test['pred_nlabel'] = pred_test['pred_nlabel']/5
    return (pred_train,pred_test)

user_train_pred_nlabel,user_test_pred_nlabel = right_test(user_train,predictors,'nlabel',user_test)
user_pred_nlabel = pd.concat([user_train_pred_nlabel,user_test_pred_nlabel])
user_pred_nlabel.to_csv(r'C:\Users\csw\Desktop\user_nlabel.csv',index=False)

