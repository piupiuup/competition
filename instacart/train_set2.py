# -*-coding:utf-8 -*-
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.stats as scs
from sklearn.cross_validation import KFold


IDIR = 'C:/Users/csw/Desktop/python/instacart/data/'
cache_path = 'F:/cache/instacart_cache2/'
pickle_path = 'F:/cache/instacart_cache/pickle/'

# pickle读数据
def load(name):
    result_path = r'F:\cache\instacart_cache\pickle\%s.pkl' % name
    try:
        result = pickle.load(open(result_path,'rb+'))
    except:
        print('地址不存在！')
    return result
# pickle写数据
def dump(var, name):
    result_path = r'F:\cache\instacart_cache\pickle\%s.pkl' % name
    try:
        pickle.dump(var,open(result_path, 'wb+'))
    except:
        print('地址不存在！')

# 对特征进行标准化
def normalize(feat):
    return feat/feat.std()

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
    df_path = r'F:\cache\instacart_cache2\user_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'orders.csv')
        df.sort_values(['user_id', 'order_number'], ascending=False, inplace=True)
        big_30 = df.groupby('user_id').head(1)
        big_30 = big_30[big_30['days_since_prior_order']==30]
        df = df[~df['user_id'].isin(big_30['user_id'].values)]
        user_n_order = df.groupby('user_id',as_index=False)['order_number'].agg({'user_n_order':'max'})
        user_n_order = user_n_order[user_n_order['user_n_order'] > 4]
        df = df[df['user_id'].isin(user_n_order['user_id'].values)]
        last_order = df.groupby('user_id').head(1)
        df = df[~df['order_id'].isin(last_order['order_id'].values)]
        last_order = df.groupby('user_id').head(1)
        df.loc[df['order_id'].isin(last_order['order_id'].values), ['eval_set']] = 'train'
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

# 读取prior
def get_prior():
    df_path = r'F:\cache\instacart_cache2\prior.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'order_products__prior.csv')
        user_order = get_user_order()
        df = df[df['order_id'].isin(user_order[user_order['eval_set']=='prior']['order_id'].values)]
        df = pd.merge(df,user_order,on='order_id',how='left')
        product = get_product()
        df = pd.merge(df,product[['product_id','aisle_id','department_id']])
        del df['eval_set']
        df.sort_values(['user_id','order_number','product_id'], ascending=True, inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 读取train
def get_train():
    df_path = r'F:\cache\instacart_cache2\train.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'order_products__prior.csv')
        user_order = get_user_order()
        df = df[df['order_id'].isin(user_order[user_order['eval_set'] == 'train']['order_id'].values)]
        df = pd.merge(df, user_order, on='order_id', how='left')
        df['label'] = 1
        del df['eval_set']
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 读取product
def get_product():
    df_path = r'F:\cache\instacart_cache2\product.hdf'
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
        df['user_percent_30'] = df['user_n_30_order']/(df['user_n_order']-1)
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
        order_products_prior = prior

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
        prior_temp = prior_temp.merge(product_rebuy_rate,on='product_id',how='left')
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


# 商品被购买过的次数
def get_product_n_item(prior):
    df_path = cache_path + 'product_n_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('product_id',as_index=False)['user_id'].agg({'product_n_item':'count'})
        df['product_n_item'] = normalize(df['product_n_item'])
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品被购买过的人数
def get_product_n_people(prior):
    df_path = cache_path + 'product_n_people.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('product_id', as_index=False)['user_id'].agg({'product_n_people': 'nunique'})
        df['product_n_people'] = normalize(df['product_n_people'])
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
        df['product_rebuy_rate'] = normalize(df['product_rebuy_rate'])
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品被购买的平均时间间隔
def get_product_avg_day_per_item(prior):
    df_path = cache_path + 'product_avg_day_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order = get_user_order()
        user_product_avg_day_per_item = get_user_product_avg_day_per_item(prior,user_order)
        df = user_product_avg_day_per_item.groupby('product_id', as_index=False)[
            'user_product_avg_day_per_item'].agg({'product_avg_day_per_item': 'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品被购买的平均次数间隔
def get_product_avg_order_per_item(prior,user_order):
    df_path = cache_path + 'product_avg_order_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_avg_order_per_item = get_user_product_avg_order_per_item(prior,user_order)
        df = user_product_avg_order_per_item.groupby('product_id',as_index=False)[
            'user_product_avg_order_per_item'].agg({'product_avg_order_per_item':'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品最后一次被购买占的比例
def get_product_last_percent(prior):
    df_path = cache_path + 'product_last_precent.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.groupby('user_id', as_index=False)['order_number'].max()
        temp = pd.merge(temp, prior, on=['user_id', 'order_number'], how='left')
        df = prior.groupby('product_id')['product_id'].agg({'product_n_item': 'count'})
        df['product_n_people'] = prior.groupby('product_id')['user_id'].nunique()
        df['product_last_n_item'] = temp.groupby('product_id')['product_id'].count()
        df = df.reset_index().fillna(0)
        df['product_last_precent_item1'] = df['product_last_n_item'] / df['product_n_item']
        df['product_last_precent_people1'] = df['product_last_n_item'] / df['product_n_people']
        df = df[['product_id','product_last_precent_item1','product_last_precent_people1']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品最后两次被购买占的比例
def get_product_last_percent2(prior):
    df_path = cache_path + 'product_last_percent2.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.groupby('user_id', as_index=False)['order_number'].max()
        temp1 = pd.merge(temp, prior, on=['user_id', 'order_number'], how='left')
        temp['order_number'] = temp['order_number'] - 1
        temp2 = pd.merge(temp, prior, on=['user_id', 'order_number'], how='left')
        temp = pd.concat([temp1,temp2])
        df = prior.groupby('product_id')['product_id'].agg({'product_n_item': 'count'})
        df['product_last_n_item'] = temp.groupby('product_id')['product_id'].count()
        df = df.reset_index().fillna(0)
        df['product_last_percent_item2'] = df['product_last_n_item'] / df['product_n_item']
        df = df[['product_id', 'product_last_percent_item2']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品的order
def get_product_order(prior):
    df_path = cache_path + 'product_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order = get_user_order()
        user_n_order = get_user_n_order(user_order)
        order_number = prior.merge(user_n_order,on='user_id',how='left')
        order_number['order_number'] = (order_number['order_number']-1)/(order_number['user_n_order']-1+0.01)
        df = order_number.groupby('product_id')['order_number'].agg({'product_order_avg': 'mean'})
        df['product_order_std'] = order_number.groupby('product_id')['order_number'].std()
        df['product_order_skew'] = order_number.groupby('product_id')['order_number'].skew()
        df['product_order_avg'] = normalize(df['product_order_avg'])
        df['product_order_std'] = normalize(df['product_order_std'])
        df['product_order_skew'] = normalize(df['product_order_skew'])
        df.reset_index(inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品的平均add_to_cart_order（相对值）
def get_product_cart_order(prior):
    df_path = cache_path + 'product_cart_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_n_item = prior.groupby('order_id',as_index=False)[
            'add_to_cart_order'].agg({'order_n_item':'max'})
        add_to_cart_order = prior.merge(order_n_item, on='order_id', how='left')
        add_to_cart_order['add_to_cart_order'] = (add_to_cart_order['add_to_cart_order'] - 1) / (add_to_cart_order['order_n_item'] - 1 + 0.01)
        df = add_to_cart_order.groupby(['product_id'])[
            'add_to_cart_order'].agg({'product_avg_cate_order': 'mean'})
        df['product_std_cart_order'] = add_to_cart_order.groupby(['product_id'])[
            'add_to_cart_order'].std()
        df.reset_index(inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 获取商品的word特征
def get_product_word2vec(prior):
    import gensim
    from sklearn.decomposition import PCA
    df_path = cache_path + 'product_word2vec.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_products_prior = prior

        order_products_prior['product_id'] = order_products_prior['product_id'].astype(str)
        products_prior = order_products_prior.groupby('user_id').apply(lambda row: row['product_id'].tolist())
        model = gensim.models.Word2Vec(products_prior.values, size=100, window=5, min_count=2, workers=4)
        def get_vector_representation(row, pos):
            return model[row.product_id][pos] if row.product_id in model else None

        pca = PCA(n_components=2)
        word2vec_new = pca.fit_transform(model.wv.syn0)
        model.wv.syn0 = word2vec_new

        df = order_products_prior[['product_id']].drop_duplicates()
        df['product_vector_1'] = df.apply(lambda row: get_vector_representation(row, 0), axis=1)
        df['product_vector_2'] = df.apply(lambda row: get_vector_representation(row, 1), axis=1)
        df['product_vector_1'] = normalize(df['product_vector_1'])
        df['product_vector_2'] = normalize(df['product_vector_2'])
        df['product_id'] = df['product_id'].astype(int)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

def get_product_word2vec2(prior):
    import gensim
    from sklearn.decomposition import PCA
    df_path = cache_path + 'product_word2vec2.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_products_prior = prior

        order_products_prior['product_id'] = order_products_prior['product_id'].astype(str)
        products_prior = order_products_prior.groupby('order_id').apply(lambda row: row['product_id'].tolist())
        model = gensim.models.Word2Vec(products_prior.values, size=100, window=5, min_count=2, workers=4)
        def get_vector_representation(row, pos):
            return model[row.product_id][pos] if row.product_id in model else None

        pca = PCA(n_components=2)
        word2vec_new = pca.fit_transform(model.wv.syn0)
        model.wv.syn0 = word2vec_new

        df = order_products_prior[['product_id']].drop_duplicates()
        df['product_vector_11'] = df.apply(lambda row: get_vector_representation(row, 0), axis=1)
        df['product_vector_21'] = df.apply(lambda row: get_vector_representation(row, 1), axis=1)
        df['product_vector_11'] = normalize(df['product_vector_11'])
        df['product_vector_21'] = normalize(df['product_vector_21'])
        df['product_id'] = df['product_id'].astype(int)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 通过商品购买的用户计算商品购买率
def get_product_rebuy_rate_buy_user(prior):
    df_path = cache_path + 'product_rebuy_rate_buy_user.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior_temp = prior[['user_id','product_id']].copy()
        user_rebuy_rate = get_user_rebuy_rate(prior)
        prior_temp = prior_temp.merge(user_rebuy_rate,on='user_id',how='left')
        product_rebuy_rate_buy_user = prior_temp.groupby('product_id',as_index=False)[
            'user_rebuy_rate'].agg({'product_rebuy_rate_buy_user':'mean'})
        product_rebuy_rate = get_product_rebuy_rate(prior)
        product_rebuy_rate_buy_user = product_rebuy_rate_buy_user.merge(product_rebuy_rate,on='product_id',how='left')
        product_rebuy_rate_buy_user['product_rebuy_rate_buy_user_rate'] = product_rebuy_rate_buy_user[
            'product_rebuy_rate_buy_user']/product_rebuy_rate_buy_user['product_rebuy_rate']
        df = product_rebuy_rate_buy_user[['product_id','product_rebuy_rate_buy_user','product_rebuy_rate_buy_user_rate']]
    return df

# aisle基础特征
def get_aisle_feat(prior):
    df_path = cache_path + 'aisle_feat.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('aisle_id')['aisle_id'].agg({'aisle_item_count':'count'})        # 这个aisle被所有人购买过多少次
        df['aisle_n_user'] = prior.groupby('aisle_id')['user_id'].nunique()                 # 这个aisle被多少分购买过
        df['aisle_avg_item_per_user'] = df['aisle_item_count'] / df['aisle_n_user']         # 平均每人购买多少次
        temp = prior.groupby(['aisle_id', 'user_id'], as_index=False)['aisle_id'].agg({'aisle_user_n_item':'count'})
        df['aisle_std_pre_user'] = temp.groupby('aisle_id')['aisle_user_n_item'].std()      # 每个人购买次数的方差
        df['aisle_skew_pre_user'] = temp.groupby('aisle_id')['aisle_user_n_item'].agg({'aisle_skew_pre_user':skew})# 每个人购买次数的偏度指数
        df.reset_index(inplace=True)
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

# 商品重复购买比例
def get_order_train_rebuy_rate(train,prior,user_order,eval):
    df_path = cache_path + 'order_train_rebuy_rate_%s.hdf' % eval
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = get_candicate(prior, user_order)
        product = get_product()
        df_temp = df.merge(product,on='product_id')
        df_temp = pd.merge(df_temp,train[['order_id','product_id','label']],
                              on=['order_id','product_id'],how='left').fillna(0)
        user_list_train = train['user_id'].unique().tolist()
        n = int(len(user_list_train)/3)
        result = None
        for user_list_sub in [user_list_train[:n],user_list_train[n:2*n],user_list_train[2*n:]]:
            user_list_other = list(set(user_list_train)-set(user_list_sub))
            train_temp = df_temp[df_temp['user_id'].isin(user_list_other)]
            order_train_product_rebuy_rate = train_temp.groupby('product_id',as_index=False)[
                'label'].agg({'order_train_product_rebuy':'sum',
                              'order_train_product_buy':'count'})
            order_train_aisle_rebuy_rate = train_temp.groupby('aisle_id', as_index=False)[
                'label'].agg({'order_train_aisle_rebuy': 'sum',
                              'order_train_aisle_buy': 'count'})
            order_train_department_rebuy_rate = train_temp.groupby('department_id', as_index=False)[
                'label'].agg({'order_train_department_rebuy': 'sum',
                              'order_train_department_buy':'count'})
            df = df_temp[df_temp['user_id'].isin(user_list_sub)].merge(order_train_product_rebuy_rate,on='product_id',how='left').merge(
                order_train_aisle_rebuy_rate,on='aisle_id',how='left').merge(
                order_train_department_rebuy_rate,on='department_id',how='left')
            df['order_train_product_rebuy_rate'] = (df['order_train_product_rebuy'])/\
                                                   (df['order_train_product_buy'].apply(lambda x:x if x>100 else np.nan))
            df['order_train_aisle_rebuy_rate'] = (df['order_train_aisle_rebuy']) / \
                                                   (df['order_train_aisle_buy'])
            df['order_train_department_rebuy_rate'] = (df['order_train_department_rebuy']) / \
                                                   (df['order_train_department_buy'])
            if result is None:
                result = df
            else:
                result = pd.concat([result,df])
        if eval == 'test':
            test_user_id = user_order[user_order['eval_set']=='test']['user_id'].unique().tolist()
            product_rebuy_rate = result[['user_id', 'product_id', 'order_train_product_rebuy_rate', 'order_train_aisle_rebuy_rate',
                         'order_train_department_rebuy_rate']].groupby('product_id',as_index=False).mean()
            result = df_temp[df_temp['user_id'].isin(test_user_id)].merge(product_rebuy_rate,on='product_id',how='left')
        df = result[['order_id', 'product_id', 'order_train_product_rebuy_rate', 'order_train_aisle_rebuy_rate',
                         'order_train_department_rebuy_rate']]

        #df['order_train_product_rebuy_rate'] = df['order_train_product_rebuy_rate'].apply(lambda x: round(x, 2))
        #df['order_train_aisle_rebuy_rate'] = df['order_train_aisle_rebuy_rate'].apply(lambda x: round(x, 2))
        #df['order_train_department_rebuy_rate'] = df['order_train_department_rebuy_rate'].apply(lambda x: round(x, 2))
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

# 用户倒数第二次购买此商品的时间间隔和次数间隔
def get_user_product_last_second(prior,user_order):
    df_path = cache_path + 'user_product_last_scend.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior_temp = prior.groupby(['user_id','product_id']).tail(2)
        prior_temp.sort_values(['user_id','product_id','order_number'],ascending=False,inplace=True)
        prior_temp = prior_temp[['user_id','product_id','order_number','date']]
        prior_temp = rank(prior_temp,['user_id','product_id'],'order_number',ascending=False,name='rank')
        df = prior_temp[prior_temp['rank']==1]
        df.rename(columns = {'date':'user_product_last_second_day','order_number':'user_product_last_scend_order'},inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df, user_n_order, on='user_id', how='left')
        df['user_product_last_scend_order'] = df['user_n_order'] - df['user_product_last_scend_order']
        df = df[['user_id','product_id','user_product_last_scend_order','user_product_last_second_day']]
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

# 用户购买此商品的平均order_number
def get_user_product_avg_order(prior):
    df_path = cache_path + 'user_product_avg_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order = get_user_order()
        user_n_order = get_user_n_order(user_order)
        order_number = prior.merge(user_n_order,on='user_id',how='left')
        order_number['order_number'] = (order_number['order_number']-1)/(order_number['user_n_order']-1+0.01)
        df = order_number.groupby(['user_id','product_id'],as_index=False)[
            'order_number'].agg({'user_product_avg_order':'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买此商品的平均add_to_cart_order（相对值）
def get_user_product_avg_cate_order(prior):
    df_path = cache_path + 'user_product_avg_cate_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_n_item = prior.groupby('order_id',as_index=False)[
            'add_to_cart_order'].agg({'order_n_item':'max'})
        add_to_cart_order = prior.merge(order_n_item, on='order_id', how='left')
        add_to_cart_order['add_to_cart_order'] = (add_to_cart_order['add_to_cart_order'] - 1) / (add_to_cart_order['order_n_item'] - 1 + 0.01)
        df = add_to_cart_order.groupby(['user_id', 'product_id'], as_index=False)[
            'add_to_cart_order'].agg({'user_product_avg_cate_order': 'mean'})
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

# 用户上次购买这个商品的时间差
def get_user_product_diff_of_time(prior,user_order):
    df_path = cache_path + 'user_product_diff_of_time.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id', 'product_id'], keep='last')
        df.rename(columns={'order_hour_of_day': 'user_product_last_hour', 'order_dow': 'user_product_last_week'}, inplace=True)
        user_candicate_order = user_order[user_order['eval_set']!='prior']
        df = pd.merge(df, user_candicate_order, on='user_id', how='left')
        df['user_product_diff_of_hour'] = (df['user_product_last_hour'] - df['order_hour_of_day']).apply(lambda x:min(abs(x),abs(24-abs(x))))
        df['user_product_diff_of_week'] = (df['user_product_last_week'] - df['order_dow']).apply(lambda x:min(abs(x),abs(7-abs(x))))
        df = df[['user_id', 'product_id', 'user_product_diff_of_hour', 'user_product_diff_of_week']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df


# 用户购买的aisle中包含多少个product_id
def get_user_aisle_n_product(prior):
    df_path = cache_path + 'user_aisle_n_product.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id', 'aisle_id'], as_index=False)[
            'product_id'].agg({'user_aisle_n_product':'nunique'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买本aisle次数/用户一共购买aisle次数（order）
def get_user_aisle_n_item_per_order(prior):
    df_path = cache_path + 'user_aisle_n_item_per_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id', 'aisle_id'], as_index=False)[
            'order_id'].agg({'user_aisle_n_item': 'count',
                             'user_aisle_n_order':'nunique'})
        df['user_aisle_n_item_per_order'] = df['user_aisle_n_item']/df['user_aisle_n_order']
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户第一次购买此aisle的时间间隔和次数间隔
def get_user_aisle_first(prior,user_order):
    df_path = cache_path + 'user_aisle_first.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id','aisle_id'],keep='first')
        df.rename(columns = {'date':'user_aisle_first_day','order_number':'user_aisle_first_order'},inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df, user_n_order, on='user_id', how='left')
        df['user_aisle_first_order'] = df['user_n_order'] - df['user_aisle_first_order']
        df = df[['user_id','aisle_id','user_aisle_first_day','user_aisle_first_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户最后一次购买此aisle的时间间隔和次数间隔
def get_user_aisle_last(prior,user_order):
    df_path = cache_path + 'user_aisle_last.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id','aisle_id'],keep='last')
        df.rename(columns = {'date':'user_aisle_last_day','order_number':'user_aisle_last_order'},inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df, user_n_order, on='user_id', how='left')
        df['user_aisle_last_order'] = df['user_n_order'] - df['user_aisle_last_order']
        df = df[['user_id','aisle_id','user_aisle_last_day','user_aisle_last_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买此aisle的平均add_to_cart_order（相对值）
def get_user_aisle_avg_cate_order(prior):
    df_path = cache_path + 'user_aisle_avg_cate_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_n_item = prior.groupby('order_id',as_index=False)[
            'add_to_cart_order'].agg({'order_n_item':'max'})
        add_to_cart_order = prior.merge(order_n_item, on='order_id', how='left')
        add_to_cart_order['add_to_cart_order'] = (add_to_cart_order['add_to_cart_order'] - 1) / (add_to_cart_order['order_n_item'] - 1 + 0.01)
        df = add_to_cart_order.groupby(['user_id', 'aisle_id'], as_index=False)[
            'add_to_cart_order'].agg({'user_aisle_avg_cate_order': 'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买此aisle的平均order_number
def get_user_aisle_avg_order(prior):
    df_path = cache_path + 'user_aisle_avg_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order = get_user_order()
        user_n_order = get_user_n_order(user_order)
        order_number = prior.merge(user_n_order,on='user_id',how='left')
        order_number['order_number'] = (order_number['order_number']-1)/(order_number['user_n_order']-1+0.01)
        df = order_number.groupby(['user_id','aisle_id'],as_index=False)[
            'order_number'].agg({'user_aisle_avg_order':'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户购买此aisle的重复购买率
def get_user_aisle_rebuy_rate(prior):
    df_path = cache_path + 'user_aisle_rebuy_rate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id','aisle_id'],as_index=False)['reordered'].agg({'user_aisle_buy':'count',
                                                                                    'user_aisle_rebuy':'sum'}).fillna(0)
        df['user_aisle_rebuy_rate'] = df['user_aisle_rebuy'] / df['user_aisle_buy']
        df = df[['user_id','aisle_id','user_aisle_rebuy_rate']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df


# 用户购买的department中包含多少个product_id
def get_user_department_n_product(prior):
    df_path = cache_path + 'user_department_n_product.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id', 'department_id'], as_index=False)[
            'product_id'].agg({'user_department_n_product':'nunique'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买本department次数/用户一共购买department次数（order）
def get_user_department_n_item_per_order(prior):
    df_path = cache_path + 'user_department_n_item_per_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id', 'department_id'], as_index=False)[
            'order_id'].agg({'user_department_n_item': 'count',
                             'user_department_n_order':'nunique'})
        df['user_department_n_item_per_order'] = df['user_department_n_item']/df['user_department_n_order']
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户第一次购买此department的时间间隔和次数间隔
def get_user_department_first(prior,user_order):
    df_path = cache_path + 'user_department_first.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id','department_id'],keep='first')
        df.rename(columns = {'date':'user_department_first_day','order_number':'user_department_first_order'},inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df, user_n_order, on='user_id', how='left')
        df['user_department_first_order'] = df['user_n_order'] - df['user_department_first_order']
        df = df[['user_id','department_id','user_department_first_day','user_department_first_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户最后一次购买此department的时间间隔和次数间隔
def get_user_department_last(prior,user_order):
    df_path = cache_path + 'user_department_last.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.drop_duplicates(['user_id','department_id'],keep='last')
        df.rename(columns = {'date':'user_department_last_day','order_number':'user_department_last_order'},inplace=True)
        user_n_order = get_user_n_order(user_order)
        df = pd.merge(df, user_n_order, on='user_id', how='left')
        df['user_department_last_order'] = df['user_n_order'] - df['user_department_last_order']
        df = df[['user_id','department_id','user_department_last_day','user_department_last_order']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买此department的平均add_to_cart_order（相对值）
def get_user_department_avg_cate_order(prior):
    df_path = cache_path + 'user_department_avg_cate_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        order_n_item = prior.groupby('order_id',as_index=False)[
            'add_to_cart_order'].agg({'order_n_item':'max'})
        add_to_cart_order = prior.merge(order_n_item, on='order_id', how='left')
        add_to_cart_order['add_to_cart_order'] = (add_to_cart_order['add_to_cart_order'] - 1) / (add_to_cart_order['order_n_item'] - 1 + 0.01)
        df = add_to_cart_order.groupby(['user_id', 'department_id'], as_index=False)[
            'add_to_cart_order'].agg({'user_department_avg_cate_order': 'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户购买此department的平均order_number
def get_user_department_avg_order(prior):
    df_path = cache_path + 'user_department_avg_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order = get_user_order()
        user_n_order = get_user_n_order(user_order)
        order_number = prior.merge(user_n_order,on='user_id',how='left')
        order_number['order_number'] = (order_number['order_number']-1)/(order_number['user_n_order']-1+0.01)
        df = order_number.groupby(['user_id','department_id'],as_index=False)[
            'order_number'].agg({'user_department_avg_order':'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户购买此department的重复购买率
def get_user_department_rebuy_rate(prior):
    df_path = cache_path + 'user_department_rebuy_rate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id','department_id'],as_index=False)['reordered'].agg({'user_department_buy':'count',
                                                                                    'user_department_rebuy':'sum'}).fillna(0)
        df['user_department_rebuy_rate'] = df['user_department_rebuy'] / df['user_department_buy']
        df = df[['user_id','department_id','user_department_rebuy_rate']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df


# 整体商品在一天内的分布：
def get_all_product_hour(prior):
    df_path = cache_path + 'all_product_hour.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['order_hour_of_day'],as_index=False)['user_id'].agg({'all_product_hour':'count'})
        df['all_product_hour'] = df['all_product_hour']/(df['all_product_hour'].sum())
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 整体商品在一周内的分布：
def get_all_product_week(prior):
    df_path = cache_path + 'all_product_week.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['order_dow'], as_index=False)['user_id'].agg({'all_product_week': 'count'})
        df['all_product_week'] = df['all_product_week'] / (df['all_product_week'].sum())
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品购买热度在一天内的分布
def get_product_hour(prior):
    df_path = cache_path + 'product_hour.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['product_id','order_hour_of_day'],as_index=False)['user_id'].agg({'product_hour':'count'})
        product_day = df.groupby('product_id',as_index=False)['product_hour'].agg({'product_day':'sum'})
        df = pd.merge(df,product_day,on='product_id',how='left')
        df['product_hour'] = df['product_hour']/df['product_day']
        all_product_hour = get_all_product_hour(prior)
        df = pd.merge(df, all_product_hour, on='order_hour_of_day', how='left')
        df['product_hour'] = df['product_hour'] / df['all_product_hour']
        df = df[['product_id','order_hour_of_day','product_hour']]
        df['product_id'] = df['product_id'].astype(int)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品购买热度在一周内的分布
def get_product_week(prior):
    df_path = cache_path + 'product_week.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['product_id','order_dow'],as_index=False)['user_id'].agg({'product_week':'count'})
        product_day = df.groupby('product_id',as_index=False)['product_week'].agg({'product_all_week':'sum'})
        df = pd.merge(df,product_day,on='product_id',how='left')
        df['product_week'] = df['product_week']/df['product_all_week']
        all_product_week = get_all_product_week(prior)
        df = pd.merge(df, all_product_week, on='order_dow', how='left')
        df['product_week'] = df['product_week'] / df['all_product_week']
        df = df[['product_id','order_dow','product_week']]
        df['product_id'] = df['product_id'].astype(int)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# aisle购买热度在一天内的分布
def get_aisle_hour(prior):
    df_path = cache_path + 'aisle_hour.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['aisle_id','order_hour_of_day'],as_index=False)['user_id'].agg({'aisle_hour':'count'})
        aisle_day = df.groupby('aisle_id',as_index=False)['aisle_hour'].agg({'aisle_day':'sum'})
        df = pd.merge(df,aisle_day,on='aisle_id',how='left')
        df['aisle_hour'] = df['aisle_hour']/df['aisle_day']
        all_product_hour = get_all_product_hour(prior)
        df = pd.merge(df, all_product_hour, on='order_hour_of_day', how='left')
        df['aisle_hour'] = df['aisle_hour'] / df['all_product_hour']
        df = df[['aisle_id','order_hour_of_day','aisle_hour']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# aisle购买热度在一周内的分布
def get_aisle_week(prior):
    df_path = cache_path + 'aisle_week.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['aisle_id','order_dow'],as_index=False)['user_id'].agg({'aisle_week':'count'})
        product_all_week = df.groupby('aisle_id',as_index=False)['aisle_week'].agg({'aisle_all_week':'sum'})
        df = pd.merge(df,product_all_week,on='aisle_id',how='left')
        df['aisle_week'] = df['aisle_week']/df['aisle_all_week']
        all_product_week = get_all_product_week(prior)
        df = pd.merge(df, all_product_week, on='order_dow', how='left')
        df['aisle_week'] = df['aisle_week'] / df['all_product_week']
        df = df[['aisle_id','order_dow','aisle_week']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# department购买热度在一天内的分布
def get_department_hour(prior):
    df_path = cache_path + 'department_hour.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['department_id','order_hour_of_day'],as_index=False)['user_id'].agg({'department_hour':'count'})
        department_day = df.groupby('department_id',as_index=False)['department_hour'].agg({'department_day':'sum'})
        df = pd.merge(df,department_day,on='department_id',how='left')
        df['department_hour'] = df['department_hour']/df['department_day']
        all_product_hour = get_all_product_hour(prior)
        df = pd.merge(df, all_product_hour, on='order_hour_of_day', how='left')
        df['department_hour'] = df['department_hour'] / df['all_product_hour']
        df = df[['department_id','order_hour_of_day','department_hour']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# department购买热度在一周内的分布
def get_department_week(prior):
    df_path = cache_path + 'department_week.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['department_id','order_dow'],as_index=False)['user_id'].agg({'department_week':'count'})
        product_all_week = df.groupby('department_id',as_index=False)['department_week'].agg({'department_all_week':'sum'})
        df = pd.merge(df,product_all_week,on='department_id',how='left')
        df['department_week'] = df['department_week']/df['department_all_week']
        all_product_week = get_all_product_week(prior)
        df = pd.merge(df, all_product_week, on='order_dow', how='left')
        df['department_week'] = df['department_week'] / df['all_product_week']
        df = df[['department_id','order_dow','department_week']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 添加二次特征
def get_second_feat(df):

    df['user_product_percent_order'] = df['user_product_n_item'] / df['user_n_order']
    df['user_product_percent_item'] = df['user_product_n_item'] / df['user_n_item']

    # 距离上一次购买的时间间隔/用户平均每次购买的时间间隔
    df['user_product_rate_day1'] = df['user_product_last_day'] / (df['user_product_avg_day_per_item']+0.01)
    # 距离上一次购买的次数间隔/用户平均每次购买的次数间隔
    df['user_product_rate_order1'] = df['user_product_last_order'] / (df['user_product_avg_order_per_item']+0.01)
    # 距离上一次购买的时间间隔/全部用户平均每次购买的时间间隔
    df['user_product_rate_day2'] = df['user_product_last_day'] / (df['product_avg_day_per_item']+0.01)
    # 距离上一次购买的次数间隔/全部用户平均每次购买的次数间隔
    df['user_product_rate_order2'] = df['user_product_last_order'] / (df['product_avg_order_per_item']+0.01)

    # 用户购买本aisle个数/用户一共购买个数（item）
    df['user_aisle_percent_of_all'] = df['user_aisle_n_item'] / df['user_n_item']

    # 距离最后一次购买时间/用户多少天购买一次
    df['user_exp_order'] = df['user_last_day'] / df['user_n_day_per_order']
    # 距离最后一次购买时间*用户平均每天购买多少个
    df['user_exp_item'] = df['user_last_day'] * df['user_n_item_per_day']

    # 商品被购买过的次数/商品被购买过的总人数
    df['product_n_item_per_people'] = df['product_n_item'] / df['product_n_people']
    df['product_n_item_per_people'] = normalize(df['product_n_item_per_people'])


    return df

#构建用户训练集和测试集
def make_train_set2(eval):
    df_path = cache_path + '%s_set.hdf' % eval
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior = get_prior()
        train = get_train()
        user_order = get_user_order()

        df = get_candicate(prior,user_order, eval)                      # 构造样本
        action_feat = get_action_feat(user_order)                       # 构造行为基础特征
        order_train_rebuy_rate = get_order_train_rebuy_rate(train,prior,user_order,eval)# train中商品重复购买比例

        user_product_n_item = get_user_product_n_item(prior)            # 用户购买此商品多少次
        user_product_first = get_user_product_first(prior,user_order)   # 用户第一次购买此商品的时间间隔和次数间隔
        user_product_last = get_user_product_last(prior,user_order)     # 用户最后一次购买此商品的时间间隔 和次数间隔
        user_product_last_second = get_user_product_last_second(prior, user_order)  # 用户倒数第二次购买此商品间隔的时间和次数
        user_product_avg_day_per_item = get_user_product_avg_day_per_item(prior,user_order) # （用户第一次购买此商品的时间间隔-用户最后一次购买此商品的时间间隔）/(用户购买本商品多少次-1)
        user_product_avg_order_per_item = get_user_product_avg_order_per_item(prior,user_order) # （用户第一次购买此商品的次数间隔-用户最后一次购买此商品的次数间隔）/(用户购买本商品多少次-1)
        user_product_avg_order = get_user_product_avg_order(prior)      # 购买的平均order_number（相对值）
        user_product_avg_cate_order = get_user_product_avg_cate_order(prior) # 用户购买此商品的平均add_to_cart_order（相对值）
        user_product_rebuy_rate = get_user_product_rebuy_rate(prior)    # 用户商品重复否购买率
        user_product_diff_of_time = get_user_product_diff_of_time(prior,user_order)# 用户上次购买这个商品的时间差


        #user_aisle_n_order = get_user_aisle_n_order(prior)              # 用户购买本aisle多少次（order）
        #user_aisle_n_item = get_user_aisle_n_item(prior)                  # 用户购买本aisle对少个（item）
        user_aisle_n_product = get_user_aisle_n_product(prior)          # 用户购买的aisle中包含多少个product_id
        user_aisle_n_item_per_order = get_user_aisle_n_item_per_order(prior)# 用户购买本aisle次数/用户一共购买aisle次数（order）
        user_aisle_first = get_user_aisle_first(prior, user_order)      # 用户第一次购买此aisle的时间间隔和次数间隔
        user_aisle_last = get_user_aisle_last(prior, user_order)        # 用户最后一次购买此aisle的时间间隔和次数间隔
        user_aisle_avg_cate_order = get_user_aisle_avg_cate_order(prior)# 用户购买此aisle的平均add_to_cart_order（相对值）
        user_aisle_avg_order = get_user_aisle_avg_order(prior)          # 用户购买此aisle的平均order_number（相对值）
        user_aisle_rebuy_rate = get_user_aisle_rebuy_rate(prior)        # 用户aisle重复购买率


        user_department_n_product = get_user_department_n_product(prior)  # 用户购买的department中包含多少个product_id
        user_department_n_item_per_order = get_user_department_n_item_per_order(prior)  # 用户购买本department次数/用户一共购买department次数（order）
        user_department_first = get_user_department_first(prior, user_order)  # 用户第一次购买此department的时间间隔和次数间隔
        user_department_last = get_user_department_last(prior, user_order)  # 用户最后一次购买此department的时间间隔和次数间隔
        user_department_avg_cate_order = get_user_department_avg_cate_order(prior)  # 用户购买此department的平均add_to_cart_order（相对值）
        user_department_avg_order = get_user_department_avg_order(prior)  # 用户购买此department的平均order_number（相对值）
        user_department_rebuy_rate = get_user_department_rebuy_rate(prior)  # 用户aisle重复购买率



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
        user_avg_day_per_item = get_user_avg_day_per_item(prior,user_order)# 用户购买商品的平均时间间隔
        user_avg_order_per_item = get_user_avg_order_per_item(prior,user_order)# 用户购买商品的平均次数间隔
        user_percent_30 = get_user_percent_30(user_order)               # 用户隔30天购买的次数占总次数的比例
        user_n_previous_item = get_user_n_previous_item(prior)          # 用户上一次购买个数
        user_word2vec = get_user_word2vec(prior)                        # 添加用户的work2vec
        user_rebuy_rate_by_product = get_user_rebuy_rate_by_product(prior)  # 根据商品重复购买率判断用户类型


        product_feat = get_product()[['product_id','aisle_id','department_id']]# 商品基础特征
        product_n_item = get_product_n_item(prior)                      # 商品被购买过的次数
        product_n_people = get_product_n_people(prior)                  # 商品被购买过的人数
        product_rebuy_rate = get_product_rebuy_rate(prior)              # 商品重复购买率
        product_avg_day_per_item = get_product_avg_day_per_item(prior)  # 商品被购买的平均时间间隔
        product_avg_order_per_item = get_product_avg_order_per_item(prior,user_order)# 商品被购买的平均次数间隔
        product_percent_last = get_product_last_percent(prior)          # 商品最后一次被购买占的比例
        product_percent_last2 = get_product_last_percent2(prior)        # 商品最后两次被购买占的比例
        product_order = get_product_order(prior)                        # 商品的order
        product_cart_order = get_product_cart_order(prior)              # 商品的平均add_to_cart_order
        product_word2vec = get_product_word2vec(prior)                  # 添加商品的work2vec
        product_word2vec2 = get_product_word2vec2(prior)                # 添加商品的work2vec2
        product_rebuy_rate_by_user = get_product_rebuy_rate_buy_user(prior)  # 通过商品购买的用户计算商品购买率

        aisle_feat = get_aisle_feat(prior)                              # aisle基础特征
        product_hour        = get_product_hour(prior)                   # 商品购买热度在一天内的分布
        product_week        = get_product_week(prior)                   # 商品购买热度在一周内的分布
        aisle_hour          = get_aisle_hour(prior)                     # aisle购买热度在一天内的分布
        aisle_week          = get_aisle_week(prior)                     # aisle购买热度在一周内的分布
        department_hour     = get_department_hour(prior)                # department购买热度在一天内的分布
        department_week     = get_department_week(prior)                # department购买热度在一周内的分布




        print('将特征组合到一起')
        df = pd.merge(df, action_feat,                      on='order_id', how='left')
        df = pd.merge(df, order_train_rebuy_rate,           on=['order_id','product_id'], how='left')

        df = pd.merge(df, user_product_n_item,              on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_first,               on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_last,                on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_last_second,         on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_avg_day_per_item,    on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_avg_order_per_item,  on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_avg_order,           on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_avg_cate_order,      on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_rebuy_rate,          on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_diff_of_time,        on=['user_id', 'product_id'], how='left')

        df = pd.merge(df, product_feat,                     on='product_id', how='left')
        df = pd.merge(df, user_aisle_n_product,             on=['user_id', 'aisle_id'], how='left')
        df = pd.merge(df, user_aisle_n_item_per_order,      on=['user_id', 'aisle_id'], how='left')
        df = pd.merge(df, user_aisle_first,                 on=['user_id', 'aisle_id'], how='left')
        df = pd.merge(df, user_aisle_last,                  on=['user_id', 'aisle_id'], how='left')
        df = pd.merge(df, user_aisle_avg_cate_order,        on=['user_id', 'aisle_id'], how='left')
        df = pd.merge(df, user_aisle_avg_order,             on=['user_id', 'aisle_id'], how='left')
        df = pd.merge(df, user_aisle_rebuy_rate,            on=['user_id', 'aisle_id'], how='left')

        # df = pd.merge(df, user_department_n_product,        on=['user_id', 'department_id'], how='left')
        # df = pd.merge(df, user_department_n_item_per_order, on=['user_id', 'department_id'], how='left')
        # df = pd.merge(df, user_department_first,            on=['user_id', 'department_id'], how='left')
        # df = pd.merge(df, user_department_last,             on=['user_id', 'department_id'], how='left')
        # df = pd.merge(df, user_department_avg_cate_order,   on=['user_id', 'department_id'], how='left')
        # df = pd.merge(df, user_department_avg_order,        on=['user_id', 'department_id'], how='left')
        # df = pd.merge(df, user_department_rebuy_rate,       on=['user_id', 'department_id'], how='left')

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

        df = pd.merge(df, product_n_item,                   on='product_id', how='left')
        df = pd.merge(df, product_n_people,                 on='product_id', how='left')
        df = pd.merge(df, product_rebuy_rate,               on='product_id', how='left')
        df = pd.merge(df, product_avg_day_per_item,         on='product_id', how='left')
        df = pd.merge(df, product_avg_order_per_item,       on='product_id', how='left')
        df = pd.merge(df, product_percent_last,             on='product_id', how='left')
        df = pd.merge(df, product_percent_last2,            on='product_id', how='left')
        df = pd.merge(df, product_order,                    on='product_id', how='left')
        df = pd.merge(df, product_cart_order,               on='product_id', how='left')
        df = pd.merge(df, product_word2vec,                 on='product_id', how='left')
        df = pd.merge(df, product_word2vec2,                 on='product_id', how='left')
        df = pd.merge(df, product_rebuy_rate_by_user,       on='product_id', how='left')

        df = pd.merge(df, aisle_feat,                       on='aisle_id', how='left')
        df = pd.merge(df, product_hour,                     on=['product_id', 'order_hour_of_day'], how='left')
        df = pd.merge(df, product_week,                     on=['product_id', 'order_dow'], how='left')
        df = pd.merge(df, aisle_hour,                       on=['aisle_id', 'order_hour_of_day'], how='left')
        df = pd.merge(df, aisle_week,                       on=['aisle_id', 'order_dow'], how='left')
        df = pd.merge(df, department_hour,                  on=['department_id', 'order_hour_of_day'], how='left')
        df = pd.merge(df, department_week,                  on=['department_id', 'order_dow'], how='left')

        df = get_second_feat(df)                            # 添加二次特征
        print('添加label')
        df = pd.merge(df, train[['user_id', 'product_id', 'label']], on=['user_id', 'product_id'], how='left')
        df['label'].fillna(0, inplace=True)
        df = df.fillna(-100)
        df = df[df['user_n_order'] > 2]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
        print('data size: {}'.format(df.shape))
        print(eval + ' over~')
    return df
user_order = get_user_order()
df_train2 = make_train_set2('train')
dump(df_train2,'df_train2')