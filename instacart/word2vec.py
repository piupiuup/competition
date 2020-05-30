import os
import gensim
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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

# 读取prior
def get_prior():
    df_path = r'F:\cache\instacart_cache\prior.hdf'
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
    df_path = r'F:\cache\instacart_cache\train.hdf'
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
    df_path = r'F:\cache\instacart_cache\product.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'products.csv')
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

order_products_train = get_train()
order_products_prior = get_prior()
product = get_product()

order_products_prior['product_id'] = order_products_prior['product_id'].astype(str)


products_prior = order_products_prior.groupby('user_id').apply(lambda row: row['product_id'].tolist())

model = gensim.models.Word2Vec(products_prior.values, size=100, window=5, min_count=2,workers=4)
model.save('product2vec.model')

def get_vector_representation(row, pos):
    return model[row.product_id][pos] if row.product_id in model else None

pca = PCA(n_components=2)
word2vec_new = pca.fit_transform(model.wv.syn0)
model.wv.syn0 = word2vec_new

product['product_id'] = product['product_id'].astype(str)
product['product_vector_1'] = product.apply(lambda row: get_vector_representation(row,0),axis=1)
product['product_vector_2'] = product.apply(lambda row: get_vector_representation(row,1),axis=1)

feats = ['product_id','product_vector_1','product_vector_2']
product['product_id'] = product['product_id'].astype(int)
dump(product[feats],'product_word2vec')


# 获取用户
    order_products_prior['user_id'] = order_products_prior['user_id'].astype(str)
    user_prior = order_products_prior.groupby('product_id').apply(lambda row: row['user_id'].tolist())

    model = gensim.models.Word2Vec(user_prior.values, size=100, window=5, min_count=2,workers=4)
    model.save('user2vec.model')

    def get_vector_representation(row, pos):
        return model[row.user_id][pos] if row.user_id in model else None

    pca = PCA(n_components=2)
    word2vec_new = pca.fit_transform(model.wv.syn0)
    model.wv.syn0 = word2vec_new
    user = order_products_prior[['user_id']].drop_duplicates()
    user['user_id'] = user['user_id'].astype(str)
    user['user_vector_1'] = user.apply(lambda row: get_vector_representation(row,0),axis=1)
    user['user_vector_2'] = user.apply(lambda row: get_vector_representation(row,1),axis=1)
    user['user_id'] = user['user_id'].astype(int)

dump(user,'user_word2vec')













