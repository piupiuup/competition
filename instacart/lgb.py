import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.stats as scs


IDIR = r'C:\Users\csw\Desktop\python\instacart\data\\'

def f1(y_true,y_pred):
    if (type(y_true) == float) or (len(y_true)==0):
        if (type(y_pred) == float) or (len(y_pred)==0):
            return 1
        else:
            y_true = []
    if type(y_pred) == float:
        y_pred = []
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
        exp = 1
        for pred in pred_list:
            exp = exp * (1-pred)
        for pred in pred_list:
            n_product += 1
            TP += pred
            f1 = TP/n_product
            if f1 < f1_temp:
                if exp > f1_temp:
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
    df_path = r'F:\cache\instacart_cache\user_order.hdf'
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

# 读取prior
def get_prior():
    df_path = r'F:\cache\instacart_cache\prior.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = pd.read_csv(IDIR + 'order_products__prior.csv')
        user_order = get_user_order()
        df = pd.merge(df,user_order,on='order_id',how='left')
        df.sort_values(['user_id','product_id','order_number'],ascending=True,inplace=True)
        product = get_product()
        df = pd.merge(df,product[['product_id','aisle_id','department_id']])
        del df['eval_set']
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

# 构造样本集
def get_candicate(prior,user_order):
    df_path = r'F:\cache\instacart_cache\candicate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order_temp = user_order[user_order['eval_set'] != 'prior']
        df = pd.merge(user_order_temp[['user_id','order_id']],
                      prior[['user_id','product_id']], on='user_id', how='left')
        df = df.drop_duplicates(['user_id', 'product_id'])
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户活跃天数
def get_user_feat(prior,user_order):
    df_path = r'F:\cache\instacart_cache\user_feat.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order_temp = user_order[user_order['eval_set'] == 'prior']
        df = user_order_temp.groupby('user_id')['order_id'].agg({'user_n_order':'count'})           # 用户购买次数
        df['user_n_day'] = user_order_temp.groupby('user_id')['days_since_prior_order'].sum()       # 用户购买时间跨度
        df['user_n_item'] = prior.groupby('user_id')['product_id'].count()                          # 用户购买商品总个数
        df['user_n_product'] = prior.groupby('user_id')['product_id'].nunique()                     # 用户购买商品种类数
        df['user_avg_day_per_order'] = df['user_n_day'] / (df['user_n_order']-1)                    # 用户平均每隔多少天购买一次
        df['user_avg_item_per_order'] = df['user_n_item'] / df['user_n_order']                      # 用户平均每次购买多少个
        df['user_avg_item_per_day'] = df['user_avg_item_per_order'] / (df['user_avg_day_per_order']+0.01)  # 用户平均每天购买都少个
        # 用户平均每次购买的新增商品
        temp = prior[~prior['days_since_prior_order'].isnull()]
        df['user_n_new_product'] = temp[temp['reordered']==0].groupby('user_id')['reordered'].count()# 用户购买新增商品个数
        df['user_avg_new_per_order'] = df['user_n_new_product'] / (df['user_n_order']-1)             # 用户平均每次购买多少个
        user_product_n_item = get_user_product_avg_day_per_item(prior)
        df['user_avg_order_per_product'] = user_product_n_item.groupby('user_id')['user_product_n_item'].mean()
        df['user_avg_order_per_product'] = df['user_avg_order_per_product']/df['user_n_order']
        df['user_percent_of_new'] = df['user_avg_new_per_order']/df['user_avg_item_per_order']
        del temp,df['user_n_new_product']
        df.reset_index(inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 用户基础特征
def get_user_feat(prior,user_order):
    df_path = r'F:\cache\instacart_cache\user_feat.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_order_temp = user_order[user_order['eval_set'] == 'prior']
        df = user_order_temp.groupby('user_id')['order_id'].agg({'user_n_order':'count'})           # 用户购买次数
        df['user_n_day'] = user_order_temp.groupby('user_id')['days_since_prior_order'].sum()       # 用户购买时间跨度
        df['user_n_item'] = prior.groupby('user_id')['product_id'].count()                          # 用户购买商品总个数
        df['user_n_product'] = prior.groupby('user_id')['product_id'].nunique()                     # 用户购买商品种类数
        df['user_avg_day_per_order'] = df['user_n_day'] / (df['user_n_order']-1)                    # 用户平均每隔多少天购买一次
        df['user_avg_item_per_order'] = df['user_n_item'] / df['user_n_order']                      # 用户平均每次购买多少个
        df['user_avg_item_per_day'] = df['user_avg_item_per_order'] / (df['user_avg_day_per_order']+0.01)  # 用户平均每天购买都少个
        # 用户平均每次购买的新增商品
        temp = prior[~prior['days_since_prior_order'].isnull()]
        df['user_n_new_product'] = temp[temp['reordered']==0].groupby('user_id')['reordered'].count()# 用户购买新增商品个数
        df['user_avg_new_per_order'] = df['user_n_new_product'] / (df['user_n_order']-1)             # 用户平均每次购买多少个
        user_product_n_item = get_user_product_avg_day_per_item(prior)
        df['user_avg_order_per_product'] = user_product_n_item.groupby('user_id')['user_product_n_item'].mean()
        df['user_avg_order_per_product'] = df['user_avg_order_per_product']/df['user_n_order']
        df['user_percent_of_new'] = df['user_avg_new_per_order']/df['user_avg_item_per_order']
        del temp,df['user_n_new_product']
        df.reset_index(inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品基础特征
def get_product_feat(prior):
    df_path = r'F:\cache\instacart_cache\product_feat.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:

        df = prior.groupby('product_id')['product_id'].agg({'product_item_count':'count'})  # 这个产品被所有人购买过多少次
        df['product_n_user'] = prior.groupby('product_id')['user_id'].nunique()             # 这个产品被多少分购买过
        df['product_avg_item_per_user'] = df['product_item_count'] / df['product_n_user']   # 平均每人购买多少次
        temp = prior.groupby(['product_id', 'user_id'], as_index=False)['order_dow'].count()
        df['product_std_pre_user'] = temp.groupby('product_id')['order_dow'].std()          # 每个人购买次数的方差
        df['product_skew_pre_user'] = temp.groupby('product_id')['order_dow'].agg({'product_skew_pre_user':skew})# 每个人购买次数的偏度指数
        df.reset_index(inplace=True)
        product = get_product()
        df = pd.merge(df,product[['product_id', 'aisle_id', 'department_id']],on='product_id',how='left')
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# aisle基础特征
def get_aisle_feat(prior):
    df_path = r'F:\cache\instacart_cache\aisle_feat.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('aisle_id')['aisle_id'].agg({'aisle_item_count':'count'})        # 这个aisle被所有人购买过多少次
        df['aisle_n_user'] = prior.groupby('aisle_id')['user_id'].nunique()                 # 这个aisle被多少分购买过
        df['aisle_avg_item_per_user'] = df['aisle_item_count'] / df['aisle_n_user']         # 平均每人购买多少次
        temp = prior.groupby(['aisle_id', 'user_id'], as_index=False)['aisle-id'].agg({'aisle_user_n_item':'count'})
        df['aisle_std_pre_user'] = temp.groupby('aisle_id')['aisle_user_n_item'].std()      # 每个人购买次数的方差
        df['aisle_skew_pre_user'] = temp.groupby('aisle_id')['aisle_user_n_item'].agg({'aisle_skew_pre_user':skew})# 每个人购买次数的偏度指数
        df.reset_index(inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 平均多少天购买一次
def get_product_mdn_per_day(prior):
    df_path = r'F:\cache\instacart_cache\product_mdn_per_day.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_avg_day_per_item = get_user_product_avg_day_per_item(prior)
        df = user_product_avg_day_per_item.groupby('product_id',as_index=False)[
            'user_product_avg_day_per_item'].agg({'product_mdn_per_day':'median'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 产品平均每次被购买的概率
def get_product_mdn_per_order(prior):
    df_path = r'F:\cache\instacart_cache\product_mdn_per_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_expectation_per_order = get_user_product_expectation_per_order(prior)
        df = user_product_expectation_per_order.groupby('product_id', as_index=False)[
            'user_product_expectation_per_order1'].agg({'product_mdn_per_order': 'median'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品被重复购买的几率
def get_product_percent_less_than_2(prior):
    df_path = r'F:\cache\instacart_cache\product_percent_less_than_2.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_n_item = prior.groupby(['user_id', 'product_id'], as_index=False)['user_id'].agg(
            {'user_product_n_item': 'count'})
        user_product_n_item['less than 2'] = (user_product_n_item['user_product_n_item'] < 2).astype(np.int32)
        product_percent_less_than_2 = user_product_n_item.groupby('product_id')[
                                          'less than 2'].sum() / user_product_n_item.groupby('product_id').size()
        df = pd.DataFrame(product_percent_less_than_2).reset_index()  # 有多少人购买了一次就不再购买了
        df.columns = ['product_id','product_percent_less_than_2']
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 所有产品的order中位数
def get_product_avg_order(prior) :
    df_path = r'F:\cache\instacart_cache\product_avg_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        user_product_avg_order = get_user_product_avg_order(prior)
        df = user_product_avg_order.groupby('product_id',as_index=False)[
            'user_product_avg_order'].agg({'product_avg_order':'median'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品第一次购买次数/商品购买的总次数
def get_product_precent_reorder(prior):
    df_path = r'F:\cache\instacart_cache\product_precent_reorder.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby('product_id')['user_id'].agg({'product_n_user': 'nunique'})
        df['product_n_item'] = prior.groupby('product_id')['user_id'].count()
        df['product_precent_reorder'] = df['product_n_user']/df['product_n_item']
        df.reset_index(inplace=True)
        df = df[['product_id','product_precent_reorder']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品最近一次购买占全部购买的比例
def get_product_precent_last(prior):
    df_path = r'F:\cache\instacart_cache\product_precent_last.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.groupby('user_id',as_index=False)['order_number'].max()
        temp = pd.merge(temp,prior,on=['user_id','order_number'],how='left')
        df = prior.groupby('product_id')['product_id'].agg({'product_n_item':'count'})
        df['product_last_n_item'] = temp.groupby('product_id')['product_id'].count()
        df = df.reset_index().fillna(0)
        df['product_precent_last'] = df['product_last_n_item']/df['product_n_item']
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品重复购买率
def get_product_rebuy_rate(prior):
    df_path = r'F:\cache\instacart_cache\product_rebuy_rate.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.copy()
        temp['user_product_rank'] = temp.groupby(['user_id', 'product_id']).cumcount() + 1
        temp['buy'] = temp['user_product_rank'].apply(lambda x: x * (x + 1) / 2 - 1)
        temp['rebuy'] = temp['user_product_rank'].apply(lambda x: x * (x - 1) / 2)
        df = temp.groupby('product_id').agg({'buy': {'product_sum_of_buy': 'sum'},
                                            'rebuy': {'product_sum_of_rebuy': 'sum'}}).fillna(0)
        df.columns = df.columns.droplevel(0)
        df.reset_index(inplace=True)
        df['product_rebuy_rate'] = df['product_sum_of_rebuy'] / df['product_sum_of_buy']
        df = df[['product_id', 'product_rebuy_rate']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 行为基础特征
def get_action_feat(user_order):
    df_path = r'F:\cache\instacart_cache\action.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = user_order[user_order['eval_set'] != 'prior'][[
            'order_id',  'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        df.rename(columns={'order_number':'user_n_order','days_since_prior_order':'user_last_days'},inplace=True)
         #次数， 周几，时间，距离上次天数
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户购买此商品多少次
def get_user_product_n_item(prior):
    df_path = r'F:\cache\instacart_cache\user_product_n_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id','product_id'],as_index=False)[
            'user_id'].agg({'user_product_n_item':'count'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 此用户平均多少天购买一次此商品
def get_user_product_avg_day_per_item(prior):
    df_path = r'F:\cache\instacart_cache\user_product_avg_day_per_item.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.copy()
        temp.sort_values('date',ascending=True,inplace=True)
        user_product_max_date = temp.drop_duplicates(['user_id','product_id'],keep='last')[['user_id','product_id','date']]
        user_product_n_item = prior.groupby(['user_id','product_id'],as_index=False)['user_id'].agg({'user_product_n_item':'count'})
        df = pd.merge(user_product_max_date,user_product_n_item,on=['user_id','product_id'],how='left')
        df['user_product_avg_day_per_item'] = df['date']/(df['user_product_n_item']-1+0.01)
        df = df[['user_id','product_id','user_product_n_item','user_product_avg_day_per_item']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 产品平均每次被购买的概率
def get_user_product_expectation_per_order(prior):
    df_path = r'F:\cache\instacart_cache\user_user_product_expectation_per_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.copy()
        temp.sort_values('order_number', inplace=True)
        user_product_min_order = temp.drop_duplicates(['user_id', 'product_id'], keep='first')[
            ['user_id', 'product_id', 'order_number']]
        user_product_max_order = temp.groupby(['user_id', 'product_id'],as_index=False)[
            'order_number'].agg({'user_product_max_order':'max'})
        df = pd.merge(user_product_min_order,user_product_max_order,on=['user_id', 'product_id'],how='left')
        df['user_product_n_order'] = df['user_product_max_order'] - df['order_number']
        user_product_n_item = prior.groupby(['user_id', 'product_id'], as_index=False)['user_id'].agg(
            {'user_product_n_item': 'count'})
        df = pd.merge(df,user_product_n_item,on=['user_id', 'product_id'],how='left')
        df['user_product_expectation_per_order1'] = (df['user_product_n_item'] - 0.5) / (
            df['user_product_n_order'] + 0.01)
        df['user_product_expectation_per_order2'] = (df['user_product_n_item'] - 0.5) / (
            df['user_product_max_order'] + 0.01)
        df = df[['user_id', 'product_id', 'user_product_expectation_per_order1',
                 'user_product_expectation_per_order2']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 购买的平均order_number
def get_user_product_avg_order(prior):
    df_path = r'F:\cache\instacart_cache\user_product_avg_order.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['user_id','product_id'],as_index=False)[
            'order_number'].agg({'user_product_avg_order':'mean'})
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 本次购买距离上一次购买时间
def get_user_product_last_time(prior):
    df_path = r'F:\cache\instacart_cache\user_product_last_time.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.copy()
        user_order = get_user_order()
        temp.sort_values('date', ascending=True, inplace=True)
        user_product_min_date = temp.drop_duplicates(['user_id', 'product_id'], keep='first')[
            ['user_id', 'product_id', 'order_number', 'date']]
        user_product_min_date.rename(columns={'order_number':'user_product_last_order'},inplace=True)
        df = pd.merge(user_product_min_date, user_order[user_order['eval_set']!='prior'], on='user_id', how='left')
        df['user_product_last_time'] = df['date'] + df['days_since_prior_order']
        user_product_avg_day_per_item = get_user_product_avg_day_per_item(prior)
        product_mdn_per_day = get_product_mdn_per_day(prior)
        df = pd.merge(df, user_product_avg_day_per_item, on=['user_id','product_id'],how='left')
        df = pd.merge(df, product_mdn_per_day, on='product_id', how='left')
        df['expectation_of_day_product'] = df['user_product_last_time'] / (df['product_mdn_per_day']+0.01)
        df['expectation_of_day_user_product'] = df['user_product_last_time'] / (df['user_product_avg_day_per_item']+0.01)
        df = df[['user_id', 'product_id', 'user_product_last_time','user_product_last_order',
                 'expectation_of_day_product','expectation_of_day_user_product']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 本次购买距离第一次购买时间
def get_user_product_first_time(prior):
    df_path = r'F:\cache\instacart_cache\user_product_first_time.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        temp = prior.copy()
        user_order = get_user_order()
        temp.sort_values('date', ascending=True, inplace=True)
        user_product_max_date = temp.drop_duplicates(['user_id', 'product_id'], keep='last')[
            ['user_id', 'product_id', 'order_number', 'date']]
        user_product_max_date.rename(columns={'order_number':'user_product_first_order'},inplace=True)
        df = pd.merge(user_product_max_date, user_order[user_order['eval_set']!='prior'], on='user_id', how='left')
        df['user_product_first_time'] = df['date'] + df['days_since_prior_order']
        df = df[['user_id', 'product_id', 'user_product_first_order','user_product_first_time']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 整体商品在一天内的分布：
def get_all_product_hour(prior):
    df_path = r'F:\cache\instacart_cache\all_product_hour.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['order_hour_of_day'],as_index=False)['user_id'].agg({'all_product_hour':'count'})
        df['all_product_hour'] = df['all_product_hour']/(df['all_product_hour'].sum())
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df


# 整体商品在一周内的分布：
def get_all_product_week(prior):
    df_path = r'F:\cache\instacart_cache\all_product_week.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        df = prior.groupby(['order_dow'], as_index=False)['user_id'].agg({'all_product_week': 'count'})
        df['all_product_week'] = df['all_product_week'] / (df['all_product_week'].sum())
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品购买热度在一天内的分布
def get_product_hour(prior):
    df_path = r'F:\cache\instacart_cache\product_hour.hdf'
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
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 商品购买热度在一周内的分布
def get_product_week(prior):
    df_path = r'F:\cache\instacart_cache\product_week.hdf'
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
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# aisle购买热度在一天内的分布
def get_aisle_hour(prior):
    df_path = r'F:\cache\instacart_cache\aisle_hour.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        product = get_product()
        temp = pd.merge(prior,product,on='product_id',how='left')
        df = temp.groupby(['aisle_id','order_hour_of_day'],as_index=False)['user_id'].agg({'aisle_hour':'count'})
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
    df_path = r'F:\cache\instacart_cache\aisle_week.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        product = get_product()
        temp = pd.merge(prior, product, on='product_id', how='left')
        df = temp.groupby(['aisle_id','order_dow'],as_index=False)['user_id'].agg({'aisle_week':'count'})
        product_all_week = df.groupby('aisle_id',as_index=False)['aisle_week'].agg({'aisle_all_week':'sum'})
        df = pd.merge(df,product_all_week,on='aisle_id',how='left')
        df['aisle_week'] = df['aisle_week']/df['aisle_all_week']
        all_product_week = get_all_product_week(prior)
        df = pd.merge(df, all_product_week, on='order_dow', how='left')
        df['aisle_week'] = df['aisle_week'] / df['all_product_week']
        df = df[['aisle_id','order_dow','aisle_week']]
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

# 添加二次特征
def get_second_feat(df):
    df['user_product_last_order'] = df['user_n_order'] - df['user_product_last_order']
    df['user_product_first_order'] = df['user_n_order'] - df['user_product_first_order']

    return df

#构建用户训练集和测试集
def make_train_set():
    df_path = r'F:\cache\instacart_cache\train_set.hdf'
    if os.path.exists(df_path) & 1:
        df = pd.read_hdf(df_path, 'w')
    else:
        prior = get_prior()
        train = get_train()
        user_order = get_user_order()

        df = get_candicate(prior,user_order)                            # 构造样本
        action_feat = get_action_feat(user_order)                       # 构造行为基础特征

        user_product_n_item = get_user_product_n_item(prior)            # 用户购买此商品多少次
        user_product_avg_day_per_item = get_user_product_avg_day_per_item(prior)  # 此用户平均多少天购买一次此商品
        user_product_expectation_per_order = get_user_product_expectation_per_order(prior)  # 产品平均每次被购买的概率
        user_product_avg_order = get_user_product_avg_order(prior)      # 购买的平均order_number
        user_product_last_time = get_user_product_last_time(prior)      # 本次购买距离上一次购买时间
        user_product_first_time = get_user_product_first_time(prior)    # 本次购买距离第一次购买时间

        user_n_day = get_user_n_day(user_order)                         # 用户活跃天数
        user_feat           = get_user_feat(prior,user_order)           # 构造用户基础特征

        product_feat        = get_product_feat(prior)                   # 构造商品基础特征
        product_mdn_per_day = get_product_mdn_per_day(prior)            # 全部用户平均多少天购买一次
        product_mdn_per_order = get_product_mdn_per_order(prior)        # 产品平均每次被购买的概率
        product_percent_less_than_2 = get_product_percent_less_than_2(prior)  # 产品被用户重复购买的概率
        product_avg_order = get_product_avg_order(prior)                # 所有产品的order中位数
        product_precent_reorder = get_product_precent_reorder(prior)    # 商品第一次购买次数/商品购买的总次数
        product_precent_last = get_product_precent_last(prior)          # 商品最近一次购买占全部购买的比例
        product_rebuy_rate = get_product_rebuy_rate(prior)              # 商品重复购买率

        aisle_feat = get_aisle_feat(prior)                              # aisle基础特征


        # 本次购买距离上一次购买次数


        product_hour        = get_product_hour(prior)                   # 商品购买热度在一天内的分布
        product_week        = get_product_week(prior)                   # 商品购买热度在一周内的分布
        aisle_hour          = get_aisle_hour(prior)                     # aisle购买热度在一天内的分布
        aisle_week          = get_aisle_week(prior)                     # aisle购买热度在一周内的分布
        #department_hour     = get_department_hour(prior)                # department购买热度在一天内的分布
        #department_week     = get_department_week(prior)                # department购买热度在一周内的分布


        print('将特征组合到一起')
        df = pd.merge(df, user_feat,                        on='user_id', how='left')
        df = pd.merge(df, action_feat,                      on='order_id', how='left')
        df = pd.merge(df, product_feat,                     on='product_id', how='left')
        df = pd.merge(df, product_mdn_per_day,              on='product_id', how='left')
        df = pd.merge(df, product_mdn_per_order,            on='product_id', how='left')
        df = pd.merge(df, product_percent_less_than_2,      on='product_id', how='left')
        df = pd.merge(df, product_avg_order,                on='product_id', how='left')
        df = pd.merge(df, product_precent_reorder,          on='product_id', how='left')
        df = pd.merge(df, product_precent_last,             on='product_id', how='left')
        df = pd.merge(df, product_rebuy_rate,               on='product_id', how='left')
        df = pd.merge(df, aisle_feat,                       on='aisle', how='left')
        df = pd.merge(df, user_product_avg_day_per_item,    on=['user_id','product_id'], how='left')
        df = pd.merge(df, user_product_expectation_per_order,on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_avg_order,           on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_last_time,           on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, user_product_first_time,          on=['user_id', 'product_id'], how='left')
        df = pd.merge(df, product_hour,                     on=['product_id', 'order_hour_of_day'], how='left')
        df = pd.merge(df, product_week,                     on=['product_id', 'order_dow'], how='left')
        df = pd.merge(df, aisle_hour,                       on=['aisle_id', 'order_hour_of_day'], how='left')
        df = pd.merge(df, aisle_week,                       on=['aisle_id', 'order_dow'], how='left')

        df = get_second_feat(df)                            # 添加二次特征
        print('添加label')
        df = pd.merge(df, train[['user_id', 'product_id', 'label']], on=['user_id', 'product_id'], how='left')
        df['label'].fillna(0, inplace=True)
        df.to_hdf(df_path, 'w', complib='blosc', complevel=5)
    return df

df = make_train_set()
df = df.fillna(-100)

user_order = get_user_order()
train_user_list = list(user_order[user_order['eval_set']=='train']['user_id'].unique())
test_user_list = list(user_order[user_order['eval_set']=='test']['user_id'].unique())
df_train = df[df['user_id'].isin(train_user_list)]
df_test = df[df['user_id'].isin(test_user_list)]

# 线下调参
train = df_train[:int(df_train.shape[0]*0.7)]
test = df_train[int(df_train.shape[0]*0.7):]

features = [ 'user_n_order', 'user_n_day',
       'user_n_item', 'user_n_product', 'user_avg_day_per_order',
       'user_avg_item_per_order', 'user_avg_item_per_day',
       'user_avg_new_per_order', 'user_percent_of_new', 'product_item_count',
       'product_n_user', 'product_avg_item_per_user', 'product_std_pre_user',
       'product_skew_pre_user', 'aisle_id', 'department_id',
       'product_mdn_per_day', 'product_mdn_per_order',
       'product_percent_less_than_2', 'product_avg_order',
       'product_precent_reorder', 'order_dow', 'order_hour_of_day',
       'days_since_prior_order', 'user_product_n_item',
       'user_product_avg_day_per_item', 'user_product_expectation_per_order1',
        'user_product_expectation_per_order2',
       'user_product_avg_order', 'user_product_last_time',
       'user_product_last_order', 'expectation_of_day_product',
       'expectation_of_day_user_product', 'user_product_first_order',
       'user_product_first_time', 'product_hour', 'product_week',
        'aisle_hour','aisle_week','user_avg_order_per_product',
        'product_precent_last','product_rebuy_rate']

lgb_train = lgb.Dataset(train[features],train.label)
lgb_eval = lgb.Dataset(test[features],test.label, reference=lgb_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth':5,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
print('Start training...')
# train

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                verbose_eval = 10,
                early_stopping_rounds=10)

preds = gbm.predict(test[features])
test['pred'] = preds
TRESHOLD = 0.175
y_true = get_result(test[test['label']==1])
y_true = pd.merge(test[['order_id']].drop_duplicates(),y_true,on='order_id',how='left')
#y_pred = get_result(test[test['pred']>TRESHOLD])
y_pred = get_result2(test)
y_pred = pd.merge(y_true[['order_id']],y_pred,on='order_id',how='left')
print('f1得分为：%f' % (instacart_grade(y_true,y_pred)))


y_true = get_result(test[test['label']==1])
order_n_product = test[test['label']==1].groupby('order_id').size()
y_pred = get_result2(test,order_n_product)

'''
# xgb参数测试
import xgboost
xgb_train = xgboost.DMatrix(train[features],train.label)
xgb_eval = xgboost.DMatrix(test[features],test.label)

xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(xgb_eval, "test")]
bst = xgboost.train(params=xgb_params,
                    dtrain=xgb_train,
                    num_boost_round=5000,
                    evals=watchlist,
                    verbose_eval=10,
                    early_stopping_rounds=10)
'''


################### 线上提交 ###################
d_train = lgb.Dataset(df_train[features],df_train.label)
d_test = lgb.Dataset(df_test[features], reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth':5,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

ROUNDS = 612
print('light GBM train :-)')
bst = lgb.train(params, d_train, ROUNDS)
print('light GBM predict')
preds = bst.predict(df_test[features])
df_test['pred'] = preds
TRESHOLD = 0.2
#y_pred = get_result(test[test['pred']>TRESHOLD])
y_pred = get_result2(df_test)
y_pred['products'] = y_pred['products'].apply(lambda x: list_to_str(x))
y_pred = pd.merge(user_order[user_order['eval_set']=='test'][['order_id']],y_pred,on='order_id',how='left')
y_pred.to_csv(r'C:\Users\csw\Desktop\python\instacart\submission\0724(1).csv', index=False)

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in user_order[user_order['eval_set']=='test'].order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv(r'C:\Users\csw\Desktop\python\instacart\submission\0723(1).csv', index=False)


