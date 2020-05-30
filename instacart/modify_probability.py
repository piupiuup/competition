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

# 第二种按照最佳阀值获取结果
def get_result2(data,order_none=None):
    '''
    :param data: pd.DataFrame格式  包含['order_id','product_id','pred']
    :return: 返回 pd.DataFrame 格式结果  ['order_id','products']
    '''
    # 寻找最佳阀值
    def get_max_exp(pred_list, n_product, order_id,order_none):
        f1_temp = 0     # 期望f1
        TP = 0          # 期望正确个数
        PNone = (1.0-pred_list).prod() if order_none is None else order_none[order_id]
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
        TRESHOLD = get_max_exp(grouped['pred'].values,user_n_product[order_id],order_id,order_none)#输入概率备选商品的购买概率，获取最佳阀值
        result[order_id] = list(grouped['product_id'].values[grouped['pred'].values>TRESHOLD])  # 根据阀值选择商品
        result[order_id] = [None] if len(result[order_id])==0 else result[order_id]
    result = pd.Series(result).to_frame()
    result.reset_index(inplace=True)
    result.columns = ['order_id','products']
    return result

def modify_probability(order_product_prob,order_nlabel,c=0.92):
    order_sum = order_product_prob.groupby('order_id',as_index=False)['pred'].agg({'order_sum':'sum'})
    order_sum = pd.merge(order_sum,order_nlabel,on='order_id',how='left')
    order_sum['m'] = order_sum['order_sum']/order_sum['pred_nlabel']*(1-c) + c
    order_product_prob_temp = order_product_prob.copy()
    order_product_prob_temp = order_product_prob_temp.merge(order_sum[['order_id','m']],on='order_id',how='left')
    order_product_prob_temp['pred'] = order_product_prob_temp['pred'] * order_product_prob_temp['m']

    return order_product_prob_temp[['order_id','product_id','pred']]

df_pred = load('df_pred')
order_nlable = load('order_nlabel')
order_none = pd.read_csv(r'')
oeder_none = dict(zip(order_none['order_id'].values,order_none['reorder_num'].values))

order_nlabel = pd.read_csv(r'C:\Users\csw\Desktop\user_pred_nlabel.csv')
eval_test_temp = modify_probability(eval_test,order_nlabel,c=0.92)
y_pred = get_result2(eval_test_temp)
y_pred = pd.merge(y_true[['order_id']],y_pred,on='order_id',how='left')
print('f1得分为：%f' % (instacart_grade(y_true,y_pred)))

