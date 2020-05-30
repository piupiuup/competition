#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from tqdm import tqdm

action_2_path = r"C:\Users\csw\Desktop\python\JData\data\JData_action_201602.csv"
action_3_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Action_201603.csv"
action_4_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Action_201604.csv"
comment_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Comment.csv"
product_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Product.csv"
user_path = r"C:\Users\csw\Desktop\python\JData\data\JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]

#将年龄转化为int型
def convert_age(age_str):
    if age_str == u'-1':
        return 1
    elif age_str == u'15岁以下':
        return 0
    elif age_str == u'16-25岁':
        return 3
    elif age_str == u'26-35岁':
        return 4
    elif age_str == u'36-45岁':
        return 5
    elif age_str == u'46-55岁':
        return 2
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


#计算相差的天数
def diff_of_days1(day1,day2):
    try:
        return (pd.Timestamp(day1)-pd.Timestamp(day2)).days
    except:
        return np.nan
def diff_of_days(day1, day2):
    d = {'1': 0, '2': 31, '3': 60, '4': 91}
    try:
        return (d[day1[6]] + int(day1[8:10])) - (d[day2[6]] + int(day2[8:10]))
    except:
        return np.nan
# 相差的小时数
def diff_of_hours(day1, day2):
    d = {'1': 0, '2': 31, '3': 60, '4': 91}
    try:
        days = (d[day1[6]] + int(day1[8:10])) - (d[day2[6]] + int(day2[8:10]))
        hours = int(day2[11:13])
        return (days * 24 - hours)
    except:
        return np.nan
# 相差的分钟数
def diff_of_minutes(day1, day2):
    d = {'1': 0, '2': 31, '3': 60, '4': 91}
    try:
        days = (d[day1[6]] + int(day1[8:10])) - (d[day2[6]] + int(day2[8:10]))
        try:
            minutes1 = int(day1[11:13]) * 60 + int(day1[14:16])
        except:
            minutes1 = 0
        try:
            minutes2 = int(day2[11:13]) * 60 + int(day2[14:16])
        except:
            minutes2 = 0
        return (days * 1440 - minutes2 + minutes1)
    except:
        return np.nan

#日期的加减
def date_add_days(start_date, days):
    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

#循环提取特征
def get_cycle(get_function, end_date, key=['user_id'], n=6):
    '''
    get_function： 为提取特征所需要的函数
    end_date： 为提取特征的截止日期
    key：  为不同日期区间提取特征merge在一起是用到的键
    '''
    dump_path = r'F:\cache_porduct\_%s_%s.pkl' % (get_function.__name__, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = None
        print(get_function.__name__,end='   ')
        for i in tqdm((1, 2, 3, 5, 7, 10, 15, 21, 30, 60)):
            start_date = date_add_days(end_date, -i)
            if actions is None:
                actions = get_function(start_date, end_date, i)
            else:
                actions = pd.merge(get_function(start_date, end_date, i), actions, on=key, how='left')
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
# 提取用户id
def get_user_id(end_date, i):
    dump_path = r'F:\cache\user_id_%s_%ddays.pkl' % (end_date,i)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_date = start_date.strftime('%Y-%m-%d')
        actions = get_cate8(start_date, end_date)  # type: pd.DataFrame
        actions = actions[['user_id']].drop_duplicates()
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


##########################读取数据##########################
#读取action原始数据
def get_actions(start_date=None, end_date=None):
    if start_date==None:
        dump_path = r'F:\cache\all_action.pkl'
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb+'))
        else:
            action_2 = pd.read_csv(action_2_path)
            action_3 = pd.read_csv(action_3_path)
            action_4 = pd.read_csv(action_4_path)
            actions = pd.concat([action_2, action_3, action_4])
            actions.sort_values(['user_id','time'], ascending=True, inplace=True)
            pickle.dump(actions, open(dump_path, 'wb+'))
        return actions
    else:
        actions = get_actions()
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        return actions
#读取action中cate==8的数据
def get_cate8(start_date, end_date):
    dump_path = r'F:\cache\cate8.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    else:
        actions = get_actions()
        actions = actions[actions['cate'] == 8]
        pickle.dump(actions, open(dump_path, 'wb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions
#读取cate8中回头看到的数据
def get_back_cate8():
    dump_path = r'F:\cache\back_cate8.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8('2016-01-31','2016-04-16')
        actions = actions[['user_id','sku_id','time']]
        sku_id = actions['sku_id'].values
        flag = list(sku_id[:-1]!=sku_id[1:])
        flag.insert(0,True)
        actions = actions[flag]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#读取action中cate!=8的数据
def get_other(start_date, end_date):
    dump_path = r'F:\cache\cate_other.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    else:
        actions = get_actions()
        actions = actions[actions.cate != 8]
        pickle.dump(actions, open(dump_path, 'wb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions

#读取计算非cate8转化率所使用的购买前一天的数据
def get_conversion(start_date, end_date):
    dump_path = r'F:\cache\conversion_actions.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    else:
        actions = get_other('2016-01-31', '2016-04-16')
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        buy = actions[actions['type']==4].drop_duplicates(['user_id','cate'],keep='first')
        buy.rename(columns={'date':'first_buy_time'},inplace=True)
        actions = pd.merge(actions,buy[['user_id','cate','first_buy_time']],on=['user_id','cate'],how='left')
        actions['first_buy_time'] = actions['first_buy_time'].fillna('2016-04-17')
        actions = actions[actions['date']<actions['first_buy_time']]
        actions = actions[['user_id', 'sku_id', 'time', 'model_id', 'type', 'cate', 'brand']]
        buy = buy[['user_id', 'sku_id', 'time', 'model_id', 'type', 'cate', 'brand']]
        actions = pd.concat([actions,buy]).sort_values(['user_id','time'])
        pickle.dump(actions, open(dump_path, 'wb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions

#读取计算cate8转化率所使用的购买前一天的数据
def get_cate8_conversion(start_date, end_date):
    dump_path = r'F:\cache\cate8_conversion_actions.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    else:
        actions = get_cate8('2016-01-31', '2016-04-16')
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        buy = actions[actions['type']==4].drop_duplicates(['user_id','cate'],keep='first')
        buy.rename(columns={'date':'first_buy_time'},inplace=True)
        actions = pd.merge(actions,buy[['user_id','cate','first_buy_time']],on=['user_id','cate'],how='left')
        actions['first_buy_time'] = actions['first_buy_time'].fillna('2016-04-17')
        actions = actions[actions['date']<actions['first_buy_time']]
        actions = actions[['user_id', 'sku_id', 'time', 'model_id', 'type', 'cate', 'brand']]
        buy = buy[['user_id', 'sku_id', 'time', 'model_id', 'type', 'cate', 'brand']]
        actions = pd.concat([actions,buy]).sort_values(['user_id','time'])
        pickle.dump(actions, open(dump_path, 'wb+'))
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions

#读取前n天cate8相关的交互ID
def get_ID(end_date,i):
    dump_path = r'F:\cache\ID_%ddays_%s.pkl' % (i,end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_date = start_date.strftime('%Y-%m-%d')
        actions = get_actions('2016-01-31', end_date)  # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date) & (actions.cate == 8)]
        actions = actions[['user_id','sku_id']].copy().drop_duplicates()
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#获取样本标签(将用户的多次购买合并为一次)
def get_labels(start_date, end_date):
    dump_path = r'F:\cache\labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions[['user_id', 'sku_id']].drop_duplicates()
        actions['label'] = 1
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
# 提取交互的id
def get_action_id(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_id_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','sku_id']].drop_duplicates()
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

##############################################################################
'''''''''''''''''''''''''''''''''''商品特征'''''''''''''''''''''''''''''''''''
##############################################################################
#商品基本信息（种类，品牌，三个属性）
def get_basic_product_feat():
    dump_path = r'F:\cache_porduct\basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, 'rb+'))
    else:
        product = pd.read_csv(product_path)
        product['a1'] = product['a1'].map({-1: 1, 1: 4, 2: 2, 3: 3})
        product['a2'] = product['a2'].map({-1: 1, 1: 3, 2: 2})
        product['a3'] = product['a3'].map({-1: 1, 1: 3, 2: 2})
        del product['cate']
        pickle.dump(product, open(dump_path, 'wb+'))
    return product

# 商品comment信息（评论数，差评率）
def get_comments(end_date):
    dump_path = r'F:\cache_porduct\comments_accumulate_%s.pkl' % (end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path, 'rb+'))
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[comments.dt == comment_date_begin]
        del comments['dt']
        pickle.dump(comments, open(dump_path, 'wb+'))
    return comments

# 商品type点击转化率(点击个数)
def get_product_action_conversion1(end_date, n_days):
    feature = ['sku_id', 'product_1_convern', 'product_2_convern', 'product_3_convern',
               'product_5_convern', 'product_6_convern', 'product_action_convern']
    dump_path = r'F:\cache_porduct\product_action_conversion1_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8_conversion(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_1_convern'] = actions['action_4'] / actions['action_1']
        actions['product_2_convern'] = actions['action_4'] / actions['action_2']
        actions['product_3_convern'] = actions['action_4'] / actions['action_3']
        actions['product_5_convern'] = actions['action_4'] / actions['action_5']
        actions['product_6_convern'] = actions['action_4'] / actions['action_6']
        actions['product_action_convern'] = actions[feature[1:6]].sum(axis = 1)
        actions = actions[feature]
        actions.replace(np.inf, 1, inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
# 商品type点击转化率(是否点击)
def get_product_action_conversion2(end_date, n_days):
    dump_path = r'F:\cache_porduct\product_action_conversion2_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8_conversion(start_date, end_date)
        labels = get_labels(start_date, end_date)
        actions = actions.drop_duplicates(['user_id', 'sku_id', 'type'])
        actions = pd.merge(actions, labels, on=['user_id', 'sku_id'], how='left').fillna(0)
        type1 = actions[actions['type'] == 1].groupby('sku_id', as_index=False)['label'].agg(
            {'product_count_of_type1': 'count', 'product_buy_of_type1': 'sum'})
        type2 = actions[actions['type'] == 2].groupby('sku_id',as_index=False)['label'].agg(
            {'product_count_of_type2':'count','product_buy_of_type2':'sum'})
        type3 = actions[actions['type'] == 3].groupby('sku_id', as_index=False)['label'].agg(
            {'product_count_of_type3': 'count', 'product_buy_of_type3': 'sum'})
        type5 = actions[actions['type'] == 5].groupby('sku_id', as_index=False)['label'].agg(
            {'product_count_of_type5': 'count', 'product_buy_of_type5': 'sum'})
        type1['product_type1_conversion'] = type1['product_buy_of_type1'] / type1['product_count_of_type1']
        type2['product_type2_conversion'] = type2['product_buy_of_type2'] / type2['product_count_of_type2']
        type3['product_type3_conversion'] = type3['product_buy_of_type3'] / type3['product_count_of_type3']
        type5['product_type5_conversion'] = type5['product_buy_of_type5'] / type5['product_count_of_type5']
        actions = pd.merge(type1, type2,on='sku_id',how='left')
        actions = pd.merge(actions, type3, on='sku_id', how='left')
        actions = pd.merge(actions, type5, on='sku_id', how='left')
        actions = actions[['sku_id','product_type1_conversion','product_type2_conversion','product_type3_conversion','product_type5_conversion']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


# 品牌品牌转化率（按照是否点击算）
def get_brand_conversion1(end_date, n_days=60):
    dump_path = r'F:\cache_porduct\brand_conversion1_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        data = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        labels = get_labels(start_date, end_date)
        actions = actions.drop_duplicates(['user_id', 'sku_id'])
        actions = pd.merge(actions, labels, on=['user_id', 'sku_id'], how='left').fillna(0)
        data = actions.groupby('brand', as_index=False)['label'].agg({'count_buy_brand': 'sum', 'count_look_brand': 'count'})
        data['conversion_of_brand'] = data['count_buy_brand'] / data['count_look_brand']
        pickle.dump(data, open(dump_path, 'wb+'))
    return data
# 品牌品牌转化率（按照type类型算）
def get_brand_conversion2(end_date, n_days):
    dump_path = r'F:\cache_porduct\brand_conversion2_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8_conversion(start_date, end_date)
        labels = get_labels(start_date, end_date)
        actions = actions.drop_duplicates(['user_id', 'sku_id', 'type'])
        actions = pd.merge(actions, labels, on=['user_id', 'sku_id'], how='left').fillna(0)
        type1 = actions[actions['type'] == 1].groupby('brand', as_index=False)['label'].agg(
            {'product_brand_count_of_type1': 'count', 'product_brand_buy_of_type1': 'sum'})
        type2 = actions[actions['type'] == 2].groupby('brand',as_index=False)['label'].agg(
            {'product_brand_count_of_type2':'count','product_brand_buy_of_type2':'sum'})
        type3 = actions[actions['type'] == 3].groupby('brand', as_index=False)['label'].agg(
            {'product_brand_count_of_type3': 'count', 'product_brand_buy_of_type3': 'sum'})
        type5 = actions[actions['type'] == 5].groupby('brand', as_index=False)['label'].agg(
            {'product_brand_count_of_type5': 'count', 'product_brand_buy_of_type5': 'sum'})
        type1['product_brand_type1_conversion'] = type1['product_brand_buy_of_type1'] / type1['product_brand_count_of_type1']
        type2['product_brand_type2_conversion'] = type2['product_brand_buy_of_type2'] / type2['product_brand_count_of_type2']
        type3['product_brand_type3_conversion'] = type3['product_brand_buy_of_type3'] / type3['product_brand_count_of_type3']
        type5['product_brand_type5_conversion'] = type5['product_brand_buy_of_type5'] / type5['product_brand_count_of_type5']
        actions = pd.merge(type1, type2,on='brand',how='left')
        actions = pd.merge(actions, type3, on='brand', how='left')
        actions = pd.merge(actions, type5, on='brand', how='left')
        del actions['product_brand_buy_of_type2'],actions['product_brand_buy_of_type3'],actions['product_brand_buy_of_type5']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions



# 用户对品牌的关注度（除以最大值）
def get_user_brand_love2(end_date, n_days):
    dump_path = r'F:\cache_porduct\user_brand_love2_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        user_brand = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        count_of_user_brand = actions.groupby(['user_id','brand'], as_index=False)['cate'].agg(
            {'count_of_user_brand':'count'})
        max_of_user_brand = count_of_user_brand.groupby('user_id', as_index=False)['count_of_user_brand'].agg(
            {'max_of_user_brand': 'max'})
        user_brand = pd.merge(count_of_user_brand,max_of_user_brand,on='user_id',how='left')
        user_brand['love_of_brand'] = user_brand['count_of_user_brand']/user_brand['max_of_user_brand']
        user_brand = user_brand[['user_id','brand','love_of_brand']]
        pickle.dump(user_brand, open(dump_path, 'wb+'))
    return user_brand

# 用户对商品的关注度（除以最大值）
def get_user_product_love(end_date, n_days):
    dump_path = r'F:\cache_porduct\user_product_love_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        user_product = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        count_of_user_product = actions.groupby(['user_id','sku_id'], as_index=False)['cate'].agg(
            {'count_of_user_product':'count'})
        max_of_user_product = count_of_user_product.groupby('user_id', as_index=False)['count_of_user_product'].agg(
            {'max_of_user_product': 'max'})
        user_product = pd.merge(count_of_user_product,max_of_user_product,on='user_id',how='left')
        user_product['love_of_product'] = user_product['count_of_user_product']/user_product['max_of_user_product']
        user_product = user_product[['user_id','sku_id','love_of_product']]
        pickle.dump(user_product, open(dump_path, 'wb+'))
    return user_product

# 用户对商品属性a1的关注度（除以最大值）
def get_user_a1_love(end_date, n_days):
    dump_path = r'F:\cache_porduct\user_a1_love_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        user_product = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        product = get_basic_product_feat()
        actions = pd.merge(actions,product,on='sku_id',how='left')
        count_of_user_a1 = actions.groupby(['user_id','a1'], as_index=False)['cate'].agg(
            {'count_of_user_a1':'count'})
        max_of_user_a1 = count_of_user_a1.groupby('user_id', as_index=False)['count_of_user_a1'].agg(
            {'max_of_user_a1': 'max'})
        user_product = pd.merge(count_of_user_a1,max_of_user_a1,on='user_id',how='left')
        user_product['love_of_a1'] = user_product['count_of_user_a1']/user_product['max_of_user_a1']
        user_product = user_product[['user_id','a1','love_of_a1']]
        pickle.dump(user_product, open(dump_path, 'wb+'))
    return user_product

# 用户对商品属性a2的关注度（除以最大值）
def get_user_a2_love(end_date, n_days):
    dump_path = r'F:\cache_porduct\user_a2_love_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        user_product = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        product = get_basic_product_feat()
        actions = pd.merge(actions,product,on='sku_id',how='left')
        count_of_user_a2 = actions.groupby(['user_id','a2'], as_index=False)['cate'].agg(
            {'count_of_user_a2':'count'})
        max_of_user_a2 = count_of_user_a2.groupby('user_id', as_index=False)['count_of_user_a2'].agg(
            {'max_of_user_a2': 'max'})
        user_product = pd.merge(count_of_user_a2,max_of_user_a2,on='user_id',how='left')
        user_product['love_of_a2'] = user_product['count_of_user_a2']/user_product['max_of_user_a2']
        user_product = user_product[['user_id','a2','love_of_a2']]
        pickle.dump(user_product, open(dump_path, 'wb+'))
    return user_product

# 用户对商品属性a2的关注度（除以最大值）
def get_user_a3_love(end_date, n_days):
    dump_path = r'F:\cache_porduct\user_a3_love_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        user_product = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        product = get_basic_product_feat()
        actions = pd.merge(actions,product,on='sku_id',how='left')
        count_of_user_a3 = actions.groupby(['user_id','a3'], as_index=False)['cate'].agg(
            {'count_of_user_a3':'count'})
        max_of_user_a3 = count_of_user_a3.groupby('user_id', as_index=False)['count_of_user_a3'].agg(
            {'max_of_user_a3': 'max'})
        user_product = pd.merge(count_of_user_a3,max_of_user_a3,on='user_id',how='left')
        user_product['love_of_a3'] = user_product['count_of_user_a3']/user_product['max_of_user_a3']
        user_product = user_product[['user_id','a3','love_of_a3']]
        pickle.dump(user_product, open(dump_path, 'wb+'))
    return user_product

#统计用户浏览商品的次序
def get_action_rank(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_rank_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        action_rank = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'time']]
        first_action_rank = actions.drop_duplicates(['user_id', 'sku_id'], keep='first')
        last_action_rank = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        first_action_rank['diff_of_time'] = first_action_rank['time'].apply(lambda x:pd.Timestamp(x).value)
        last_action_rank['diff_of_time'] = last_action_rank['time'].apply(lambda x:pd.Timestamp(x).value)
        first_action_rank['first_rank'] = first_action_rank.groupby(['user_id'])['diff_of_time'].rank(ascending=True,method='first')
        last_action_rank['last_rank'] = last_action_rank.groupby(['user_id'])['diff_of_time'].rank(ascending=True,method='first')
        first_max_rank = first_action_rank.groupby('user_id', as_index=False)['first_rank'].agg({'first_max_rank':'max'})
        last_max_rank = last_action_rank.groupby('user_id', as_index=False)['last_rank'].agg({'last_max_rank':'max'})
        first_action_rank = pd.merge(first_action_rank, first_max_rank, on='user_id', how='left')
        last_action_rank = pd.merge(last_action_rank, last_max_rank, on='user_id', how='left')

        first_action_rank['first_rank'] = first_action_rank['first_rank']/first_action_rank['first_max_rank']
        last_action_rank['last_rank'] = last_action_rank['last_rank']/last_action_rank['last_max_rank']

        action_rank = pd.merge(first_action_rank,last_action_rank, on=['user_id', 'sku_id'], how='left')
        action_rank = action_rank[['user_id', 'sku_id', 'first_rank', 'last_rank']]
        pickle.dump(action_rank,open(dump_path, 'wb+'))
    return action_rank

#统计用户添加购物车的次序
def get_action_type2_rank(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_type2_rank_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        action_rank = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type']==2]
        actions = actions[['user_id', 'sku_id', 'time']]
        last_action_rank = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        last_action_rank['diff_of_time'] = last_action_rank['time'].apply(lambda x:pd.Timestamp(x).value)
        last_action_rank['last_rank'] = last_action_rank.groupby(['user_id'])['diff_of_time'].rank(ascending=True,method='first')
        last_max_rank = last_action_rank.groupby('user_id', as_index=False)['last_rank'].agg({'last_max_rank':'max'})
        last_action_rank = pd.merge(last_action_rank, last_max_rank, on='user_id', how='left')
        last_action_rank['last_type2_rank'] = last_action_rank['last_rank']/last_action_rank['last_max_rank']

        action_rank = last_action_rank[['user_id', 'sku_id', 'last_type2_rank']]
        pickle.dump(action_rank,open(dump_path, 'wb+'))
    return action_rank

#统计用户添加购物车的次序
def get_action_type6_rank(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_type6_rank_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        action_rank = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type']==6]
        actions = actions[['user_id', 'sku_id', 'time']]
        last_action_rank = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        last_action_rank['diff_of_time'] = last_action_rank['time'].apply(lambda x:pd.Timestamp(x).value)
        last_action_rank['last_rank'] = last_action_rank.groupby(['user_id'])['diff_of_time'].rank(ascending=True,method='first')
        last_max_rank = last_action_rank.groupby('user_id', as_index=False)['last_rank'].agg({'last_max_rank':'max'})
        last_action_rank = pd.merge(last_action_rank, last_max_rank, on='user_id', how='left')
        last_action_rank['last_type6_rank'] = last_action_rank['last_rank']/last_action_rank['last_max_rank']

        action_rank = last_action_rank[['user_id', 'sku_id', 'last_type6_rank']]
        pickle.dump(action_rank,open(dump_path, 'wb+'))
    return action_rank


#统计用户添加购物车的次序
def get_action_type3(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_type3_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        action_rank = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'].isin([2,3])]
        last_action = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        last_action['type3'] = last_action['type']==3
        last_action['type3'] = last_action['type3'].astype(np.int)
        action_rank = last_action[['user_id', 'sku_id', 'type3']]
        pickle.dump(action_rank,open(dump_path, 'wb+'))
    return action_rank

#统计交互type1次数（累计）
def get_action_type1_acc(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_type1_acc_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent = 0.59
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type']==1]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x:diff_of_days(end_date,x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x:expoent**x)
        actions = actions.groupby(['user_id','sku_id'],as_index=False)['diff_of_days'].agg({'action_type1_acc':'sum'})
        max_active_type1 = actions.groupby('user_id', as_index=False)['action_type1_acc'].agg(
            {'max_active_type1': 'max'})
        actions = pd.merge(actions, max_active_type1, on='user_id', how='left')
        actions['action_type1_acc'] = actions['action_type1_acc'] / actions['max_active_type1']
        actions = actions[['user_id', 'sku_id', 'action_type1_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计交互type1次数（累计）
def get_action_type6_acc(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_type6_acc_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.005
        expoent1 = 0.37
        expoent2 = 0.999
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type']==1]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x:diff_of_days(end_date,x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby(['user_id','sku_id'],as_index=False)['diff_of_days'].agg({'action_type6_acc':'sum'})
        max_active_type6 = actions.groupby('user_id', as_index=False)['action_type6_acc'].agg(
            {'max_active_type6': 'max'})
        actions = pd.merge(actions, max_active_type6, on='user_id', how='left')
        actions['action_type6_acc'] = actions['action_type6_acc'] / actions['max_active_type6']
        actions = actions[['user_id', 'sku_id', 'action_type6_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计交互type1和type6的百分比（累计）
def get_action_type1type6_acc(end_date, n_days):
    dump_path = r'F:\cache\action_type1type6_acc_%s_%ddays.pkl' % (end_date, -n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.2
        expoent1 = 0.6
        expoent2 = 0.97
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'].isin([1, 6])]
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby(['user_id', 'sku_id', 'type'])['diff_of_days'].sum().unstack().fillna(0)
        actions['action_type1type6_acc'] = actions[1] / (actions[6] + actions[1]) + 0.01
        actions.reset_index(inplace=True)
        max_type1type6 = actions.groupby('user_id', as_index=False)['action_type1type6_acc'].agg(
            {'max_type1type6': 'max'})
        actions = pd.merge(actions, max_type1type6, on='user_id', how='left')
        actions['action_type1type6_acc'] = actions['action_type1type6_acc'] / actions['max_type1type6']
        actions = actions[['user_id', 'sku_id', 'action_type1type6_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃分钟数(累计)
def get_action_active_minutes_acc(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_minutes_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.006
        expoent1 = 0.71
        expoent2 = 0.993
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] != 3]
        actions['minute'] = actions['time'].apply(lambda x: x[:16])
        actions = actions[['user_id','sku_id', 'minute']].drop_duplicates()
        actions['minute'] = actions['minute'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_minutes'] = actions['minute'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby(['user_id','sku_id'], as_index=False)['diff_of_minutes'].agg({'action_active_minutes_acc': 'sum'})
        max_active_minutes = actions.groupby('user_id', as_index=False)['action_active_minutes_acc'].agg(
            {'max_active_minutes': 'max'})
        actions = pd.merge(actions, max_active_minutes, on='user_id', how='left')
        actions['action_active_minutes_acc'] = actions['action_active_minutes_acc'] / actions['max_active_minutes']
        actions = actions[['user_id', 'sku_id', 'action_active_minutes_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 统计交互活跃小时数(累计)
def get_action_active_hours_acc(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_hours_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.005
        expoent1 = 0.2
        expoent2 = 0.999
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] != 3]
        actions['hour'] = actions['time'].apply(lambda x: x[:14])
        actions = actions[['user_id','sku_id', 'hour']].drop_duplicates()
        actions['hour'] = actions['hour'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_hours'] = actions['hour'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby(['user_id','sku_id'], as_index=False)['diff_of_hours'].agg({'action_active_hours_acc': 'sum'})
        max_active_hours = actions.groupby('user_id', as_index=False)['action_active_hours_acc'].agg(
            {'max_active_hours': 'max'})
        actions = pd.merge(actions,max_active_hours,on='user_id',how='left')
        actions['action_active_hours_acc'] = actions['action_active_hours_acc'] / actions['max_active_hours']
        actions = actions[['user_id', 'sku_id', 'action_active_hours_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃天数(累计)
def get_action_active_days_acc(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_days_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent = 0.995
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type']!=3]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions = actions[['user_id','sku_id','date']].drop_duplicates()
        actions['diff_of_days'] = actions['date'].apply(lambda x:diff_of_days(end_date,x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x:expoent**x)
        actions = actions.groupby(['user_id','sku_id'],as_index=False)['diff_of_days'].agg({'action_active_days_acc':'sum'})
        max_active_days = actions.groupby('user_id',as_index=False)['action_active_days_acc'].agg({'max_active_days':'max'})
        actions = pd.merge(actions,max_active_days,on='user_id',how='left')
        actions['action_active_days_acc'] = actions['action_active_days_acc']/actions['max_active_days']
        actions = actions[['user_id','sku_id','action_active_days_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 用户是否购买过同品牌非cate8的产品
def get_same_brand(end_date, n_days):
    dump_path = r'F:\cache_porduct\same_brand_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_other(start_date,end_date)
        actions = actions[actions['type']==4].drop_duplicates(['user_id','cate'])
        actions = actions.groupby(['user_id','brand'], as_index=False)['cate'].agg({'same_brand':'count'})
        actions = actions[['user_id', 'brand', 'same_brand']]
        pickle.dump(actions, open(dump_path, 'wb+'))

    return actions

#统计用户回头看的次数
def get_action_back(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_back_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_back_cate8()
        actions = actions[(actions['time']>start_date) & (actions['time']<end_date)]
        actions = actions.groupby(['user_id','sku_id'], as_index=False)['time'].agg({'action_back':'count'})
        max_active_back = actions.groupby('user_id',as_index=False)['action_back'].agg({'max_active_days':'max'})
        actions = pd.merge(actions,max_active_back,on='user_id',how='left')
        actions['action_back'] = actions['action_back']/actions['max_active_days']
        actions = actions[['user_id','sku_id','action_back']]

        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 交互最后一次浏览行为
def get_action_last_type(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_last_type_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions.drop_duplicates(['user_id','sku_id'], keep='last')
        actions['action_last_type'] = actions['type'].map({2:1,5:2,1:3,1:4,6:5,3:6})
        actions = actions[['user_id', 'sku_id', 'action_last_type']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 交互的model种类数
def get_action_nunique_model(end_date, n_days):
    dump_path = r'F:\cache_porduct\action_nunique_model_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','sku_id','model_id']].drop_duplicates()
        actions = actions.groupby(['user_id','sku_id'],as_index=False)['model_id'].agg({'action_nunique_model':'count'})
        max_nunique_model = actions.groupby('user_id', as_index=False)['action_nunique_model'].agg({'max_nunique_model': 'max'})
        actions = pd.merge(actions, max_nunique_model, on='user_id', how='left')
        actions['action_nunique_model'] = actions['action_nunique_model'] / actions['max_nunique_model']
        actions = actions[['user_id', 'sku_id', 'action_nunique_model']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#构建用户训练集和测试集
def make_product_train_set(train_end_date, test_start_date, test_end_date):
    dump_path = r'F:\cache_porduct\product_train_set_%s.pkl' % (train_end_date)
    if os.path.exists(dump_path) & 1:
        product = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = "2016-01-31"
        labels = get_labels(test_start_date, test_end_date)                                 # 获取标签
        action_id = get_action_id(train_end_date, 30)                                       # 提取交互id
        product_info = get_basic_product_feat()                                             # 提取商品基本信息
        comment = get_comments(train_end_date)                                              # 用户评论信息
        #product_action_conversion = get_product_action_conversion1(train_end_date,60)      # 商品点击次数转化率
        product_action_conversion = get_product_action_conversion2(train_end_date,50)       # 商品点击次数转化率
        #product_brand_conversion = get_brand_conversion1(train_end_date,60)                # 商品品牌转化率
        product_brand_conversion = get_brand_conversion2(train_end_date,50)                 # 商品品牌转化率
        user_brand_love = get_user_brand_love2(train_end_date,50)                           # 用户对品牌的喜爱程度
        action_rank = get_action_rank(train_end_date,50)                                    # 用户浏览物品的顺序
        # 添加购物车顺序
        # 收藏顺序
        user_product_love = get_user_product_love(train_end_date,50)                        # 用户对商品的喜爱程度
        user_a1_love = get_user_a1_love(train_end_date,50)                                  # 用户对物品属性a1的喜爱程度
        user_a2_love = get_user_a2_love(train_end_date,50)                                  # 用户对物品属性a2的喜爱程度
        user_a3_love = get_user_a3_love(train_end_date,50)                                  # 用户对物品属性a3的喜爱程度
        #action_type1_acc = get_action_type1_acc(train_end_date,50)                          # 交互type1次数
        #action_type6_acc = get_action_type6_acc(train_end_date,50)                          # 交互type6次数
        action_type1type6_acc = get_action_type1type6_acc(train_end_date,50)                          # 交互type1type6的比
        action_type2_rank = get_action_type2_rank(train_end_date,50)                        # 交互添加过购物车顺序
        action_type6_rank = get_action_type6_rank(train_end_date,50)                        # 交互添加过购物车顺序
        action_type3 = get_action_type3(train_end_date,50)                                  # 最终是否删除购物车
        action_active_minutes_acc = get_action_active_minutes_acc(train_end_date,n_days=50) # 交互活跃天数
        action_active_hours_acc = get_action_active_hours_acc(train_end_date,n_days=50)     # 交互活跃天数
        action_active_days_acc = get_action_active_days_acc(train_end_date,n_days=50)       # 交互活跃天数
        same_brand = get_same_brand(train_end_date,n_days=50)                               # 用户是否购买过同品牌其他产品
        action_back = get_action_back(train_end_date,n_days=50)                             # 用户回头看的次数
        action_last_type = get_action_last_type(train_end_date,n_days=50)                   # 交互最后一次type类型
        action_nunique_model = get_action_nunique_model(train_end_date,n_days=50)           # 交互的model_id 种类数

        product = pd.merge(action_id, labels,                   how='left', on=['user_id','sku_id'])
        product = pd.merge(product, product_info,               how='left', on='sku_id')
        product = pd.merge(product, comment,                    how='left', on='sku_id')
        product = pd.merge(product, product_action_conversion,  how='left', on='sku_id')
        product = pd.merge(product, product_brand_conversion,   how='left', on='brand')
        product = pd.merge(product, user_brand_love,            how='left', on=['user_id','brand'])
        product = pd.merge(product, same_brand,                 how='left', on=['user_id','brand'])
        product = pd.merge(product, action_rank,                how='left', on=['user_id','sku_id'])
        product = pd.merge(product, user_product_love,          how='left', on=['user_id','sku_id'])
        product = pd.merge(product, user_a1_love,               how='left', on=['user_id','a1'])
        product = pd.merge(product, user_a2_love,               how='left', on=['user_id','a2'])
        product = pd.merge(product, user_a3_love,               how='left', on=['user_id','a3'])
        #product = pd.merge(product, action_type1_acc,           how='left', on=['user_id','sku_id'])
        #product = pd.merge(product, action_type6_acc,           how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_type1type6_acc,      how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_type2_rank,          how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_type6_rank,          how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_type3,               how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_active_minutes_acc,  how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_active_hours_acc,    how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_active_days_acc,     how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_back,                how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_last_type,           how='left', on=['user_id','sku_id'])
        product = pd.merge(product, action_nunique_model,       how='left', on=['user_id','sku_id'])
        if train_end_date != '2016-04-16':
            product = product[product['user_id'].isin(product[product['label'] == 1]['user_id'].values)]


        product['label'] = product['label'].fillna(0)
        product = product.fillna(-1000000)
        pickle.dump(product, open(dump_path, 'wb+'))

    return  product




product1 = make_product_train_set('2016-04-11', '2016-04-11', '2016-04-16')
print(1)
product2 = make_product_train_set('2016-04-06', '2016-04-06', '2016-04-11')
print(2)
product3 = make_product_train_set('2016-04-01', '2016-04-01', '2016-04-06')
print(3)
product4 = make_product_train_set('2016-03-27', '2016-03-27', '2016-04-01')
print(4)
product5 = make_product_train_set('2016-03-22', '2016-03-22', '2016-03-27')
print(5)
product0 = make_product_train_set('2016-04-16', '2016-04-16', '2016-04-21')
print(0)