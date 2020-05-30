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
                "2016-03-21", "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]

#将年龄转化为int型
def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
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
        minutes = int(day2[11:13]) * 60 + int(day2[14:16])
        return (days * 1440 - minutes)
    except:
        return np.nan
#日期的加减
def date_add_days(start_date, days):
    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date
#循环提取特征
def get_cycle(get_function, end_date, key=['user_id']):
    '''
    get_function： 为提取特征所需要的函数
    end_date： 为提取特征的截止日期
    key：  为不同日期区间提取特征merge在一起是用到的键
    '''
    dump_path = r'F:\cache\_%s_%s.pkl' % (get_function.__name__, end_date)
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
                if (i < 6):
                    actions = pd.merge(get_function(start_date, end_date, i), actions, on=key, how='left')
                else:
                    actions = pd.merge(actions, get_function(start_date, end_date, i), on=key, how='left')
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

##########################读取数据##########################
#读取action原始数据
def get_actions(start_date=None, end_date=None):
    if start_date==None:
        dump_path = r'F:\cache\all_action_2016-01-31_2016-04-16.pkl'
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
        dump_path = r'F:\cache\all_action_%s_%s.pkl' % (start_date, end_date)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb+'))
        else:
            actions = get_actions()
            actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
            pickle.dump(actions, open(dump_path, 'wb+'))
        return actions
#读取action中cate==8的数据
def get_cate8(start_date, end_date):
    dump_path = r'F:\cache\cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#读取cate8中回头看到的数据
def get_back_cate8():
    dump_path = r'F:\cache\back_cate8.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8('2016-01-31','2016-04-16')
        actions = actions[['user_id','sku_id','time']]
        time = actions['sku_id'].values
        flag = list(time[:-1]!=time[1:])
        flag.insert(0,True)
        actions = actions[flag]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#读取action中cate!=8的数据
def get_other(start_date, end_date):
    dump_path = r'F:\cache\cate_other_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        action_2 = pd.read_csv(action_2_path)
        action_3 = pd.read_csv(action_3_path)
        action_4 = pd.read_csv(action_4_path)
        actions = pd.concat([action_2, action_3, action_4]) # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date) & (actions.cate != 8)]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#读取前n天cate8相关的数据
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




#################################################################
'''''''''''''''''''''''''''特征统计'''''''''''''''''''''''''''''
#################################################################

###############################用户特征##################################
#用户基本信息（年龄，性别，会员等级，注册日期）
def get_basic_user_feat(end_date):
    dump_path = r'F:\cache\basic_user_%s.pkl' % (end_date)
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb+'))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        user['user_reg_tm'] = user['user_reg_tm'].apply(lambda x:diff_of_days1(end_date,x))
        pickle.dump(user, open(dump_path, 'wb+'))
    return user


#统计用户浏览cate8的时间
def get_user_tm(start_date,train_end_date):
    dump_path = r'F:\cache\user_tm_%s_%s.pkl' % (start_date, train_end_date)
    if os.path.exists(dump_path):
        user_tm = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, train_end_date)
        user_first_tm = actions.drop_duplicates('user_id',keep='first')
        user_last_tm = actions.drop_duplicates('user_id', keep='last')
        user_first_tm['user_first_tm'] = user_first_tm['time'].map(lambda x: diff_of_minutes(train_end_date, x))
        user_last_tm['user_last_tm'] = user_last_tm['time'].map(lambda x: diff_of_minutes(train_end_date, x))
        user_tm = pd.merge(user_first_tm, user_last_tm, on='user_id', how='left')
        user_tm['user_sep_tm'] = user_tm['user_first_tm'] - user_tm['user_last_tm']
        user_tm = user_tm[['user_id', 'user_last_tm', 'user_first_tm', 'user_sep_tm']]
        pickle.dump(user_tm, open(dump_path, 'wb+'))
    return user_tm


#统计用户浏览cate8的时间
def get_user_type_tm(start_date,train_end_date):
    dump_path = r'F:\cache\user_type_tm_%s_%s.pkl' % (start_date, train_end_date)
    if os.path.exists(dump_path):
        user_type_last_tm = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, train_end_date)
        user_type_last_tm = actions.drop_duplicates(['user_id','type'], keep='last')
        user_type_last_tm['action_first_tm'] = user_type_last_tm['time'].map(lambda x: diff_of_minutes(train_end_date, x))
        user_type_last_tm = user_type_last_tm[['user_id','type','action_first_tm']]
        user_type_last_tm.set_index(['user_id','type'],inplace=True)
        user_type_last_tm = user_type_last_tm.unstack()
        user_type_last_tm.columns = ['action_first_tm_1','action_first_tm_2','action_first_tm_3',
                                       'action_first_tm_4','action_first_tm_5','action_first_tm_6']
        user_type_last_tm.reset_index(inplace=True)
        pickle.dump(user_type_last_tm, open(dump_path, 'wb+'))
    return user_type_last_tm


#用户转化率(将用户的多次购买合并为一次)
def get_user_conversion(start_date, end_date):
    feature = ['user_id', 'user_1_conversion', 'user_2_conversion', 'user_3_conversion',
               'user_5_conversion', 'user_6_conversion', 'user_action_conversion','action_4']
    dump_path = r'F:\cache\user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_other(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_1_conversion'] = actions['action_4'] / actions['action_1']
        actions['user_2_conversion'] = actions['action_4'] / actions['action_2']
        actions['user_3_conversion'] = actions['action_4'] / actions['action_3']
        actions['user_5_conversion'] = actions['action_4'] / actions['action_5']
        actions['user_6_conversion'] = actions['action_4'] / actions['action_6']
        actions['user_action_conversion'] = actions[feature[1:6]].sum(axis=1)
        actions = actions[feature]
        actions.rename(columns={'action_4':'other_4'},inplace=True)
        actions.replace(np.inf,1,inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户cate8行为特征（各种行为总次数）
def get_user_action_feat(start_date, end_date, i):
    dump_path = r'F:\cache\user_action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','type']]
        df = pd.get_dummies(actions['type'], prefix='%d_user_action' % i)
        df['%d_user_action' % i] = df.sum(axis=1)
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['%d_percent_user_action_1' % i] = actions['%d_user_action_1' % i] / actions['%d_user_action' % i]
        actions['%d_percent_user_action_6' % i] = actions['%d_user_action_6' % i] / actions['%d_user_action' % i]
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户所有行为特征（各种行为总次数）
def get_user_all_action_feat(start_date, end_date, i):
    dump_path = r'F:\cache\user_all_action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id','type']]
        df = pd.get_dummies(actions['type'], prefix='%d_user_all_action' % i)
        df['%d_user_all_action' % i] = df.sum(axis=1)
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['%d_percent_user_all_action_1' % i] = actions['%d_user_all_action_1' % i] / actions['%d_user_all_action' % i]
        actions['%d_percent_user_all_action_6' % i] = actions['%d_user_all_action_6' % i] / actions['%d_user_all_action' % i]
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户非cate8行为特征（各种行为总次数）
def get_user_other_feat(start_date, end_date, i):
    dump_path = r'F:\cache\user_other_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_other(start_date, end_date)
        n_cate = actions[['user_id','cate']].drop_duplicates()
        n_cate = n_cate.groupby('user_id',as_index=False).count()
        n_cate.rename(columns={'cate':'%d_n_cate' % i},inplace=True)

        n_product = actions[['user_id','sku_id']].drop_duplicates()
        n_product = n_product.groupby('user_id', as_index=False).count()
        n_product.rename(columns={'sku_id': '%d_n_product' % i}, inplace=True)

        actions = actions[['user_id','type']]
        df = pd.get_dummies(actions['type'], prefix='%d_user_other' % i)
        df['%d_user_other' % i] = df.sum(axis=1)
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions = pd.merge(actions, n_cate, on='user_id',how='left')
        actions = pd.merge(actions, n_product, on='user_id', how='left')
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#用户行为种类（type）对用的商品个数
def get_user_type_sku(start_date, end_date, i):
    dump_path = r'F:\cache\user_type_n_sku_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_type = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','sku_id','type']].drop_duplicates()
        user_type = actions.groupby(['user_id','type']).count().unstack()
        user_type.columns = list(range(1,7))
        user_type = user_type.add_prefix('%d_user_type_n_sku_' % i)
        user_type.reset_index(inplace=True)
        pickle.dump(user_type, open(dump_path, 'wb+'))
    return user_type

#用户行为种类（type）对用的频数
def get_user_type_action(start_date, end_date, i):
    dump_path = r'F:\cache\user_type_n_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_type = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','sku_id','type']]
        user_type = actions.groupby(['user_id','type']).count().unstack()
        user_type.columns = list(range(1,7))
        user_type = user_type.add_prefix('%d_user_type_n_action_' % i)
        user_type.reset_index(inplace=True)
        pickle.dump(user_type, open(dump_path, 'wb+'))
    return user_type

#用户近期model数
def get_user_n_model(start_date, end_date, i):
    dump_path = r'F:\cache\user_n_model_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_type = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','model_id']].drop_duplicates()
        user_type = actions.groupby(['user_id'],as_index=False)['model_id'].agg({'%d_n_model'% i:'count'})
        pickle.dump(user_type, open(dump_path, 'wb+'))
    return user_type


#统计用户活跃天数
def get_user_active_days(start_date, end_date, i):
    dump_path = r'F:\cache\user_action_days_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions = actions[['user_id','date']].drop_duplicates()
        actions = actions.groupby('user_id',as_index=False)['date'].count()
        actions.rename(columns = {'date': '%d_user_active_days' % i},inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃分钟数
def get_user_active_minutes(start_date, end_date, i):
    dump_path = r'F:\cache\user_action_minutes_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        actions['minute'] = actions['time'].apply(lambda x:x[:16])
        actions = actions[['user_id','minute']].drop_duplicates()
        actions = actions.groupby('user_id',as_index=False)['minute'].count()
        actions.rename(columns = {'minute': '%d_user_active_minutes' % i},inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#用户购买其他商品的最后时间
def get_user_last_buy_tm(start_date,end_date):
    dump_path = r'F:\cache\user_last_buy_tm_%s.pkl' % end_date
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_other(start_date, end_date)
        actions = actions[actions['type']==4]
        actions = actions.groupby('user_id',as_index=False)['time'].max()
        actions['user_last_buy_tm'] = actions['time'].apply(lambda x:diff_of_minutes(end_date, x))
        del actions['time']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#用户购买非cate8之前是否浏览过cate8
def ger_user_if_buy_other(start_date,end_date):
    dump_path = r'F:\cache\user_if_buy_other_tm_%s.pkl' % end_date
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date,end_date)
        actions_cate8_first_tm = actions[actions['cate']==8]
        actions_cate8_first_tm.drop_duplicates('user_id', keep='first', inplace=True)
        actions_cate8_first_tm['user_cate8_first_tm'] = actions_cate8_first_tm['time'].apply(lambda x:x[:10])
        actions_other_last_tm = actions[(actions['cate']!=8) & (actions['type']==4)]
        actions_other_last_tm.drop_duplicates('user_id',keep='last',inplace=True)
        actions_other_last_tm['buy_other_last_tm'] =  actions_other_last_tm['time'].apply(lambda x:x[:10])
        actions = pd.merge(actions_cate8_first_tm,actions_other_last_tm,on='user_id',how='left')
        actions['user_if_buy_other'] = actions.apply(lambda x:1 if diff_of_days(x['user_cate8_first_tm'],x['buy_other_last_tm'])>0 else 0,axis=1)
        actions = actions[['user_id','user_if_buy_other']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户回头看的次数
def get_action_back(start_date, end_date, i):
    dump_path = r'F:\cache\action_back_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_back_cate8()
        actions = actions[(actions['time']>start_date) & (actions['time']<end_date)]
        actions = actions.groupby(['user_id','sku_id'], as_index=False)['time'].count()
        actions.rename(columns={'time':'%d_action_back' %i},inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户行为次数（按时间进行衰减）
def get_user_accumulate(start_date, end_date):
    dump_path = r'F:\cache\user_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions['weights'] = actions['time'].map(lambda x: diff_of_days(end_date,x))
        actions['user_accumulate'] = actions['weights'].map(lambda x: math.exp(-x*0.5))
        actions = actions.groupby('user_id',as_index=False)['user_accumulate'].sum()
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


###############################商品特征##################################
#商品基本信息（种类，品牌，三个属性）
def get_basic_product_feat():
    dump_path = r'F:\cache\basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, 'rb+'))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'wb+'))
    return product

#商品comment信息（评论数，差评率）
def get_comments(end_date):
    dump_path = r'F:\cache\comments_accumulate_%s.pkl' % (end_date)
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

#商品点击转化率
def get_product_conversion(start_date, end_date):
    feature = ['sku_id', 'product_1_convern', 'product_2_convern', 'product_3_convern',
               'product_5_convern', 'product_6_convern', 'product_action_convern']
    dump_path = r'F:\cache\product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
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


#商品品牌特征
def get_brand_feat(start_date, end_date):
    dump_path = r'F:\cache\brand_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        brand = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        brand = (actions.groupby('brand',as_index=False)['cate'].count())
        brand.rename(columns = {'cate':'count_of_brand'},inplace=True)
        brand['count_buy_brand'] = actions[actions['type']==4].groupby('brand')['cate'].count()
        brand['count_buy_brand'].fillna(0,inplace=True)
        brand['conversion_of_brand'] = brand['count_buy_brand']/brand['count_of_brand']
        brand.reset_index(inplace=True)
        pickle.dump(brand, open(dump_path, 'wb+'))
    return brand

#用户对品牌的关注度
def get_user_brand(start_date, end_date) :
    dump_path = r'F:\cache\user_brand_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_brand = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        count_of_user_brand = actions.groupby(['user_id','brand'], as_index=False)['cate'].count()
        count_of_user_brand.rename(columns={'cate': 'count_of_user_brand'}, inplace=True)
        count_of_user = actions.groupby(['user_id'], as_index=False)['cate'].count()
        count_of_user.rename(columns={'cate': 'count_of_user'}, inplace=True)
        user_brand = pd.merge(count_of_user_brand,count_of_user,on='user_id',how='left')
        user_brand['love_of_brand'] = user_brand['count_of_user_brand']/user_brand['count_of_user']
        user_brand = user_brand[['user_id','brand','love_of_brand']]
        pickle.dump(user_brand, open(dump_path, 'wb+'))
    return user_brand

#################################交互特征##################################
#统计行为特征（各种行为总次数）
def get_action_feat(start_date, end_date, i):
    dump_path = r'F:\cache\action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%d_action' % i)
        df['%d_action' % i] = df.sum(axis=1)
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计行为的时间(天)
def get_action_tm(start_date, end_date):
    dump_path = r'F:\cache\action_tm_%s.pkl' % (end_date)
    if os.path.exists(dump_path):
        action_tm = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        action_last_tm = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        action_first_tm = actions.drop_duplicates(['user_id', 'sku_id'], keep='first')
        action_last_tm['action_last_tm'] = action_last_tm['time'].apply(lambda x: diff_of_minutes(end_date, x))
        action_first_tm['action_first_tm'] = action_first_tm['time'].apply(lambda x: diff_of_minutes(end_date, x))
        action_tm = pd.merge(action_last_tm,action_first_tm, on=['user_id', 'sku_id'], how='left')
        action_tm['action_sep_tm'] = action_tm['action_first_tm']-action_tm['action_last_tm']
        action_tm = action_tm[['user_id', 'sku_id', 'action_last_tm', 'action_first_tm', 'action_sep_tm']]
        pickle.dump(action_tm, open(dump_path, 'wb+'))
    return action_tm



#统计用户浏览商品的次序
def get_action_rank(start_date, end_date):
    dump_path = r'F:\cache\action_rank_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        action_rank = pickle.load(open(dump_path, 'rb+'))
    else:
        actions =  get_cate8(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'time']]
        first_action_rank = actions.drop_duplicates(['user_id', 'sku_id'], keep='first')
        last_action_rank = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        first_action_rank['first_rank'] = range(first_action_rank.shape[0])
        last_action_rank['last_rank'] = range(last_action_rank.shape[0])
        first_action_rank.index = first_action_rank.user_id
        last_action_rank.index = last_action_rank.user_id
        first_action_rank['first_min_rank'] = first_action_rank.groupby('user_id')['first_rank'].min()
        first_action_rank['first_max_rank'] = first_action_rank.groupby('user_id')['first_rank'].max()
        first_action_rank['rank_sep'] = first_action_rank['first_max_rank']-first_action_rank['first_min_rank']
        first_action_rank['first_rank'] = (first_action_rank['first_rank']-first_action_rank['first_min_rank'])/first_action_rank['rank_sep']

        last_action_rank['last_min_rank'] = last_action_rank.groupby('user_id')['last_rank'].min()
        last_action_rank['last_max_rank'] = last_action_rank.groupby('user_id')['last_rank'].max()
        last_action_rank['rank_sep'] = last_action_rank['last_max_rank'] - last_action_rank['last_min_rank']
        last_action_rank['last_rank'] = (last_action_rank['last_rank'] - last_action_rank['last_min_rank']) / last_action_rank['rank_sep']

        action_rank = pd.merge(first_action_rank,last_action_rank,on=['user_id', 'sku_id', 'rank_sep'])
        action_rank = action_rank[['user_id', 'sku_id', 'rank_sep', 'first_rank', 'last_rank']]
        pickle.dump(action_rank,open(dump_path, 'wb+'))
    return action_rank


#统计交互活跃天数
def get_active_days(start_date, end_date, i):
    dump_path = r'F:\cache\action_days_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions = actions[['user_id','sku_id','date']].drop_duplicates()
        actions = actions.groupby(['user_id','sku_id'],as_index=False)['date'].count()
        actions.rename(columns = {'date': '%d_active_days' % i},inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计交互活跃分钟数
def get_active_minutes(start_date, end_date, i):
    dump_path = r'F:\cache\action_minutes_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        actions['minute'] = actions['time'].apply(lambda x:x[:16])
        actions = actions[['user_id','sku_id','minute']].drop_duplicates()
        actions = actions.groupby(['user_id','sku_id'],as_index=False)['minute'].count()
        actions.rename(columns = {'minute': '%d_active_minutes' % i},inplace=True)
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

#由原来特征构造多次特征
def get_feat(actions):
    actions['percent_of_action'] = actions['60_action']/actions['60_user_action']
    #actions['no_buy_sep_tm'] = actions['user_last_buy_tm']-actions['user_last_tm']
    return actions


#构建训练集和测试集
def make_train_set(train_end_date, test_start_date, test_end_date):
    dump_path = r'F:\cache\train_set_%s_%s_%s.pkl' % (train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path) & 1:
        actions = pickle.load(open(dump_path, 'rb+'))
        labels = get_labels(test_start_date, test_end_date)
    else:
        start_date = "2016-01-31"
        user = get_basic_user_feat(train_end_date)                                  # 用户基本信息
        user_conversion = get_user_conversion(start_date, train_end_date)           # 用户转化率
        user_tm = get_user_tm(start_date, train_end_date)                           # 用户浏览cate8时间
        user_last_buy_tm = get_user_last_buy_tm(start_date,train_end_date)          # 用户最后购买非cate8的最后时间
        user_if_buy_other = ger_user_if_buy_other(start_date,train_end_date)        # 用户最后一次浏览cate8之后是否购买过其他商品
        #user_type_tm = get_user_type_tm(start_date, train_end_date)                 # 用户浏览（type）行为事件
        product = get_basic_product_feat()                                          # 商品基本信息
        comment = get_comments(train_end_date)                                      # 商品评论信息
        product_conversion = get_product_conversion(start_date, train_end_date)     # 商品转化率
        action_tm = get_action_tm(start_date, train_end_date)                       # 商品行为时间
        action_rank = get_action_rank(start_date, train_end_date)                   # 浏览商品次序
        #user_brand = get_user_brand(start_date, train_end_date)                     # 用户对品牌的关注度
        brand = get_brand_feat(start_date,train_end_date)                           # 品牌特征
        labels = get_labels(test_start_date, test_end_date)                         # 获取label

        date_ago = date_add_days(train_end_date,-60)
        #action_back = get_action_back(date_ago, train_end_date, 60)                 # 交互回头看次数

        user_action = get_cycle(get_user_action_feat, train_end_date, 'user_id')    # 用户行为次数特征
        #user_all_action = get_user_all_action_feat(start_date, train_end_date, i)  # 用户所有行为次数分析
        user_active_days = get_cycle(get_user_active_days, train_end_date, 'user_id')# 用户活跃天数
        #user_active_minutes = get_cycle(get_user_active_minutes, train_end_date, 'user_id')# 用户活跃分钟数
        user_other = get_cycle(get_user_other_feat, train_end_date, 'user_id')      # 用户近期浏览其他商品的次数
        #user_type_n_sku = get_cycle(get_user_type_sku, train_end_date, 'user_id')         # 用户行为种类对应的商品个数
        user_type_n_action = get_cycle(get_user_type_action, train_end_date, 'user_id')      # 用户行为种类对应的商品个数
        user_n_model = get_cycle(get_user_n_model, train_end_date, 'user_id')       # 用户的model种类数
        actions = get_cycle(get_action_feat, train_end_date, ['user_id', 'sku_id']) # 交互行为次数特征
        active_days = get_cycle(get_active_days, train_end_date, ['user_id', 'sku_id'])# 交互活跃天数
        #active_minutes = get_cycle(get_active_minutes, train_end_date, ['user_id', 'sku_id'])# 交互活跃分钟数


        user = pd.merge(user, user_conversion,    how='left', on='user_id')
        user = pd.merge(user, user_tm,            how='left', on='user_id')
        #user = pd.merge(user, user_type_tm,       how='left', on=['user_id'])
        user = pd.merge(user, user_last_buy_tm,   how='left', on='user_id')
        user = pd.merge(user, user_if_buy_other,  how='left', on='user_id')
        user = pd.merge(user, user_action,        how='left', on='user_id')
        user = pd.merge(user, user_active_days,   how='left', on='user_id')
        user = pd.merge(user, user_other,         how='left', on='user_id')
        #user = pd.merge(user, user_type_n_sku,    how='left', on='user_id')
        user = pd.merge(user, user_type_n_action, how='left', on='user_id')
        user = pd.merge(user, user_n_model,       how='left', on='user_id')
        #user = pd.merge(user, user_active_minutes,how='left', on='user_id')
        product = pd.merge(product, comment,            how='left', on='sku_id')
        product = pd.merge(product, product_conversion, how='left', on='sku_id')
        product = pd.merge(product, brand,              how='left', on='brand')
        actions = pd.merge(actions, user,               how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        # actions = pd.merge(actions, user_all_action,  how='left', on='user_id')
        actions = pd.merge(actions, action_rank,        how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, action_tm,          how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, active_days,        how='left', on=['user_id', 'sku_id'])
        #actions = pd.merge(actions, action_back,        how='left', on=['user_id', 'sku_id'])
        #actions = pd.merge(actions, active_minutes,     how='left', on=['user_id', 'sku_id'])
        #actions = pd.merge(actions, user_brand,         how='left', on=['user_id','brand'])
        actions = pd.merge(actions, labels,             how='left', on=['user_id', 'sku_id'])


        actions = get_feat(actions)
        actions['label'] = actions['label'].fillna(0)
        actions = actions.fillna(-1000000)
        pickle.dump(actions, open(dump_path, 'wb+'))

    return  actions, labels




data1, y_true1 = make_train_set('2016-04-11', '2016-04-11', '2016-04-16')
print(1)
data2, y_true2 = make_train_set('2016-04-06', '2016-04-06', '2016-04-11')
print(2)
data0, y_true0 = make_train_set('2016-04-16', '2016-04-16', '2016-04-21')
print(0)
data3, y_true3 = make_train_set('2016-04-01', '2016-04-01', '2016-04-06')
print(3)




#构建用户训练集和测试集
def make_user_train_set(train_end_date, test_start_date, test_end_date):
    dump_path = r'F:\cache\user_train_set_%s_%s_%s.pkl' % (train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path) & 1:
        actions = pickle.load(open(dump_path, 'rb+'))
        labels = get_labels(test_start_date, test_end_date)
    else:
        start_date = "2016-01-31"
        user = get_basic_user_feat(train_end_date)                                  # 用户基本信息
        user_conversion = get_user_conversion(start_date, train_end_date)           # 用户转化率
        user_tm = get_user_tm(start_date, train_end_date)                           # 用户浏览cate8时间
        user_last_buy_tm = get_user_last_buy_tm(start_date,train_end_date)          # 用户最后购买非cate8的最后时间
        user_if_buy_other = ger_user_if_buy_other(start_date,train_end_date)        # 用户最后一次浏览cate8之后是否购买过其他商品
        #user_type_tm = get_user_type_tm(start_date, train_end_date)                 # 用户浏览（type）行为事件
        labels = get_labels(test_start_date, test_end_date)                         # 获取label
        labels = labels[['user_id','label']].drop_duplicates()

        user_action = get_cycle(get_user_action_feat, train_end_date, 'user_id')    # 用户行为次数特征
        user_all_action = get_cycle(get_user_all_action_feat,train_end_date, 'user_id')# 用户所有行为次数分析
        user_active_days = get_cycle(get_user_active_days, train_end_date, 'user_id')# 用户活跃天数
        user_active_minutes = get_cycle(get_user_active_minutes, train_end_date, 'user_id')# 用户活跃分钟数
        user_other = get_cycle(get_user_other_feat, train_end_date, 'user_id')      # 用户近期浏览其他商品的次数
        user_type_n_sku = get_cycle(get_user_type_sku, train_end_date, 'user_id')  # 用户行为种类对应的商品个数
        user_type_n_action = get_cycle(get_user_type_action, train_end_date, 'user_id')# 用户行为种类对应的行为次数
        user_n_model = get_cycle(get_user_n_model, train_end_date, 'user_id')       # 用户的model种类数


        user = pd.merge(user_all_action, user,    how='left', on='user_id')
        user = pd.merge(user, user_conversion,    how='left', on='user_id')
        user = pd.merge(user, user_tm,            how='left', on='user_id')
        #user = pd.merge(user, user_type_tm,       how='left', on=['user_id'])
        user = pd.merge(user, user_last_buy_tm,   how='left', on='user_id')
        user = pd.merge(user, user_if_buy_other,  how='left', on='user_id')
        user = pd.merge(user, user_action,        how='left', on='user_id')
        user = pd.merge(user, user_active_days,   how='left', on='user_id')
        user = pd.merge(user, user_other,         how='left', on='user_id')
        user = pd.merge(user, user_type_n_sku,    how='left', on='user_id')
        user = pd.merge(user, user_type_n_action, how='left', on='user_id')
        user = pd.merge(user, user_n_model,       how='left', on='user_id')
        user = pd.merge(user, user_active_minutes,how='left', on='user_id')
        user = pd.merge(user, labels,             how='left', on='user_id')


        user['label'] = user['label'].fillna(0)
        actions = user.fillna(-1000000)
        pickle.dump(user, open(dump_path, 'wb+'))

    return  actions, labels




user1, y_user1 = make_user_train_set('2016-04-11', '2016-04-11', '2016-04-16')
print(1)
user2, y_user2 = make_user_train_set('2016-04-06', '2016-04-06', '2016-04-11')
print(2)
user0, y_user0 = make_user_train_set('2016-04-16', '2016-04-16', '2016-04-21')
print(0)
user3, y_user3 = make_user_train_set('2016-04-01', '2016-04-01', '2016-04-06')
print(3)
