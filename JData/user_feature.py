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
def diff_of_day1(day1,day2):
    try:
        return (pd.Timestamp(day1)-pd.Timestamp(day2)).days
    except:
        return np.nan
def diff_of_day(day1, day2):
    d = {'1': 0, '2': 31, '3': 60, '4': 91}
    try:
        return (d[day1[6]] + int(day1[8:10])) - (d[day2[6]] + int(day2[8:10]))
    except:
        return np.nan


##########################读取数据##########################
#读取action原始数据
def get_actions(start_date, end_date):
    dump_path = r'E:\Data\cache\all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        action_2 = pd.read_csv(action_2_path)
        action_3 = pd.read_csv(action_3_path)
        action_4 = pd.read_csv(action_4_path)
        actions = pd.concat([action_2, action_3, action_4]) # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#读取action中cate==8的数据
def get_cate8(start_date, end_date):
    dump_path = r'E:\Data\cache\cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        action_2 = pd.read_csv(action_2_path)
        action_3 = pd.read_csv(action_3_path)
        action_4 = pd.read_csv(action_4_path)
        actions = pd.concat([action_2, action_3, action_4]) # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date) & (actions.cate == 8)]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
#读取action中cate!=8的数据
def get_other(start_date, end_date):
    dump_path = r'E:\Data\cache\cate_other_%s_%s.pkl' % (start_date, end_date)
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
    dump_path = r'E:\Data\cache\ID_%ddays_%s.pkl' % (i,end_date)
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




#########################特征统计#############################

#用户基本信息（年龄，性别，会员等级，注册日期）
def get_basic_user_feat(end_date):
    dump_path = r'E:\Data\cache\basic_user_%s.pkl' % (end_date)
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb+'))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        user['user_reg_tm'] = pd.DataFrame(user['user_reg_tm'].apply(lambda x:diff_of_day1(end_date,x)))
        pickle.dump(user, open(dump_path, 'wb+'))
    return user


#统计用户浏览cate8的时间
def get_user_action_tm(start_date, end_date):
    dump_path = r'E:\Data\cache\user_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions_tm = pickle.load(open(dump_path, 'rb+'))
    else:
        actions =  get_cate8(start_date, end_date)
        actions_tm = pd.DataFrame(actions.groupby('user_id')['time'].min().apply(lambda x:diff_of_day(end_date,x))).rename(columns={'time': 'first_tm'})
        actions_tm['last_tm'] = actions.groupby('user_id')['time'].max().apply(lambda x:diff_of_day(end_date,x))
        actions_tm['sep_tm'] = actions_tm.apply(lambda x:(x[0]-x[1]),axis=1)
        actions_tm.reset_index(inplace=True)
        pickle.dump(actions_tm,open(dump_path, 'wb+'))
    return actions_tm


#统计用户浏览商品的次序
def get_user_action_rank(start_date, end_date):
    dump_path = r'E:\Data\cache\action_rank_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions_rank = pickle.load(open(dump_path, 'rb+'))
    else:
        actions =  get_cate8(start_date, end_date)
        actions_rank = pd.DataFrame(actions.groupby(['user_id','sku_id'])['time'].max()).reset_index()
        actions_rank.sort_values(['user_id','sku_id','time'],ascending=False,inplace=True)
        actions_rank.index = actions_rank.user_id
        actions_rank['rank'] = range(actions_rank.shape[0])
        actions_rank['min_rank'] = actions_rank.groupby('user_id')['rank'].min()
        actions_rank['max_rank'] = actions_rank.groupby('user_id')['rank'].max()
        actions_rank['rank_sep'] = actions_rank['max_rank']-actions_rank['min_rank']
        actions_rank['rank'] = (actions_rank['rank']-actions_rank['min_rank'])/actions_rank['max_rank']
        del actions_rank['min_rank'], actions_rank['max_rank'], actions_rank['time']
        pickle.dump(actions_rank,open(dump_path, 'wb+'))
    return actions_rank

#用户转化率(将用户的多次购买合并为一次)
def get_user_conversion(start_date, end_date):
    feature = ['user_id', 'user_1_conversion', 'user_2_conversion', 'user_3_conversion',
               'user_5_conversion', 'user_6_conversion', 'user_action_conversion']
    dump_path = r'E:\Data\cache\user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
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
        actions.replace(np.inf,1,inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户行为特征（各种行为总次数）
def get_user_action_feat(start_date, end_date, i):
    dump_path = r'E:\Data\cache\user_action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[['user_id','type']]
        df = pd.get_dummies(actions['type'], prefix='%d_user_action' % i)
        df['%d_user_action' % i] = df.sum(axis=1)
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#商品基本信息（种类，品牌，三个属性）
def get_basic_product_feat():
    dump_path = r'E:\Data\cache\basic_product.pkl'
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
    dump_path = r'E:\Data\cache\comments_accumulate_%s.pkl' % (end_date)
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
    dump_path = r'E:\Data\cache\product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
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
    dump_path = r'E:\Data\cache\brand_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        brand = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        brand = pd.DataFrame(actions.groupby('brand')['cate'].sum()/8).rename(columns={'cate':'count_of_brand'})
        brand['count_buy_brand'] = actions[actions['type']==4].groupby('brand')['cate'].sum()/8
        brand['count_buy_brand'].fillna(0,inplace=True)
        brand['conversion_of_brand'] = brand['count_buy_brand']/brand['count_of_brand']
        brand.reset_index(inplace=True)
        pickle.dump(brand, open(dump_path, 'wb+'))
    return brand

#统计行为特征（各种行为总次数）
def get_action_feat(start_date, end_date, i):
    dump_path = r'E:\Data\cache\action_accumulate_%s_%s.pkl' % (start_date, end_date)
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

#统计行为的时间
def get_action_tm(start_date, end_date):
    dump_path = r'E:\Data\cache\action_tm_%s.pkl' % (end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        grouped = actions.groupby(['user_id','sku_id','type'])['time']
        actions_first_tm = grouped.min().apply(lambda x: diff_of_day(end_date, x))
        actions_last_tm = grouped.max().apply(lambda x: diff_of_day(end_date, x))
        actions_sep_tm = actions_last_tm - actions_first_tm
        actions_first_tm = actions_first_tm.unstack()
        actions_first_tm.columns = [('first_tm_of_' + str(name)) for name in actions_first_tm.columns]
        actions_last_tm = actions_last_tm.unstack()
        actions_last_tm.columns = [('last_tm_of_' + str(name)) for name in actions_last_tm.columns]
        actions_sep_tm = actions_sep_tm.unstack()
        actions_sep_tm.columns = [('sep_of_' + str(name)) for name in actions_sep_tm.columns]
        actions = pd.concat([actions_first_tm, actions_last_tm, actions_sep_tm],axis=1)
        actions.fillna(-10000,inplace=True)
        actions.reset_index(inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions




#统计行为次数（按时间进行衰减）
def get_accumulate_action_feat(start_date, end_date):
    dump_path = r'E:\Data\cache\action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        #近期行为按时间衰减
        actions['weights'] = actions['time'].map(lambda x: diff_of_day(end_date,x))
        #actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights'] = actions['weights'].map(lambda x: math.exp(-x))
        print (actions.head(10))
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['time']
        del actions['datetime']
        del actions['weights']
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#获取样本标签(将用户的多次购买合并为一次)
def get_labels(start_date, end_date):
    dump_path = r'E:\Data\cache\labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#构建测试集
def make_test_set(train_start_date, train_end_date):
    dump_path = r'E:\Data\cache\test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = "2016-02-01"
        user = get_basic_user_feat(train_end_date)
        product = get_basic_product_feat()
        user_acc = get_user_conversion(start_date, train_end_date)
        product_acc = get_accumulate_product_feat(start_date, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        #labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_date = start_date.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_date, train_end_date, i)
            else:
                actions = pd.merge(actions, get_action_feat(start_date, train_end_date, i), how='left',
                                   on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        #actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(-1)
        actions = actions[actions['cate'] == 8]
        pickle.dump(actions, open(dump_path, 'wb+'))

    return actions

#构建训练集
def make_train_set(train_end_date, test_start_date, test_end_date, days=30):
    start_date = "2016-01-31"
    user = get_basic_user_feat(train_end_date)                                  # 用户基本信息
    user_conversion = get_user_conversion(start_date, train_end_date)           # 用户转化率
    user_tm = get_user_action_tm(start_date, train_end_date)                    # 用户浏览cate8时间
    #product = get_basic_product_feat()                                          # 商品基本信息
    #comment = get_comments(train_end_date)                                      # 商品评论信息
    #product_conversion = get_product_conversion(start_date, train_end_date)     # 商品转化率
    #action_tm = get_action_tm(start_date, train_end_date)                       # 商品行为时间
    #action_rank = get_user_action_rank(start_date, train_end_date)              # 用户浏览物品次序
    #brand = get_brand_feat(start_date,train_end_date)                           # 品牌特征
    labels = get_labels(test_start_date, test_end_date)                         # 获取label

    # generate 时间窗口
    # actions = get_accumulate_action_feat(train_start_date, train_end_date)
    actions = None                                                              # 交互行为特征
    user_action = None                                                          # 用户行为特征
    for i in (1, 2, 3, 5, 7, 10, 15, 21, 30, 60):
        start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_date = start_date.strftime('%Y-%m-%d')
        if actions is None:
            #actions = get_action_feat(start_date, train_end_date, i)
            user_action = get_user_action_feat(start_date, train_end_date, i)
        else:
            if (i<6):

                user_action = pd.merge(get_user_action_feat(start_date, train_end_date, i), user_action, how='left',
                                   on=['user_id'])
            else:

                user_action = pd.merge(user_action, get_user_action_feat(start_date, train_end_date, i), how='left',
                                   on=['user_id'])

    user = pd.merge(user, user_conversion, how='left', on='user_id')
    user = pd.merge(user, user_tm, how='left', on='user_id')
    user = pd.merge(user, user_action, how='left', on='user_id')
    user = pd.merge(user, labels[['user_id','label']], how='left', on='user_id')
    user['sku_id'] = 1
    user = user.fillna(-1)

    return  user, labels



user_data0, y_true0 = make_train_set('2016-04-16', '2016-04-16', '2016-04-21')
print(0)
user_data1, y_true1 = make_train_set('2016-04-11', '2016-04-11', '2016-04-16')
print(1)
user_data2, y_true2 = make_train_set('2016-04-06', '2016-04-06', '2016-04-11')
print(2)
user_data3, y_true3 = make_train_set('2016-04-01', '2016-04-01', '2016-04-06')
print(3)





