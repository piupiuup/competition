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

action = get_actions()
user = user = pd.read_csv(user_path, encoding='gbk')
product = product = pd.read_csv(product_path)

#每天浏览cate的人数
action['date'] = action['time'].apply(lambda x:x[:10])
n_user_perday = action.groupby('date').map(lambda x:x['user_id'].nunique())

#第一次浏览的时间段
actions = get_cate8('2016-01-31','2016-04-16')
first_tm = actions.drop_duplicates('user_id',keep='first')
first_tm['hour'] = first_tm['time'].apply(lambda x:x[11:13])
labels = get_labels('2016-01-31', '2016-04-16')
labels = labels[['user_id','label']].drop_duplicates()
a = pd.merge(first_tm,labels,on='user_id',how='left').fillna(0)
a.groupby('hour')['label'].sum()/a.groupby('hour')['label'].count()


#用户最活跃的一个小时
actions = get_cate8('2016-01-31','2016-04-16')
actions['hour'] = actions['time'].apply(lambda x:x[11:13])
most_active_hour = actions.groupby(['user_id','hour'])['hour'].count().unstack()
most_active_hour['user_most_active_hour'] = most_active_hour.apply(lambda x:x.argmax(),axis=1)
most_active_hour.reset_index(inplace=True)
most_active_hour = most_active_hour[['user_id','user_most_active_hour']]
labels = get_labels('2016-01-31', '2016-04-16')
labels = labels[['user_id','label']].drop_duplicates()
a = pd.merge(most_active_hour,labels,on='user_id',how='left').fillna(0)
a.groupby('user_most_active_hour')['label'].sum()/a.groupby('user_most_active_hour')['label'].count()

#用户model_id转化率
actions = get_conversion('2016-01-31','2016-04-16')
actions = actions.drop_duplicates(['user_id','model_id'])
labels = get_labels('2016-01-31', '2016-04-16')
labels = labels[['user_id','label']].drop_duplicates()
a = pd.merge(actions,labels,on='user_id',how='left').fillna(0)
percent = pd.DataFrame(a.groupby('model_id')['label'].count())
percent.columns = ['n_user']
percent['n_buy'] = a.groupby('model_id')['label'].sum()
percent.fillna(0,inplace=True)
percent['percent'] = percent['n_buy']/percent['n_user']
percent.reset_index(inplace=True)
