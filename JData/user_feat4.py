import time
from datetime import datetime,timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from tqdm import tqdm
import scipy.stats as scs

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
    d = {'08': 0, '09': 31}
    try:
        return (d[day1[5:7]] + int(day1[8:10])) - (d[day2[6]] + int(day2[8:10]))
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
                actions = pd.merge(get_function(start_date, end_date, i), actions, on=key, how='outer')
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#循环提取特征，不累计
def get_cycle2(get_function, end_date, key=['user_id']):
    '''
    get_function： 为提取特征所需要的函数
    end_date： 为提取特征的截止日期
    key：  为不同日期区间提取特征merge在一起是用到的键
    '''
    dump_path = r'F:\cache\cycle2_%s_%s.pkl' % (get_function.__name__, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = None
        start_date = end_date
        print(get_function.__name__,end='   ')
        for i in tqdm((1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
            end_date = start_date
            start_date = date_add_days(end_date, -i)
            if actions is None:
                actions = get_function(start_date, end_date, i)
            else:
                actions = pd.merge(get_function(start_date, end_date, i), actions, on=key, how='outer')
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
# 基尼系数
def get_gini(arr):
    arr = list(arr)
    arr = sorted(arr)
    for i in reversed(range(len(arr))):
        arr[i] = sum(arr[:(i + 1)])
    gini = 1+1/len(arr)-2*sum(arr)/arr[-1]/len(arr)
    return gini
# 计算偏度
def skew(arr):
    return scs.skew(arr)
# 计算峰度
def kurt(arr):
    return scs.kurtosis(arr)

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
            actions.drop_duplicates(inplace=True)
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

#读取计算转化率所使用的非cate8购买前一天的数据
def get_other_conversion(start_date, end_date):
    dump_path = r'F:\cache\other_conversion_actions.pkl'
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

#读取计算转化率所使用的cate8购买前一天的数据
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
#提取用户id
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
# 获取样本标签(将用户的多次购买合并为一次)
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


#统计用户浏览cate8的时间 和 最后一次删除距离上次时间 和 第一次浏览的小时
def get_user_tm(start_date,end_date):
    dump_path = r'F:\cache\user_tm_%s_%s.pkl' % (start_date,end_date)
    if os.path.exists(dump_path):
        user_tm = pickle.load(open(dump_path, 'rb+'))
    else:
        #start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        user_last_tm1 = actions.drop_duplicates('user_id', keep='last')
        user_last_tm1['user_last_tm2'] = user_last_tm1['time'].map(lambda x: diff_of_minutes(end_date, x))
        actions = actions[actions['type']!=3]
        user_first_tm = actions.drop_duplicates('user_id',keep='first')
        user_first_tm['first_tm_of_hour'] = user_first_tm['time'].apply(lambda x: int(x[11:13]))
        user_last_tm2 = actions.drop_duplicates('user_id', keep='last')
        user_last_tm2['last_tm_of_hour'] = user_last_tm2['time'].apply(lambda x: int(x[11:13]))
        actions['time'] = actions['time'].apply(lambda x: diff_of_minutes(end_date, x))
        user_tm = actions.groupby('user_id',as_index=False)['time'].agg({
                                                          'user_mean_tm': 'mean','user_median_tm': 'median',
                                                          'user_last_tm': 'min', 'user_first_tm': 'max',
                                                          'user_kurt_tm': kurt, 'user_skew_tm': skew})
        user_tm = pd.merge(user_tm, user_first_tm, on='user_id', how='left')
        user_tm = pd.merge(user_tm, user_last_tm2, on='user_id', how='left')
        user_tm = pd.merge(user_tm, user_last_tm1, on='user_id', how='left')
        user_tm['user_sep_tm'] = user_tm['user_last_tm'] - user_tm['user_first_tm']
        user_tm['if_delect'] = user_tm['user_last_tm'] - user_tm['user_last_tm2']
        user_tm = user_tm[['user_id', 'user_last_tm', 'user_first_tm', 'user_sep_tm','if_delect','user_median_tm',
                'user_mean_tm', 'first_tm_of_hour','last_tm_of_hour','user_kurt_tm', 'user_skew_tm']]
        pickle.dump(user_tm, open(dump_path, 'wb+'))
    return user_tm

# 购买一个物品平均使用天数
def get_user_use_day(end_date,n_days):
    dump_path = r'F:\cache\user_use_day_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        use_day = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_other(start_date, end_date)
        first_tm = actions.drop_duplicates(['user_id','sku_id'],keep='first')
        first_tm.rename(columns={'time':'first_tm'}, inplace=True)
        buy_tm = actions[actions['type']==4].drop_duplicates(['user_id','sku_id'],keep='first')
        buy_tm.rename(columns={'time': 'buy_tm'}, inplace=True)
        use_day = pd.merge(buy_tm,first_tm,on=['user_id','sku_id'],how='left')
        use_day['user_use_day'] = use_day.apply(lambda x:diff_of_days(x['first_tm'],x['buy_tm']),axis=1)
        use_day = use_day.groupby('user_id',as_index=False)['user_use_day'].median()
        use_day = use_day[['user_id', 'user_use_day']]
        pickle.dump(use_day, open(dump_path, 'wb+'))
    return use_day

# 用户浏览时间最多的一天距离现在的时间
def get_user_maxday_tm(end_date,n_days):
    dump_path = r'F:\cache\user_maxday_tm_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['minute'] = actions['time'].apply(lambda x: x[:16])
        actions = actions.groupby(['user_id','date'])['minute'].count().unstack()
        actions['maxday'] = actions.apply(lambda x: diff_of_days(end_date,x.argmax()), axis=1)
        actions['maxminutes'] = actions.iloc[:,:-1].apply(lambda x: x.max(), axis=1)
        actions.reset_index(inplace=True)
        actions = actions[['user_id','maxday','maxminutes']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 用户第一天浏览的信息
def get_user_first_day(end_date,n_days):
    dump_path = r'F:\cache\user_user_first_day_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        first_day = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_actions(start_date, end_date)
        actions['user_first_day_hour'] = actions['time'].apply(lambda x: int(x[11:13]))
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        first_day = actions[actions['cate']==8].drop_duplicates('user_id',keep='first')
        first_day['user_first_day'] = first_day['date']
        actions = pd.merge(actions,first_day[['user_id','user_first_day']],on='user_id',how='left')
        actions = actions[actions['date']==actions['user_first_day']]
        first_day_cate8_sku = actions[actions['cate']==8].groupby('user_id',as_index=False)['user_id'].agg(
            {'user_first_day_cate8_sku':'count'})
        first_day_all_sku = actions.groupby('user_id', as_index=False)['user_id'].agg(
            {'user_first_day_all_sku': 'count'})
        first_day = pd.merge(first_day, first_day_cate8_sku, on='user_id', how='left')
        first_day = pd.merge(first_day, first_day_all_sku, on='user_id', how='left').fillna(0)
        first_day['user_first_day_percent_of_cate8'] = first_day['user_first_day_cate8_sku']/first_day['user_first_day_all_sku']
        first_day = first_day[['user_id','user_first_day_hour','user_first_day_cate8_sku','user_first_day_percent_of_cate8']]
        pickle.dump(first_day, open(dump_path, 'wb+'))
    return first_day


# 用户最活跃的时间段
def get_user_most_active_hour(end_date,n_days):
    dump_path = r'F:\cache\user_most_active_hour_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions['hour'] = actions['time'].apply(lambda x: x[11:13])
        most_active_hour = actions.groupby(['user_id', 'hour'])['hour'].count().unstack()
        most_active_hour['user_most_active_hour'] = most_active_hour.apply(lambda x: int(x.argmax()), axis=1)
        most_active_hour.reset_index(inplace=True)
        actions = most_active_hour[['user_id','user_most_active_hour']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 统计用户是否否买过cate8
def get_user_type4(start_date,train_end_date):
    dump_path = r'F:\cache\user_type4_%s_%s.pkl' % (start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, train_end_date)
        actions = actions[actions['type']==4]
        actions.drop_duplicates('user_id',inplace=True)
        actions['type4'] = 1
        actions = actions[['user_id','type4']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户购买过多少非cate8
def get_user_other_type4(end_date, n_days):
    dump_path = r'F:\cache\user_other_type4_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        actions = get_other(start_date, end_date)
        actions = actions[actions['type']==4]
        actions = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        actions = actions.groupby('user_id', as_index=False)['sku_id'].agg({'user_n_other_type4': 'count'})
        actions = actions[['user_id', 'user_n_other_type4']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#用户浏览过多少商品
def get_user_n_sku_acc(end_date, n_days):
    dump_path = r'F:\cache\user_n_sku_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent1 = 0.99
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        actions = get_cate8(start_date, end_date)
        actions = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        actions['diff_of_days'] = actions['time'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_n_sku_acc': 'sum'})
        actions['user_n_sku_acc'] = actions['user_n_sku_acc'] / sum_weight1
        actions = actions[['user_id', 'user_n_sku_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#用户购物车里剩余多少商品，以及cate8所占的比例
def get_user_type2_del_type3(end_date, n_days):
    dump_path = r'F:\cache\user_type2_del_type3_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        cate8 = get_cate8(start_date, end_date)
        cate8 = cate8[cate8['type'].isin([2,3])].drop_duplicates(['user_id','sku_id','type'],keep='last')
        cate8_type2 = cate8.groupby('user_id',as_index=False)['sku_id'].agg({'cate8_type2':'nunique'})
        cate8 = cate8.drop_duplicates(['user_id','sku_id'],keep='last')
        cate8_type3 = cate8[cate8['type']==3].groupby('user_id', as_index=False)['type'].agg({'cate8_type3': 'count'})
        cate8_type2_type3 = pd.merge(cate8_type2,cate8_type3,on='user_id',how='left')
        cate8_type2_type3['cate8_type2_del_type3'] = cate8_type2_type3['cate8_type2'] - cate8_type2_type3['cate8_type3']
        other = get_other(start_date, end_date)
        other = other[other['type'].isin([2,3])].drop_duplicates(['user_id','sku_id','type'],keep='last')
        other_type2 = other.groupby('user_id',as_index=False)['sku_id'].agg({'other_type2':'nunique'})
        other = other.drop_duplicates(['user_id','sku_id'],keep='last')
        other_type3 = other[other['type']==3].groupby('user_id', as_index=False)['type'].agg({'other_type3': 'count'})
        other_type2_type3 = pd.merge(other_type2, other_type3, on='user_id', how='left')
        other_type2_type3['other_type2_del_type3'] = other_type2_type3['other_type2'] - other_type2_type3['other_type3']
        actions = pd.merge(cate8_type2_type3,other_type2_type3,on='user_id',how='outer').fillna(0)
        actions['percent_remain_cate8_of_all'] = actions['cate8_type2_del_type3'] / (
            actions['other_type2_del_type3'] + actions['cate8_type2_del_type3'])
        actions['percent_type2_cate8_of_all'] = actions['cate8_type2'] / (
            actions['cate8_type2'] + actions['other_type2'])
        actions = actions[['user_id','cate8_type2_del_type3', 'percent_remain_cate8_of_all','percent_type2_cate8_of_all']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


# 统计用户model信息
def get_user_model(end_date, n_days):
    dump_path = r'F:\cache\user_model_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions['model_id'] = actions['model_id'].fillna(-1).astype('int')
        actions = actions.groupby(['user_id','model_id'])['model_id'].count().unstack()
        not_buy_model_id = [336,337,342,347,348,341,343,35,36,332,346,321,330,312,38,334,311,318,329,315,344,310,345]
        not_buy_model_id = [name for name in actions.columns if name in not_buy_model_id]
        feat_id = [0,-1,210,11,13,110,216,219,217,26,27,'not_buy_id','sum']
        actions['sum'] = actions.sum(axis=1)
        actions['not_buy_id'] = actions[not_buy_model_id].apply(lambda x:1 if x.sum()>0 else 0,axis=1)
        actions = actions[feat_id].fillna(0)
        actions = actions.add_prefix('user_model_')
        actions.reset_index(inplace=True)
        actions['percent_of_model_0'] = actions['user_model_0'] / actions['user_model_sum']
        actions['percent_of_model_216_and_217'] = (actions['user_model_216']+actions['user_model_217']) / actions['user_model_sum']
        actions['percent_of_model_26_and_27'] = (actions['user_model_26']+actions['user_model_27']) / actions['user_model_sum']
        del actions['user_model_sum'],actions['user_model_26'],actions['user_model_27']
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
        del actions['product_buy_of_type2'],actions['product_buy_of_type3'],actions['product_buy_of_type5']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions
# 统计用户浏览最多的商品个数
def get_user_max_sku(end_date, n_days):
    dump_path = r'F:\cache\user_max_sku_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        max_sku = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        product = pd.read_csv(product_path)
        product['a1'] = product['a1'].map({-1: 1, 1: 4, 2: 2, 3: 3})
        product['a2'] = product['a2'].map({-1: 1, 1: 3, 2: 2})
        product['a3'] = product['a3'].map({-1: 1, 1: 3, 2: 2})
        actions = pd.merge(actions,product[['sku_id','a1','a2','a3']],on='sku_id',how='left')
        user_type_a1 = actions.groupby(['user_id','a1'])['a1'].count().unstack().apply(lambda x:x.argmax(),axis=1)
        user_type_a1 = pd.DataFrame({'user_id': user_type_a1.index, 'user_type_a1': user_type_a1})
        user_type_a2 = actions.groupby(['user_id', 'a2'])['a2'].count().unstack().apply(lambda x: x.argmax(),axis=1)
        user_type_a2 = pd.DataFrame({'user_id': user_type_a2.index, 'user_type_a2': user_type_a2})
        user_type_a3 = actions.groupby(['user_id', 'a2'])['a3'].count().unstack().apply(lambda x: x.argmax(),axis=1)
        user_type_a3 = pd.DataFrame({'user_id': user_type_a3.index, 'user_type_a3': user_type_a3})
        max_sku = actions.groupby(['user_id', 'sku_id'], as_index=False)['sku_id'].agg({'n_actions': 'count'})
        max_sku.sort_values(['user_id', 'n_actions'], ascending=False, inplace=True)
        max_sku_id = max_sku.drop_duplicates('user_id', keep='last')
        product_action_conversion = get_product_action_conversion2(end_date, 60)
        max_sku_id.rename(columns={'n_actions': 'max_sku_id'}, inplace=True)
        max_sku_id = pd.merge(max_sku_id, product_action_conversion[['sku_id', 'product_type1_conversion']],on='sku_id',how='left')
        max_sku = max_sku.groupby('user_id', as_index=False)['n_actions'].agg({'user_max_sku': 'max',
                                                                               'user_std_sku': 'std',
                                                                               'user_gini_sku': get_gini,
                                                                               'user_mean_sku': 'mean',
                                                                               'user_skew_sku': skew,
                                                                               'user_kurt_sku': kurt})
        max_sku = pd.merge(max_sku, user_type_a1, on='user_id', how='left')
        max_sku = pd.merge(max_sku, user_type_a2, on='user_id', how='left')
        max_sku = pd.merge(max_sku, user_type_a3, on='user_id', how='left')
        max_sku = pd.merge(max_sku, max_sku_id[['user_id', 'product_type1_conversion']], on='user_id', how='left')
        pickle.dump(max_sku, open(dump_path, 'wb+'))
    return max_sku


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
# 统计用户浏览最多的品牌个数
def get_user_max_brand(end_date, n_days):
    dump_path = r'F:\cache\user_max_brand_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        max_brand = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        max_brand = actions.groupby(['user_id', 'brand'], as_index=False)['sku_id'].agg({'n_actions': 'count'})
        max_brand.sort_values(['user_id', 'n_actions'], ascending=False, inplace=True)
        max_brand_id = max_brand.drop_duplicates('user_id', keep='last')
        product_brand_conversion = get_brand_conversion1(end_date, 60)
        max_brand_id.rename(columns={'n_actions': 'max_sku_id'}, inplace=True)
        max_brand_id = pd.merge(max_brand_id, product_brand_conversion[['brand', 'conversion_of_brand']],on='brand',how='left')
        max_brand = max_brand.groupby('user_id', as_index=False)['n_actions'].agg({ 'user_std_brand': 'std',
                                                                               'user_gini_brand': get_gini,
                                                                               'user_mean_brand': 'mean'})
        max_brand = pd.merge(max_brand, max_brand_id[['user_id', 'conversion_of_brand']], on='user_id', how='left')
        pickle.dump(max_brand, open(dump_path, 'wb+'))
    return max_brand

# 用户浏览的速度
def get_user_frequency(end_date, n_days):
    dump_path = r'F:\cache\user_frequency_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions['minute'] = actions['time'].apply(lambda x: x[:16])
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        action_minutes = actions.groupby('user_id',as_index=False)['minute'].agg({'action_minutes':'nunique'})
        action_days = actions.groupby('user_id',as_index=False)['date'].agg({'action_days':'nunique'})
        action_type1 = actions[actions['type']==1].groupby('user_id',as_index=False)['type'].agg({'action_type1':'count'})
        action_type6 = actions[actions['type']==6].groupby('user_id',as_index=False)['type'].agg({'action_type6':'count'})
        action_type = actions.groupby('user_id',as_index=False)['type'].agg({'action_type':'count'})
        actions = pd.merge(action_minutes,action_days,on='user_id',how='left')
        actions = pd.merge(actions,action_type1,on='user_id',how='left')
        actions = pd.merge(actions,action_type6,on='user_id',how='left')
        actions = pd.merge(actions,action_type,on='user_id',how='left').fillna(0)
        actions['user_minutes_perday'] = actions['action_minutes']/actions['action_days']
        actions['user_frequency_type1'] = actions['action_type1']/actions['action_minutes']
        actions['user_frequency_type6'] = actions['action_type6']/actions['action_minutes']
        actions['user_frequency_type'] = actions['action_type']/actions['action_minutes']
        actions = actions[['user_id','user_minutes_perday','user_frequency_type1','user_frequency_type6','user_frequency_type']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 用户最后一天的浏览分钟数
def get_user_last_minutes(end_date, n_days):
    dump_path = r'F:\cache\user_last_minutes_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions['minute'] = actions['time'].apply(lambda x: x[:16])
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        user_last_date = actions.groupby('user_id',as_index=False)['date'].agg({'date':'max'})
        actions = pd.merge(user_last_date,actions,on=['user_id','date'], how='left')
        actions = actions.groupby('user_id',as_index=False)['minute'].agg({'user_last_minutes':'nunique'})
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#用户最后一次浏览行为
def get_user_last_type(end_date, n_days):
    dump_path = r'F:\cache\user_last_type_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_cate8(start_date, end_date)
        actions = actions.drop_duplicates('user_id', keep='last')
        actions['user_last_type'] = actions['type'].map({5:1,2:2,3:3,1:4,6:5,4:6})
        actions = actions[['user_id', 'user_last_type']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户浏览cate8的时间
def get_user_type_tm(start_date,train_end_date):
    dump_path = r'F:\cache\user_type_tm_%s_%s.pkl' % (start_date, train_end_date)
    if os.path.exists(dump_path):
        user_type_last_tm = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date, train_end_date)
        actions = actions[actions['type']!=4]
        user_type_last_tm = actions.drop_duplicates(['user_id','type'], keep='last')
        user_type_last_tm['action_first_tm'] = user_type_last_tm['time'].map(lambda x: diff_of_minutes(train_end_date, x))
        user_type_last_tm = user_type_last_tm[['user_id','type','action_first_tm']]
        user_type_last_tm.set_index(['user_id','type'],inplace=True)
        user_type_last_tm = user_type_last_tm.unstack()
        user_type_last_tm.columns = ['action_first_tm_1','action_first_tm_2','action_first_tm_3',
                                       'action_first_tm_5','action_first_tm_6']
        user_type_last_tm.reset_index(inplace=True)
        pickle.dump(user_type_last_tm, open(dump_path, 'wb+'))
    return user_type_last_tm


#用户type转化率(将用户的多次购买合并为一次)
def get_user_type_conversion(end_date,n_days):
    dump_path = r'F:\cache\user_type_conversion_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        percent = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_other_conversion(start_date, end_date)
        percent = pd.DataFrame(actions.groupby('user_id')['sku_id'].agg({'count_of_other_action': 'count'}))
        percent['count_of_other_action4'] = actions[actions['type'] == 4].groupby('user_id')['sku_id'].count()
        percent['count_of_other_action1'] = actions[actions['type'] == 1].groupby('user_id')['sku_id'].count()
        percent['count_of_other_action2'] = actions[actions['type'] == 2].groupby('user_id')['sku_id'].count()
        percent['count_of_other_action3'] = actions[actions['type'] == 3].groupby('user_id')['sku_id'].count()
        percent['count_of_other_action5'] = actions[actions['type'] == 5].groupby('user_id')['sku_id'].count()
        percent['count_of_other_action6'] = actions[actions['type'] == 6].groupby('user_id')['sku_id'].count()
        percent['count_of_other_action4'] = percent['count_of_other_action4'].fillna(0)
        percent['user_1_conversion'] = percent['count_of_other_action4'] / percent['count_of_other_action1']
        percent['user_2_conversion'] = percent['count_of_other_action4'] / percent['count_of_other_action2']
        percent['user_3_conversion'] = percent['count_of_other_action4'] / percent['count_of_other_action3']
        percent['user_5_conversion'] = percent['count_of_other_action4'] / percent['count_of_other_action5']
        percent['user_6_conversion'] = percent['count_of_other_action4'] / percent['count_of_other_action6']
        percent['user_action_conversion'] = percent['count_of_other_action4'] / percent['count_of_other_action']
        percent.reset_index(inplace=True)
        percent['count_of_other_action'] = percent.apply(
            lambda x: 0 if x['count_of_other_action4'] > 0 else x['count_of_other_action'], axis=1)
        columns = ['user_id', 'user_1_conversion', 'user_2_conversion', 'user_3_conversion',
                   'user_5_conversion', 'user_6_conversion', 'user_action_conversion',
                   'count_of_other_action4','count_of_other_action']
        percent = percent[columns]
        percent.replace(np.inf,1,inplace=True)
        pickle.dump(percent, open(dump_path, 'wb+'))
    return percent


#用户sku数转化率
def get_user_sku_conversion(end_date,n_days):
    dump_path = r'F:\cache\user_sku_conversion_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        percent = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_other_conversion(start_date, end_date)
        percent = pd.DataFrame(actions.groupby('user_id')['sku_id'].agg({'look_n_other_sku': 'nunique'}))
        percent['buy_n_others_sku'] = actions[actions['type']==4].groupby('user_id')['sku_id'].nunique()
        percent['type2_n_other_sku'] = actions[actions['type'] == 2].groupby('user_id')['sku_id'].nunique()
        percent['type3_n_other_sku'] = actions[actions['type'] == 3].groupby('user_id')['sku_id'].nunique()
        percent['type5_n_other_sku'] = actions[actions['type'] == 5].groupby('user_id')['sku_id'].nunique()
        percent.reset_index(inplace=True)
        percent['buy_n_others_sku'].fillna(0,inplace=True)
        percent['user_sku_conversion_of_look'] = percent['buy_n_others_sku'] / percent['look_n_other_sku']
        percent['user_sku_conversion_of_type2'] = percent['buy_n_others_sku'] / percent['type2_n_other_sku']
        percent['user_sku_conversion_of_type3'] = percent['buy_n_others_sku'] / percent['type3_n_other_sku']
        percent['user_sku_conversion_of_type5'] = percent['buy_n_others_sku'] / percent['type5_n_other_sku']
        percent['look_n_other_sku'] = percent.apply(
            lambda x: 0 if x['buy_n_others_sku'] > 0 else x['look_n_other_sku'], axis=1)
        percent['type2_n_other_sku'] = percent.apply(
            lambda x: 0 if x['buy_n_others_sku'] > 0 else x['type2_n_other_sku'], axis=1)
        percent['type3_n_other_sku'] = percent.apply(
            lambda x: 0 if x['buy_n_others_sku'] > 0 else x['type3_n_other_sku'], axis=1)
        percent['type5_n_other_sku'] = percent.apply(
            lambda x: 0 if x['buy_n_others_sku'] > 0 else x['type5_n_other_sku'], axis=1)
        percent = percent[['user_id','look_n_other_sku','buy_n_others_sku','user_sku_conversion_of_look',
                           'user_sku_conversion_of_type2','user_sku_conversion_of_type3','user_sku_conversion_of_type5',
                           'type2_n_other_sku','type3_n_other_sku','type5_n_other_sku']]
        pickle.dump(percent, open(dump_path, 'wb+'))
    return percent



#用户cate数转化率
def get_user_cate_conversion(end_date,n_days):
    dump_path = r'F:\cache\user_cate_conversion_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        percent = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_other_conversion(start_date, end_date)
        percent = pd.DataFrame(actions.groupby('user_id')['cate'].agg({'look_n_other_cate': 'nunique'}))
        percent['buy_n_others_cate'] = actions[actions['type']==4].groupby('user_id')['cate'].nunique()
        percent['type2_n_other_cate'] = actions[actions['type'] == 2].groupby('user_id')['cate'].nunique()
        percent['type3_n_other_cate'] = actions[actions['type'] == 3].groupby('user_id')['cate'].nunique()
        percent['type5_n_other_cate'] = actions[actions['type'] == 5].groupby('user_id')['cate'].nunique()
        percent.reset_index(inplace=True)
        percent['buy_n_others_cate'].fillna(0,inplace=True)
        percent['cate_conversion_of_look'] = percent['buy_n_others_cate'] / percent['look_n_other_cate']
        percent['cate_conversion_of_type2'] = percent['buy_n_others_cate'] / percent['type2_n_other_cate']
        percent['cate_conversion_of_type3'] = percent['buy_n_others_cate'] / percent['type3_n_other_cate']
        percent['cate_conversion_of_type5'] = percent['buy_n_others_cate'] / percent['type5_n_other_cate']
        percent['look_n_other_cate'] = percent.apply(
            lambda x: 0 if x['buy_n_others_cate'] > 0 else x['look_n_other_cate'], axis=1)
        percent['type2_n_other_cate'] = percent.apply(
            lambda x: 0 if x['buy_n_others_cate'] > 0 else x['type2_n_other_cate'], axis=1)
        percent['type3_n_other_cate'] = percent.apply(
            lambda x: 0 if x['buy_n_others_cate'] > 0 else x['type3_n_other_cate'], axis=1)
        percent['type5_n_other_cate'] = percent.apply(
            lambda x: 0 if x['buy_n_others_cate'] > 0 else x['type5_n_other_cate'], axis=1)
        percent = percent[['user_id','look_n_other_cate','buy_n_others_cate','cate_conversion_of_look',
                           'cate_conversion_of_type2','cate_conversion_of_type3','cate_conversion_of_type5',
                           'type2_n_other_cate','type3_n_other_cate','type5_n_other_cate']]
        pickle.dump(percent, open(dump_path, 'wb+'))
    return percent


#用户近期浏览其他商品的model数（累计）
def get_user_other_action_acc(end_date, n_days):
    dump_path = r'F:\cache\user_other_action_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        coe = 0.005
        expoent1 = 0.1
        expoent2 = 0.93
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_other(start_date, end_date)
        actions['diff_of_days'] = actions['time'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_other_action_acc': 'sum'})
        actions['user_other_action_acc'] = actions['user_other_action_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_other_action_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

# 用户浏览cate8的次数占总次数的百分比
def get_user_percent_of_cate8_acc(end_date, n_days):
    dump_path = r'F:\cache\user_percent_of_cate8_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        coe = 0.001
        expoent1 = 0.5
        expoent2 = 0.999
        cate8 = get_cate8(start_date, end_date)
        cate8['diff_of_days'] = cate8['time'].apply(lambda x: diff_of_days(end_date, x))
        cate8['diff_of_days'] = cate8['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        cate8 = cate8.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_cate8_action_acc': 'sum'})
        other = get_other(start_date, end_date)
        other['diff_of_days'] = other['time'].apply(lambda x: diff_of_days(end_date, x))
        other['diff_of_days'] = other['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        other = other.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_other_action_acc': 'sum'})
        actions = pd.merge(cate8,other,on='user_id',how='outer').fillna(0)
        actions['user_percent_of_cate8'] = actions.apply(lambda x:x['user_cate8_action_acc']/(x['user_cate8_action_acc']+x['user_other_action_acc']),axis=1)
        actions = actions[['user_id', 'user_percent_of_cate8']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户cate8行为特征（各种行为总次数）
def get_user_action(start_date, end_date, i):
    dump_path = r'F:\cache\user_action_%s_%s.pkl' % (start_date, end_date)
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
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户全部行为特征（各种行为总次数）
def get_user_all_action(start_date, end_date, i):
    dump_path = r'F:\cache\user_all_action_%s_%s.pkl' % (start_date, end_date)
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
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户type1次数（累计）
def get_user_type1_acc(start_date, end_date):
    dump_path = r'F:\cache\user_type1_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent = 0.59
        n_days = diff_of_days(end_date,start_date)-1
        sum_weight = (expoent**(n_days+1)-expoent)/(expoent-1)
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type']==1]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x:diff_of_days(end_date,x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x:expoent**x)
        actions = actions.groupby('user_id',as_index=False)['diff_of_days'].agg({'user_type1_acc':'sum'})
        actions['user_type1_acc'] = actions['user_type1_acc']/sum_weight
        actions = actions[['user_id','user_type1_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户添加购物车商品数（累计）
def get_user_type2_acc(start_date, end_date):
    dump_path = r'F:\cache\user_type2_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.005
        expoent1 = 0.47
        expoent2 = 0.999
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] == 2]
        actions = actions[['user_id','sku_id','time']].drop_duplicates()
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_type2_acc': 'sum'})
        actions['user_type2_acc'] = actions['user_type2_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_type2_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户删除购物车次数（累计）
def get_user_type3_acc(start_date, end_date):
    dump_path = r'F:\cache\user_type3_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.00003
        expoent1 = 0.3
        expoent2 = 0.999
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] == 3]
        # actions = actions[['user_id','sku_id','time']].drop_duplicates()
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_type3_acc': 'sum'})
        actions['user_type3_acc'] = actions['user_type3_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_type3_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户关注次数（累计）
def get_user_type5_acc(start_date, end_date):
    dump_path = r'F:\cache\user_type5_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.00003
        expoent1 = 0.3
        expoent2 = 0.999
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] == 5]
        # actions = actions[['user_id','sku_id','time']].drop_duplicates()
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_type5_acc': 'sum'})
        actions['user_type5_acc'] = actions['user_type5_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_type5_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户type6次数（累计）
def get_user_type6_acc(start_date, end_date):
    dump_path = r'F:\cache\user_type6_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.005
        expoent1 = 0.37
        expoent2 = 0.999
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] == 6]
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_type6_acc': 'sum'})
        actions['user_type6_acc'] = actions['user_type6_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_type6_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户type1和type6的百分比（累计）
def get_user_type1type6_acc(start_date, end_date):
    dump_path = r'F:\cache\user_type1type6_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.2
        expoent1 = 0.6
        expoent2 = 0.97
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'].isin([1, 6])]
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby(['user_id', 'type'])['diff_of_days'].sum()
        actions = actions.unstack().fillna(0)
        actions.columns = [1, 6]
        actions['user_type1type6_acc'] = actions[1] / (actions[6] + actions[1])
        actions.reset_index(inplace=True)
        actions = actions[['user_id', 'user_type1type6_acc']]
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

#用户近期model数（累计）
def get_user_n_model_acc(end_date, n_days):
    dump_path = r'F:\cache\user_n_model_acc_%s_%ddays.pkl' % (end_date, n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent1 = 0.99
        n_days = n_days
        start_date = date_add_days(end_date, -n_days)
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        actions = get_cate8(start_date, end_date)
        actions = actions.drop_duplicates(['user_id', 'model_id'], keep='last')
        actions['date'] = actions['time'].apply(lambda x: x[:10])
        actions['diff_of_days'] = actions['date'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x: expoent1 ** x)
        actions = actions.groupby('user_id', as_index=False)['diff_of_days'].agg({'user_n_model_acc': 'sum'})
        actions['user_n_model_acc'] = actions['user_n_model_acc'] / sum_weight1
        actions = actions[['user_id', 'user_n_model_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户活跃分钟数
def get_user_active_minutes(start_date, end_date, i):
    dump_path = r'F:\cache\user_action_minutes_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type'] != 3]
        actions['minute'] = actions['time'].apply(lambda x:x[:16])
        actions = actions[['user_id','minute']].drop_duplicates()
        actions = actions.groupby('user_id',as_index=False)['minute'].count()
        actions.rename(columns = {'minute': '%d_user_active_minutes' % i},inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃分钟数(累计)
def get_user_active_minutes_acc(start_date, end_date):
    dump_path = r'F:\cache\user_action_minutes_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.006
        expoent1 = 0.71
        expoent2 = 0.993
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] != 3]
        actions['minute'] = actions['time'].apply(lambda x: x[:16])
        actions = actions[['user_id', 'minute']].drop_duplicates()
        actions['minute'] = actions['minute'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_minutes'] = actions['minute'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_minutes'].agg({'user_active_minutes_acc': 'sum'})
        actions['user_active_minutes_acc'] = actions['user_active_minutes_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_active_minutes_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions


#统计用户活跃小时数(累计)
def get_user_active_hours_acc(start_date, end_date):
    dump_path = r'F:\cache\user_action_hours_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        coe = 0.005
        expoent1 = 0.2
        expoent2 = 0.999
        n_days = diff_of_days(end_date, start_date) - 1
        sum_weight1 = (expoent1 ** (n_days + 1) - expoent1) / (expoent1 - 1)
        sum_weight2 = (expoent2 ** (n_days + 1) - expoent2) / (expoent2 - 1) * coe
        actions = get_cate8(start_date, end_date)
        actions = actions[actions['type'] != 3]
        actions['hour'] = actions['time'].apply(lambda x: x[:14])
        actions = actions[['user_id', 'hour']].drop_duplicates()
        actions['hour'] = actions['hour'].apply(lambda x: diff_of_days(end_date, x))
        actions['diff_of_hours'] = actions['hour'].apply(lambda x: expoent1 ** x + expoent2 ** x * coe)
        actions = actions.groupby('user_id', as_index=False)['diff_of_hours'].agg({'user_active_hours_acc': 'sum'})
        actions['user_active_hours_acc'] = actions['user_active_hours_acc'] / (sum_weight1 + sum_weight2)
        actions = actions[['user_id', 'user_active_hours_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃天数
def get_user_active_days(start_date, end_date, i):
    dump_path = r'F:\cache\user_action_days_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type'] != 3]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions = actions[['user_id','date']].drop_duplicates()
        actions = actions.groupby('user_id',as_index=False)['date'].count()
        actions.rename(columns = {'date': '%d_user_active_days' % i},inplace=True)
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃天数(累计)
def get_user_active_days_acc(start_date, end_date):
    dump_path = r'F:\cache\user_action_days_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent = 0.995
        n_days = diff_of_days(end_date,start_date)-1
        sum_weight = (expoent**(n_days+1)-expoent)/(expoent-1)
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type']!=3]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions = actions[['user_id','date']].drop_duplicates()
        actions['diff_of_days'] = actions['date'].apply(lambda x:diff_of_days(end_date,x))
        actions['diff_of_days'] = actions['diff_of_days'].apply(lambda x:expoent**x)
        actions = actions.groupby('user_id',as_index=False)['diff_of_days'].agg({'user_active_days_acc':'sum'})
        actions['user_active_days_acc'] = actions['user_active_days_acc']/sum_weight
        actions = actions[['user_id','user_active_days_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#统计用户活跃周数(累计)
def get_user_active_weeks_acc(start_date, end_date):
    dump_path = r'F:\cache\user_action_weeks_acc_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        expoent = 0.81
        n_weeks = (diff_of_days(end_date,start_date)-1)//7+1
        sum_weight = (expoent**(n_weeks+1)-expoent)/(expoent-1)
        actions = get_cate8(start_date,end_date)
        actions = actions[actions['type'] != 3]
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        actions['diff_of_weeks'] = actions['time'].apply(lambda x:diff_of_days(end_date,x)//7)
        actions = actions[['user_id', 'diff_of_weeks']].drop_duplicates()
        actions['diff_of_weeks'] = actions['diff_of_weeks'].apply(lambda x:expoent**x)
        actions = actions.groupby('user_id',as_index=False)['diff_of_weeks'].agg({'user_active_weeks_acc':'sum'})
        actions['user_active_weeks_acc'] = actions['user_active_weeks_acc']/sum_weight
        actions = actions[['user_id','user_active_weeks_acc']]
        pickle.dump(actions, open(dump_path, 'wb+'))
    return actions

#用户购买其他商品的时间
def get_user_other_tm(end_date,n_days):
    dump_path = r'F:\cache\user_other_tm_%s_%ddays.pkl' % (end_date,n_days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        start_date = date_add_days(end_date, -n_days)
        actions = get_other(start_date, end_date)
        other_last_buy_tm = actions[actions['type']==4].drop_duplicates('user_id',keep='last')
        other_last_buy_tm['user_other_last_buy_tm'] = other_last_buy_tm['time'].apply(lambda x:diff_of_days(end_date,x))
        other_last_tm = actions.drop_duplicates('user_id', keep='last')
        other_last_tm['user_other_last_tm'] = other_last_tm['time'].apply(lambda x:diff_of_days(end_date,x))
        actions = pd.merge(other_last_tm,other_last_buy_tm,on='user_id',how='left')
        actions = actions[['user_id','user_other_last_buy_tm','user_other_last_tm']]
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










#构造用户二次特征
def get_user_feat(user):
    user['user_n_sku_exception'] = user['user_n_sku_acc'] * user['user_sku_conversion_of_look']
    user['user_type2_sku_exception'] = user['user_type2_acc'] * user['user_sku_conversion_of_type2']
    user['user_type3_sku_exception'] = user['user_type3_acc'] * user['user_sku_conversion_of_type3']
    user['user_type1_exception'] = user['user_type1_acc'] * user['user_1_conversion']
    user['user_type2_exception'] = user['user_type2_acc'] * user['user_2_conversion']
    user['user_type3_exception'] = user['user_type3_acc'] * user['user_3_conversion']
    user['user_type6_exception'] = user['user_type6_acc'] * user['user_6_conversion']
    user['user_ave_minutes_of_day'] = user['user_active_minutes_acc']/user['user_active_days_acc']


    return user

#构建用户训练集和测试集
def make_user_train_set(train_end_date, test_start_date, test_end_date):
    dump_path = r'F:\cache\user_train_set_%s.pkl' % (train_end_date)
    if os.path.exists(dump_path) & 1:
        actions = pickle.load(open(dump_path, 'rb+'))
        labels = get_labels(test_start_date, test_end_date)
        labels = labels[['user_id', 'label']].drop_duplicates()
    else:
        start_date = "2016-01-31"
        user_id = get_user_id(train_end_date, 30)                                   # 提取用户id
        user = get_basic_user_feat(train_end_date)                                  # 用户基本信息
        user_tm = get_user_tm(start_date,train_end_date)                            # 用户浏览cate8时间
        user_use_day = get_user_use_day(train_end_date,n_days=60)                   # 购买一个物品平均天数
        user_maxday_tm = get_user_maxday_tm(train_end_date,n_days=30)               # 用户浏览时间最多的一天距离现在的时间
        user_first_day = get_user_first_day(train_end_date,n_days=30)               # 用户第一次浏览的时间段，以及当天信息
        user_last_type = get_user_last_type(train_end_date,n_days=30)               # 用户最后一次浏览行为类型
        user_most_active_hour = get_user_most_active_hour(train_end_date,n_days=30) # 用户最活跃的时间段
        user_type4 = get_user_type4(start_date, train_end_date)                     # 用户是否购买过cate8
        user_n_sku = get_user_n_sku_acc(train_end_date,n_days=30)                   # 用户浏览过多少cate8
        user_type2_del_type3 = get_user_type2_del_type3(train_end_date,n_days=30)   # 购物车里还有多少cate8商品（以及cate8所占的比重）
        user_model = get_user_model(train_end_date,n_days=30)                       # 统计用户model信息
        user_max_sku = get_user_max_sku(train_end_date,n_days=30)                   # 用户浏览最多的一个商品次数（商品信息）以及商品转化率
        user_max_brand = get_user_max_brand(train_end_date,n_days=30)               # 用户浏览最多的一个品牌次数，以及转化率
        user_frequency = get_user_frequency(train_end_date,n_days=30)               # 用户的浏览速度
        user_last_minutes = get_user_last_minutes(train_end_date,n_days=30)          # 用户最后一次浏览的分钟数

        # 用户浏览非cate8特征
        user_type_conversion = get_user_type_conversion(train_end_date,n_days=60)   # 用户type转化率
        user_sku_conversion = get_user_sku_conversion(train_end_date,n_days=60)     # 用户浏览物品数转化率
        user_cate_conversion = get_user_cate_conversion(train_end_date,n_days=60)   # 用户cate转化率
        user_other_action_acc = get_user_other_action_acc(train_end_date,n_days=30) # 用户浏览非cate8的次数
        user_percent_of_cate8_acc = get_user_percent_of_cate8_acc(train_end_date,n_days=7)# 用户cate8的浏览量占总浏览量的百分比
        user_other_tm = get_user_other_tm(train_end_date,n_days=30)                 # 用户浏览非cate8的时间
        #user_if_buy_other = ger_user_if_buy_other(start_date,train_end_date)       # 用户最后一次浏览cate8之后是否购买过其他商品
        #user_type_tm = get_user_type_tm(start_date, train_end_date)                # 用户浏览（type）行为事件


        user_type1_acc = get_user_type1_acc(start_date, train_end_date)
        user_type2_acc = get_user_type2_acc(start_date, train_end_date)
        user_type3_acc = get_user_type3_acc(start_date, train_end_date)
        user_type5_acc = get_user_type5_acc(start_date, train_end_date)
        user_type6_acc = get_user_type6_acc(start_date, train_end_date)
        user_type1type6_acc = get_user_type1type6_acc(start_date, train_end_date)
        user_active_minutes_acc = get_user_active_minutes_acc(start_date, train_end_date)
        user_active_hours_acc = get_user_active_hours_acc(start_date, train_end_date)
        user_active_days_acc = get_user_active_days_acc(start_date, train_end_date)
        user_active_weeks_acc = get_user_active_weeks_acc(start_date, train_end_date)
        #user_action = get_cycle(get_user_action, train_end_date, 'user_id', n=11)       # 用户行为次数特征
        #user_all_action = get_cycle(get_user_all_action, train_end_date, 'user_id')  # 用户所有行为次数分析
        #user_active_days = get_cycle(get_user_active_days, train_end_date, 'user_id')       # 用户活跃天数
        #user_active_minutes = get_cycle(get_user_active_minutes, train_end_date, 'user_id') # 用户活跃分钟数
        #user_other = get_cycle(get_user_other_feat, train_end_date, 'user_id')              # 用户近期浏览其他商品的次数
        #user_type_n_sku = get_cycle(get_user_type_sku, train_end_date, 'user_id')           # 用户行为种类对应的商品个数
        #user_n_model = get_cycle(get_user_n_model, train_end_date, 'user_id')               # 用户的model种类数
        user_n_model_acc = get_user_n_model_acc(train_end_date, 30)
        labels = get_labels(test_start_date, test_end_date)                                  # 获取label
        labels = labels[['user_id','label']].drop_duplicates()


        user = pd.merge(user_id, user,                  how='left', on='user_id')
        user = pd.merge(user, user_type1_acc,           how='left', on='user_id')
        user = pd.merge(user, user_type2_acc,           how='left', on='user_id')
        user = pd.merge(user, user_type3_acc,           how='left', on='user_id')
        user = pd.merge(user, user_type5_acc,           how='left', on='user_id')
        user = pd.merge(user, user_type6_acc,           how='left', on='user_id')
        user = pd.merge(user, user_type1type6_acc,      how='left', on='user_id')
        user = pd.merge(user, user_type_conversion,     how='left', on='user_id')
        user = pd.merge(user, user_sku_conversion,      how='left', on='user_id')
        user = pd.merge(user, user_cate_conversion,     how='left', on='user_id')
        user = pd.merge(user, user_other_action_acc,    how='left', on='user_id')
        user = pd.merge(user, user_percent_of_cate8_acc,how='left', on='user_id')
        user = pd.merge(user, user_tm,                  how='left', on='user_id')
        user = pd.merge(user, user_use_day,             how='left', on='user_id')
        user = pd.merge(user, user_maxday_tm,           how='left', on='user_id')
        user = pd.merge(user, user_first_day,           how='left', on='user_id')
        user = pd.merge(user, user_most_active_hour,    how='left', on='user_id')
        user = pd.merge(user, user_type4,               how='left', on='user_id')
        #user = pd.merge(user, user_other_type4,        how='left', on='user_id')
        user = pd.merge(user, user_n_sku,               how='left', on='user_id')
        user = pd.merge(user, user_type2_del_type3,     how='left', on='user_id')
        user = pd.merge(user, user_model,               how='left', on='user_id')
        user = pd.merge(user, user_max_sku,             how='left', on='user_id')
        user = pd.merge(user, user_max_brand,           how='left', on='user_id')
        user = pd.merge(user, user_frequency,           how='left', on='user_id')
        user = pd.merge(user, user_last_minutes,        how='left', on='user_id')
        user = pd.merge(user, user_last_type,           how='left', on='user_id')
        #user = pd.merge(user, user_type_tm,            how='left', on=['user_id'])
        user = pd.merge(user, user_other_tm,            how='left', on='user_id')
        user = pd.merge(user, user_active_minutes_acc,  how='left', on='user_id')
        user = pd.merge(user, user_active_hours_acc,    how='left', on='user_id')
        user = pd.merge(user, user_active_days_acc,     how='left', on='user_id')
        user = pd.merge(user, user_active_weeks_acc,    how='left', on='user_id')
        #user = pd.merge(user, user_other,              how='left', on='user_id')
        #user = pd.merge(user, user_type_n_sku,         how='left', on='user_id')
        user = pd.merge(user, user_n_model_acc,             how='left', on='user_id')
        user = pd.merge(user, labels,                   how='left', on='user_id')

        user = get_user_feat(user)
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

user11, y_user11 = make_user_train_set('2016-04-09', '2016-04-09', '2016-04-16')
print(11)




user12, y_user12 = make_user_train_set('2016-04-12', '2016-04-12', '2016-04-16')
print(12)
user13, y_user13 = make_user_train_set('2016-04-13', '2016-04-13', '2016-04-16')
print(13)
user14, y_user14 = make_user_train_set('2016-04-14', '2016-04-14', '2016-04-16')
print(14)
user15, y_user15 = make_user_train_set('2016-04-15', '2016-04-15', '2016-04-16')
print(15)

user22, y_user22 = make_user_train_set('2016-04-07', '2016-04-07', '2016-04-12')
print(22)
user23, y_user23 = make_user_train_set('2016-04-08', '2016-04-08', '2016-04-13')
print(23)
user24, y_user24 = make_user_train_set('2016-04-09', '2016-04-09', '2016-04-14')
print(24)
user25, y_user25 = make_user_train_set('2016-04-10', '2016-04-10', '2016-04-15')
print(25)

user32, y_user32 = make_user_train_set('2016-04-02', '2016-04-02', '2016-04-07')
print(32)
user33, y_user33 = make_user_train_set('2016-04-03', '2016-04-03', '2016-04-08')
print(33)
user34, y_user34 = make_user_train_set('2016-04-04', '2016-04-04', '2016-04-09')
print(34)
user35, y_user35 = make_user_train_set('2016-04-05', '2016-04-05', '2016-04-10')
print(35)

user42, y_user42 = make_user_train_set('2016-03-28', '2016-03-28', '2016-04-02')
print(42)
user43, y_user43 = make_user_train_set('2016-03-29', '2016-03-29', '2016-04-03')
print(43)
user44, y_user44 = make_user_train_set('2016-03-30', '2016-03-30', '2016-04-04')
print(44)
user45, y_user45 = make_user_train_set('2016-03-31', '2016-03-31', '2016-04-05')
print(45)
