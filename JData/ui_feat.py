#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import random
import argparse

random.seed(999)

action_1_path = "./JData_Action_201602.csv"
action_2_path = "./JData_Action_201603.csv"
action_3_path = "./JData_Action_201604.csv"
comment_path = "./JData_Comment.csv"
product_path = "./JData_Product.csv"
user_path = "./JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22",
                "2016-02-29", "2016-03-07", "2016-03-14", "2016-03-21",
                "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]


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


class data(object):
    """docstring for data"""

    def __init__(self):
        super(data, self).__init__()
        action_1 = pd.read_csv(action_1_path)
        action_2 = pd.read_csv(action_2_path)
        action_3 = pd.read_csv(action_3_path)
        actions = pd.concat([action_1, action_2, action_3])
        self.actions = actions
        self.user = pd.read_csv(user_path, encoding='gbk')
        self.product = pd.read_csv(product_path)
        self.comment = pd.read_csv(comment_path)
        self.start_date = "2016-01-01"
        self.end_date = "2016-01-01"
        self.current_actions = None
        if 'cache' not in os.listdir(os.getcwd()):
            os.mkdir('cache')

    def get_actions(self, start_date, end_date):
        if (pd.to_datetime(self.start_date) == pd.to_datetime(start_date)) & (
            pd.to_datetime(self.end_date) == pd.to_datetime(end_date)):
            return self.current_actions.copy()
        else:
            self.start_date = start_date
            self.end_date = end_date
            self.current_actions = self.actions[(self.actions.time >= start_date) & (self.actions.time < end_date)]
            return self.current_actions.copy()

    def get_basic_user_feature(self):
        data_path = './cache/basic_user.hdf'
        if os.path.exists(data_path):
            user = pd.read_hdf(data_path, 'w')
        else:
            user = self.user
            user['age'] = user['age'].map(convert_age)
            age = pd.get_dummies(user["age"], prefix="age")
            sex = pd.get_dummies(user["sex"], prefix="sex")
            user_lv = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
            user = pd.concat([user['user_id'], age, sex, user_lv], axis=1)
            user.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return user

    def get_basic_product_feature(self):
        data_path = './cache/basic_product.hdf'
        if os.path.exists(data_path):
            product = pd.read_hdf(data_path, 'w')
        else:
            product = self.product
            attr1 = pd.get_dummies(product["a1"], prefix="a1")
            attr2 = pd.get_dummies(product["a2"], prefix="a2")
            attr3 = pd.get_dummies(product["a3"], prefix="a3")
            product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1, attr2, attr3], axis=1)
            product.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return product

    def get_user_action_ratio(self, start_date, end_date):
        features = ['user_id',
                    'user_action_1_ratio', 'user_action_2_ratio',
                    'user_action_3_ratio', 'user_action_5_ratio',
                    'user_action_6_ratio']
        data_path = './cache/user_action_ratio_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            types = pd.get_dummies(actions['type'], prefix='action')
            actions = pd.concat([actions['user_id'], types], axis=1)
            actions = actions.groupby(['user_id'], as_index=False).sum()
            actions['user_action_1_ratio'] = actions['action_4'] / (actions['action_1'] + 1e-1)
            actions['user_action_2_ratio'] = actions['action_4'] / (actions['action_2'] + 1e-1)
            actions['user_action_3_ratio'] = actions['action_4'] / (actions['action_3'] + 1e-1)
            actions['user_action_5_ratio'] = actions['action_4'] / (actions['action_5'] + 1e-1)
            actions['user_action_6_ratio'] = actions['action_4'] / (actions['action_6'] + 1e-1)
            actions = actions[features]
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def get_product_action_ratio(self, start_date, end_date):
        features = ['sku_id',
                    'product_action_1_ratio', 'product_action_2_ratio',
                    'product_action_3_ratio', 'product_action_5_ratio',
                    'product_action_6_ratio']
        data_path = './cache/product_action_ratio_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            types = pd.get_dummies(actions['type'], prefix='action')
            actions = pd.concat([actions['sku_id'], types], axis=1)
            actions = actions.groupby(['sku_id'], as_index=False).sum()
            actions['product_action_1_ratio'] = actions['action_4'] / (actions['action_1'] + 1e-1)
            actions['product_action_2_ratio'] = actions['action_4'] / (actions['action_2'] + 1e-1)
            actions['product_action_3_ratio'] = actions['action_4'] / (actions['action_3'] + 1e-1)
            actions['product_action_5_ratio'] = actions['action_4'] / (actions['action_5'] + 1e-1)
            actions['product_action_6_ratio'] = actions['action_4'] / (actions['action_6'] + 1e-1)
            actions = actions[features]
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def get_comment_feature(self, start_date, end_date):
        data_path = './cache/comment_feature_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            comment = pd.read_hdf(data_path, 'w')
        else:
            comment = self.comment
            comment_date_begin = comment_date[0]
            comment_date_end = end_date
            for date in reversed(comment_date):
                if date < comment_date_end:
                    comment_date_begin = date
                    break
            comment = comment[(comment.dt >= comment_date_begin) & (comment.dt < comment_date_end)]
            df = pd.get_dummies(comment['comment_num'], prefix='comment_num')
            comment = pd.concat([comment, df], axis=1)
            comment = comment[
                ['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3',
                 'comment_num_4']]
            comment.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return comment

    def get_user_last_action_time(self, start_date, end_date):
        data_path = './cache/user_last_action_time_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            last_action_time = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions.sort_values('time')
            last_action_time = actions.groupby(['user_id', 'type'], as_index=False)['time'].last()
            last_action_time['time'] = (pd.to_datetime(last_action_time['time'].str[:10]) - pd.to_datetime(
                end_date[:10])) / np.timedelta64(1, 'D')
            df = pd.get_dummies(last_action_time.type, prefix='last_action_time').mul(last_action_time['time'], axis=0)
            last_action_time = pd.concat([last_action_time[['user_id']], df], axis=1)
            last_action_time = last_action_time.groupby(['user_id'], as_index=False).sum()
            last_action_time.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return last_action_time

    def get_user_sku_last_action_time(self, start_date, end_date):
        data_path = './cache/user_sku_last_action_time_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            last_action_time = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions.sort_values('time')
            last_action_time = actions.groupby(['user_id', 'sku_id', 'type'], as_index=False)['time'].last()
            last_action_time['time'] = (pd.to_datetime(last_action_time['time'].str[:10]) - pd.to_datetime(
                end_date[:10])) / np.timedelta64(1, 'D')
            df = pd.get_dummies(last_action_time.type, prefix='last_action_time').mul(last_action_time['time'], axis=0)
            last_action_time = pd.concat([last_action_time[['user_id', 'sku_id']], df], axis=1)
            last_action_time = last_action_time.groupby(['user_id', 'sku_id'], as_index=False).sum()
            last_action_time.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return last_action_time

    def get_labels(self, start_date, end_date):
        data_path = './cache/labels_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions[actions['type'] == 4]
            actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
            actions['label'] = 1
            actions = actions[['user_id', 'sku_id', 'label']]
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def get_register_distance(self, end_date):
        data_path = './cache/user_register_distance_%s.hdf' % (end_date)
        if os.path.exists(data_path):
            user = pd.read_hdf(data_path, 'w')
        else:
            user = self.user
            user['distance'] = (pd.to_datetime(user.user_reg_tm) - pd.to_datetime(end_date)) / np.timedelta64(1, 'D')
            user = user[['user_id', 'distance']]
            user.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return user

    def get_user_window_feature(self, begin_date, end_date):
        data_path = './cache/get_user_window_feature_%s_%s.hdf' % (begin_date, end_date)
        if os.path.exists(data_path):
            user_window_feature = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(begin_date, end_date)
            # first two features: the first and last time of user to interact / time_range
            user_action_first = actions.groupby('user_id').first()
            user_action_last = actions.groupby('user_id').last()
            time_range = ((pd.to_datetime(end_date) - pd.to_datetime(begin_date)) / np.timedelta64(1, 'D')) + 1
            user_first_action_ratio = ((pd.to_datetime(end_date) - pd.to_datetime(
                user_action_first.time)) / np.timedelta64(1, 'D')) / time_range
            user_first_action_ratio = 1 - user_first_action_ratio
            user_first_action_ratio = pd.DataFrame(user_first_action_ratio).reset_index(level=0)
            user_first_action_ratio.rename(
                columns={user_first_action_ratio.columns[-1]: 'user_first_action_ratio_%s_%s' % (begin_date, end_date)},
                inplace=True)
            user_last_action_ratio = ((pd.to_datetime(end_date) - pd.to_datetime(
                user_action_last.time)) / np.timedelta64(1, 'D')) / time_range
            user_last_action_ratio = 1 - user_last_action_ratio
            user_last_action_ratio = pd.DataFrame(user_last_action_ratio).reset_index(level=0)
            user_last_action_ratio.rename(
                columns={user_last_action_ratio.columns[-1]: 'user_last_action_ratio_%s_%s' % (begin_date, end_date)},
                inplace=True)
            # second feature: user active date / total date
            user_action_time = actions.groupby(['user_id', 'time'], as_index=False).count()
            user_action_time = user_action_time[['user_id', 'time']].groupby('user_id', as_index=False).count()
            user_action_time['time'] = user_action_time['time'] / time_range
            user_action_time.rename(columns={'time': 'user_active_rate_%s_%s' % (begin_date, end_date)}, inplace=True)
            # third three feature: user avg, min, max days range from interaction to buy
            user_action_buy = actions[actions.type == 4].groupby(['user_id', 'sku_id']).count()
            user_interact_buy = actions.set_index(keys=['user_id', 'sku_id']).loc[user_action_buy.index]
            user_interact_buy = user_interact_buy.reset_index(level=(0, 1)).groupby(
                ['user_id', 'sku_id', 'time', 'type'], as_index=False).first()
            user_first_buy_time = pd.to_datetime(
                user_interact_buy[user_interact_buy.type == 4].groupby(['user_id', 'sku_id']).first().time).reset_index(
                level=(0, 1))
            user_first_op_time = pd.to_datetime(
                user_interact_buy[user_interact_buy.type != 4].groupby(['user_id', 'sku_id']).first().time).reset_index(
                level=(0, 1))
            user_first_buy_time = pd.merge(user_first_buy_time, user_first_op_time, how='left',
                                           on=['user_id', 'sku_id'])
            user_first_buy_time['user_buy_range'] = (user_first_buy_time['time_x'] - user_first_buy_time[
                'time_y']) / np.timedelta64(1, 'D')
            user_first_buy_time = user_first_buy_time.fillna(0)
            user_buy_range = user_first_buy_time[['user_id', 'user_buy_range']]
            user_buy_range_avg = user_buy_range.groupby(['user_id'], as_index=False).mean()
            user_buy_range_min = user_buy_range.groupby(['user_id'], as_index=False).min()
            user_buy_range_max = user_buy_range.groupby(['user_id'], as_index=False).max()
            user_buy_range_stat = user_action_first[['sku_id', 'type']].reset_index(level=0)
            user_buy_range_stat = pd.merge(user_buy_range_stat, user_buy_range_avg, how='left', on=['user_id'])
            user_buy_range_stat = pd.merge(user_buy_range_stat, user_buy_range_min, how='left', on=['user_id'])
            user_buy_range_stat = pd.merge(user_buy_range_stat, user_buy_range_max, how='left', on=['user_id'])
            user_buy_range_stat = user_buy_range_stat.fillna(time_range)
            user_buy_range_stat[user_buy_range_stat.columns[-3:]] /= time_range
            user_buy_range_stat[user_buy_range_stat.columns[-3:]] = 1 - user_buy_range_stat[
                user_buy_range_stat.columns[-3:]]
            user_buy_range_stat.rename(
                columns={user_buy_range_stat.columns[-3]: 'user_range_avg_%s_%s' % (begin_date, end_date)},
                inplace=True)
            user_buy_range_stat.rename(
                columns={user_buy_range_stat.columns[-2]: 'user_range_min_%s_%s' % (begin_date, end_date)},
                inplace=True)
            user_buy_range_stat.rename(
                columns={user_buy_range_stat.columns[-1]: 'user_range_max_%s_%s' % (begin_date, end_date)},
                inplace=True)
            # fourth six features: all of the ratio features
            user_action_type = pd.get_dummies(actions['type'], prefix='action')
            user_action_type = pd.concat([actions['user_id'], user_action_type], axis=1)
            user_action_type = user_action_type.groupby(['user_id'], as_index=False).sum()
            user_action_type['buy_vs_cart_ratio_%s_%s' % (begin_date, end_date)] = user_action_type['action_4'] / (
            user_action_type['action_2'] + user_action_type['action_4'] + 1e-6)
            user_action_type['buy_vs_dcart_ratio_%s_%s' % (begin_date, end_date)] = user_action_type['action_4'] / (
            user_action_type['action_3'] + user_action_type['action_4'] + 1e-6)
            user_action_type['buy_vs_attention_ratio_%s_%s' % (begin_date, end_date)] = user_action_type['action_4'] / (
            user_action_type['action_5'] + user_action_type['action_4'] + 1e-6)
            user_action_type['buy_vs_click_ratio_%s_%s' % (begin_date, end_date)] = user_action_type['action_4'] / (
            user_action_type['action_1'] + user_action_type['action_4'] + 1e-6)
            user_action_type['buy_vs_surf_ratio_%s_%s' % (begin_date, end_date)] = user_action_type['action_4'] / (
            user_action_type['action_6'] + user_action_type['action_4'] + 1e-6)
            user_action_type['cart_vs_dcart_ratio_%s_%s' % (begin_date, end_date)] = user_action_type['action_2'] / (
            user_action_type['action_2'] + user_action_type['action_3'] + 1e-6)
            user_action_type[user_action_type.columns[1:7]] = np.log10(
                user_action_type[user_action_type.columns[1:7]].values + 1)
            # fifth one features: the number of bought sku v.s. the number of operation sku
            user_sku_buy = actions[actions.type == 4].groupby(['user_id', 'sku_id'], as_index=False).count()
            user_sku_buy = user_sku_buy.groupby(['user_id'], as_index=False).count()
            user_sku_op = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
            user_sku_op = user_sku_op.groupby(['user_id'], as_index=False).count()
            user_sku_op = pd.merge(user_sku_op, user_sku_buy, how='left', on=['user_id'])
            user_sku_op = user_sku_op.fillna(0)
            user_sku_op['buy_vs_op_%s_%s' % (begin_date, end_date)] = user_sku_op['sku_id_y'] / user_sku_op['sku_id_x']
            # sixth two features: the number of days for buy vs the number of days for operation
            user_sku_buy_day = actions[actions.type == 4].groupby(['user_id', 'time'], as_index=False).count()
            user_sku_buy_day = user_sku_buy_day.groupby('user_id', as_index=False).count()
            user_sku_op_day = actions.groupby(['user_id', 'time'], as_index=False).count()
            user_sku_op_day = user_sku_op_day.groupby('user_id', as_index=False).count()
            user_sku_op_day = pd.merge(user_sku_op_day, user_sku_buy_day, how='left', on=['user_id'])
            user_sku_op_day = user_sku_op_day.fillna(0)
            user_sku_op_day['buy_vs_op_day_%s_%s' % (begin_date, end_date)] = user_sku_op_day['sku_id_y'] / \
                                                                              user_sku_op_day['sku_id_x']
            # final concatenation
            user_window_feature = pd.concat(
                [user_first_action_ratio['user_id'], user_first_action_ratio[user_first_action_ratio.columns[-1]], \
                 user_last_action_ratio[user_last_action_ratio.columns[-1]], \
                 user_action_time[user_action_time.columns[-1]], \
                 user_buy_range_stat[user_buy_range_stat.columns[-3:]], \
                 user_action_type[user_action_type.columns[1:]], \
                 user_sku_op['buy_vs_op_%s_%s' % (begin_date, end_date)], \
                 user_sku_op_day['buy_vs_op_day_%s_%s' % (begin_date, end_date)]], axis=1)
            user_window_feature.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return user_window_feature

    def get_sku_window_feature(self, begin_date, end_date):
        data_path = './cache/get_sku_window_feature_%s_%s.hdf' % (begin_date, end_date)
        if os.path.exists(data_path):
            sku_window_feature = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(begin_date, end_date)
            actions.sort_values(by=['sku_id', 'time'], inplace=True)
            # first two features: the first and last time of user to interact / time_range
            sku_action_first = actions.groupby('sku_id').first()
            sku_action_last = actions.groupby('sku_id').last()
            time_range = ((pd.to_datetime(end_date) - pd.to_datetime(begin_date)) / np.timedelta64(1, 'D')) + 1
            sku_first_action_ratio = ((pd.to_datetime(end_date) - pd.to_datetime(
                sku_action_first.time)) / np.timedelta64(1, 'D')) / time_range
            sku_first_action_ratio = 1 - sku_first_action_ratio
            sku_first_action_ratio = pd.DataFrame(sku_first_action_ratio).reset_index(level=0)
            sku_first_action_ratio.rename(
                columns={sku_first_action_ratio.columns[-1]: 'sku_first_action_ratio_%s_%s' % (begin_date, end_date)},
                inplace=True)
            sku_last_action_ratio = ((pd.to_datetime(end_date) - pd.to_datetime(sku_action_last.time)) / np.timedelta64(
                1, 'D')) / time_range
            sku_last_action_ratio = 1 - sku_last_action_ratio
            sku_last_action_ratio = pd.DataFrame(sku_last_action_ratio).reset_index(level=0)
            sku_last_action_ratio.rename(
                columns={sku_last_action_ratio.columns[-1]: 'sku_last_action_ratio_%s_%s' % (begin_date, end_date)},
                inplace=True)
            # second feature: user active date / total date
            sku_action_time = actions.groupby(['sku_id', 'time'], as_index=False).count()
            sku_action_time = sku_action_time[['sku_id', 'time']].groupby('sku_id', as_index=False).count()
            sku_action_time['time'] = sku_action_time['time'] / time_range
            sku_action_time.rename(columns={'time': 'sku_active_rate_%s_%s' % (begin_date, end_date)}, inplace=True)
            # third three feature: user avg, min, max days range from interaction to buy
            sku_action_buy = actions[actions.type == 4].groupby(['sku_id', 'user_id']).count()
            sku_interact_buy = actions.set_index(keys=['sku_id', 'user_id']).loc[sku_action_buy.index]
            sku_interact_buy = sku_interact_buy.reset_index(level=(0, 1)).groupby(['sku_id', 'user_id', 'time', 'type'],
                                                                                  as_index=False).first()
            sku_first_buy_time = pd.to_datetime(
                sku_interact_buy[sku_interact_buy.type == 4].groupby(['sku_id', 'user_id']).first().time).reset_index(
                level=(0, 1))
            sku_first_op_time = pd.to_datetime(
                sku_interact_buy[sku_interact_buy.type != 4].groupby(['sku_id', 'user_id']).first().time).reset_index(
                level=(0, 1))
            sku_first_buy_time = pd.merge(sku_first_buy_time, sku_first_op_time, how='left', on=['sku_id', 'user_id'])
            sku_first_buy_time['sku_buy_range'] = (sku_first_buy_time['time_x'] - sku_first_buy_time[
                'time_y']) / np.timedelta64(1, 'D')
            sku_first_buy_time = sku_first_buy_time.fillna(0)
            sku_buy_range = sku_first_buy_time[['sku_id', 'sku_buy_range']]
            sku_buy_range_avg = sku_buy_range.groupby(['sku_id'], as_index=False).mean()
            sku_buy_range_min = sku_buy_range.groupby(['sku_id'], as_index=False).min()
            sku_buy_range_max = sku_buy_range.groupby(['sku_id'], as_index=False).max()
            sku_buy_range_stat = sku_action_first[['user_id', 'type']].reset_index(level=0)
            sku_buy_range_stat = pd.merge(sku_buy_range_stat, sku_buy_range_avg, how='left', on=['sku_id'])
            sku_buy_range_stat = pd.merge(sku_buy_range_stat, sku_buy_range_min, how='left', on=['sku_id'])
            sku_buy_range_stat = pd.merge(sku_buy_range_stat, sku_buy_range_max, how='left', on=['sku_id'])
            sku_buy_range_stat = sku_buy_range_stat.fillna(time_range)
            sku_buy_range_stat[sku_buy_range_stat.columns[-3:]] /= time_range
            sku_buy_range_stat[sku_buy_range_stat.columns[-3:]] = 1 - sku_buy_range_stat[
                sku_buy_range_stat.columns[-3:]]
            sku_buy_range_stat.rename(
                columns={sku_buy_range_stat.columns[-3]: 'sku_range_avg_%s_%s' % (begin_date, end_date)}, inplace=True)
            sku_buy_range_stat.rename(
                columns={sku_buy_range_stat.columns[-2]: 'sku_range_min_%s_%s' % (begin_date, end_date)}, inplace=True)
            sku_buy_range_stat.rename(
                columns={sku_buy_range_stat.columns[-1]: 'sku_range_max_%s_%s' % (begin_date, end_date)}, inplace=True)
            # fourth six features: all of the ratio features
            sku_action_type = pd.get_dummies(actions['type'], prefix='action')
            sku_action_type = pd.concat([actions['sku_id'], sku_action_type], axis=1)
            sku_action_type = sku_action_type.groupby(['sku_id'], as_index=False).sum()
            sku_action_type['sku_buy_vs_cart_ratio_%s_%s' % (begin_date, end_date)] = sku_action_type['action_4'] / (
            sku_action_type['action_2'] + sku_action_type['action_4'] + 1e-6)
            sku_action_type['sku_buy_vs_dcart_ratio_%s_%s' % (begin_date, end_date)] = sku_action_type['action_4'] / (
            sku_action_type['action_3'] + sku_action_type['action_4'] + 1e-6)
            sku_action_type['sku_buy_vs_attention_ratio_%s_%s' % (begin_date, end_date)] = sku_action_type[
                                                                                               'action_4'] / (
                                                                                           sku_action_type['action_5'] +
                                                                                           sku_action_type[
                                                                                               'action_4'] + 1e-6)
            sku_action_type['sku_buy_vs_click_ratio_%s_%s' % (begin_date, end_date)] = sku_action_type['action_4'] / (
            sku_action_type['action_1'] + sku_action_type['action_4'] + 1e-6)
            sku_action_type['sku_buy_vs_surf_ratio_%s_%s' % (begin_date, end_date)] = sku_action_type['action_4'] / (
            sku_action_type['action_6'] + sku_action_type['action_4'] + 1e-6)
            sku_action_type['sku_cart_vs_dcart_ratio_%s_%s' % (begin_date, end_date)] = sku_action_type['action_2'] / (
            sku_action_type['action_2'] + sku_action_type['action_3'] + 1e-6)
            sku_action_type[sku_action_type.columns[1:7]] = np.log10(
                sku_action_type[sku_action_type.columns[1:7]].values + 1)
            # fifth one features: the number of bought user v.s. the number of operation user
            sku_user_buy = actions[actions.type == 4].groupby(['sku_id', 'user_id'], as_index=False).count()
            sku_user_buy = sku_user_buy.groupby(['sku_id'], as_index=False).count()
            sku_user_op = actions.groupby(['sku_id', 'user_id'], as_index=False).count()
            sku_user_op = sku_user_op.groupby(['sku_id'], as_index=False).count()
            sku_user_op = pd.merge(sku_user_op, sku_user_buy, how='left', on=['sku_id'])
            sku_user_op = sku_user_op.fillna(0)
            sku_user_op['sku_buy_vs_op_%s_%s' % (begin_date, end_date)] = sku_user_op['user_id_y'] / sku_user_op[
                'user_id_x']
            # sixth one features: the number of days for buy vs the number of days for operation
            sku_user_buy_day = actions[actions.type == 4].groupby(['sku_id', 'time'], as_index=False).count()
            sku_user_buy_day = sku_user_buy_day.groupby('sku_id', as_index=False).count()
            sku_user_op_day = actions.groupby(['sku_id', 'time'], as_index=False).count()
            sku_user_op_day = sku_user_op_day.groupby('sku_id', as_index=False).count()
            sku_user_op_day = pd.merge(sku_user_op_day, sku_user_buy_day, how='left', on=['sku_id'])
            sku_user_op_day = sku_user_op_day.fillna(0)
            sku_user_op_day['sku_buy_vs_op_day_%s_%s' % (begin_date, end_date)] = sku_user_op_day['user_id_y'] / \
                                                                                  sku_user_op_day['user_id_x']
            # seventh one features: whether it is new sku
            sku_first_action_ratio['new_sku_%s_%s' % (begin_date, end_date)] = (
            ((pd.to_datetime(end_date) - pd.to_datetime(sku_action_first.time)) \
             / np.timedelta64(1, 'D')) < 3).astype(np.int).values
            # eighth five feature: op ratio for different ratio of users
            self_data = pd.read_csv(user_path, encoding='gbk')
            level_action = pd.merge(actions, self_data, how='left', on=['user_id'])
            level_action_count = level_action.groupby(['sku_id', 'user_lv_cd'], as_index=False).count()
            level_action_total_count = level_action.groupby(['sku_id'], as_index=False).count()
            for i in range(5):
                tmp_count = level_action_count[level_action_count.user_lv_cd == (i + 1)]
                tmp_name = 'user_count_level_' + str(i + 1) + '_%s_%s' % (begin_date, end_date)
                tmp_count.rename(columns={'time': tmp_name}, inplace=True)
                level_action_total_count = pd.merge(level_action_total_count, \
                                                    tmp_count[['sku_id', tmp_name]], how='left', on='sku_id')
            level_action_total_count = level_action_total_count.fillna(0)
            for i in range(5):
                level_action_total_count[level_action_total_count.columns[(i + 1) * -1]] /= level_action_total_count[
                    'time']
            # last concatenation
            sku_window_feature = pd.concat([sku_first_action_ratio['sku_id'], \
                                            sku_first_action_ratio[sku_first_action_ratio.columns[-2:]], \
                                            sku_last_action_ratio[sku_last_action_ratio.columns[-1]], \
                                            sku_action_time[sku_action_time.columns[-1]], \
                                            sku_buy_range_stat[sku_buy_range_stat.columns[-3:]], \
                                            sku_action_type[sku_action_type.columns[1:]], \
                                            sku_user_op['sku_buy_vs_op_%s_%s' % (begin_date, end_date)], \
                                            sku_user_op_day['sku_buy_vs_op_day_%s_%s' % (begin_date, end_date)], \
                                            level_action_total_count[level_action_total_count.columns[-5:]]], axis=1)
            sku_window_feature.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return sku_window_feature

    def get_user_sku_window_feature(self, begin_date, end_date):
        data_path = './cache/get_user_sku_window_feature_%s_%s.hdf' % (begin_date, end_date)
        if os.path.exists(data_path):
            user_sku_window_feature = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(begin_date, end_date)
            actions.sort_values(by=['user_id', 'sku_id', 'time'], inplace=True)
            # first two features: the first and last time of user_sku to interact / time_range
            user_sku_action_first = actions.groupby(['user_id', 'sku_id']).first()
            user_sku_action_last = actions.groupby(['user_id', 'sku_id']).last()
            time_range = ((pd.to_datetime(end_date) - pd.to_datetime(begin_date)) / np.timedelta64(1, 'D')) + 1
            user_sku_first_action_ratio = ((pd.to_datetime(end_date) - pd.to_datetime(
                user_sku_action_first.time)) / np.timedelta64(1, 'D')) / time_range
            user_sku_first_action_ratio = 1 - user_sku_first_action_ratio
            user_sku_first_action_ratio = pd.DataFrame(user_sku_first_action_ratio).reset_index(level=(0, 1))
            user_sku_first_action_ratio.rename(columns={
                user_sku_first_action_ratio.columns[-1]: 'user_sku_first_action_ratio_%s_%s' % (begin_date, end_date)},
                                               inplace=True)
            user_sku_last_action_ratio = ((pd.to_datetime(end_date) - pd.to_datetime(
                user_sku_action_last.time)) / np.timedelta64(1, 'D')) / time_range
            user_sku_last_action_ratio = 1 - user_sku_last_action_ratio
            user_sku_last_action_ratio = pd.DataFrame(user_sku_last_action_ratio).reset_index(level=(0, 1))
            user_sku_last_action_ratio.rename(columns={
                user_sku_last_action_ratio.columns[-1]: 'user_sku_last_action_ratio_%s_%s' % (begin_date, end_date)},
                                              inplace=True)
            # second feature: user_sku active date / total date
            user_sku_action_time = actions.groupby(['user_id', 'sku_id', 'time'], as_index=False).count()
            user_sku_action_time = user_sku_action_time[['user_id', 'sku_id', 'time']].groupby(['user_id', 'sku_id'],
                                                                                               as_index=False).count()
            user_sku_action_time['time'] = user_sku_action_time['time'] / time_range
            user_sku_action_time.rename(columns={'time': 'user_sku_active_rate_%s_%s' % (begin_date, end_date)},
                                        inplace=True)
            # third three feature: user_sku avg days range from interaction to buy
            user_sku_action_buy = actions[actions.type == 4].groupby(['user_id', 'sku_id']).count()
            user_sku_interact_buy = actions.set_index(keys=['user_id', 'sku_id']).loc[user_sku_action_buy.index]
            user_sku_interact_buy = user_sku_interact_buy.reset_index(level=(0, 1)).groupby(
                ['user_id', 'sku_id', 'time', 'type'], as_index=False).first()
            user_sku_first_buy_time = pd.to_datetime(user_sku_interact_buy[user_sku_interact_buy.type == 4].groupby(
                ['user_id', 'sku_id']).first().time).reset_index(level=(0, 1))
            user_sku_first_op_time = pd.to_datetime(user_sku_interact_buy[user_sku_interact_buy.type != 4].groupby(
                ['user_id', 'sku_id']).first().time).reset_index(level=(0, 1))
            user_sku_first_buy_time = pd.merge(user_sku_first_buy_time, user_sku_first_op_time, how='left',
                                               on=['user_id', 'sku_id'])
            user_sku_first_buy_time['user_buy_range'] = (user_sku_first_buy_time['time_x'] - user_sku_first_buy_time[
                'time_y']) / np.timedelta64(1, 'D')
            user_sku_first_buy_time = user_sku_first_buy_time.fillna(0)
            user_sku_buy_range = user_sku_first_buy_time[['user_id', 'sku_id', 'user_buy_range']]
            user_sku_buy_range_avg = user_sku_buy_range.groupby(['user_id', 'sku_id'], as_index=False).mean()
            user_sku_buy_range_stat = user_sku_action_first[['type']].reset_index(level=(0, 1))
            user_sku_buy_range_stat = pd.merge(user_sku_buy_range_stat, user_sku_buy_range_avg, how='left',
                                               on=['user_id', 'sku_id'])
            user_sku_buy_range_stat = user_sku_buy_range_stat.fillna(time_range)
            user_sku_buy_range_stat[user_sku_buy_range_stat.columns[-1]] /= time_range
            user_sku_buy_range_stat[user_sku_buy_range_stat.columns[-1]] = 1 - user_sku_buy_range_stat[
                user_sku_buy_range_stat.columns[-1]]
            user_sku_buy_range_stat.rename(
                columns={user_sku_buy_range_stat.columns[-1]: 'user_sku_range_mean_%s_%s' % (begin_date, end_date)},
                inplace=True)
            # fourth six features: all of the ratio features
            user_sku_action_type = pd.get_dummies(actions['type'], prefix='action_%s_%s' % (begin_date, end_date))
            user_sku_action_type = pd.concat([actions[['user_id', 'sku_id']], user_sku_action_type], axis=1)
            user_sku_action_type = user_sku_action_type.groupby(['user_id', 'sku_id'], as_index=False).sum()
            user_sku_action_type['user_sku_buy_vs_cart_ratio_%s_%s' % (begin_date, end_date)] = user_sku_action_type[
                                                                                                    'action_%s_%s_4' % (
                                                                                                    begin_date,
                                                                                                    end_date)] / \
                                                                                                (user_sku_action_type[
                                                                                                     'action_%s_%s_2' % (
                                                                                                     begin_date,
                                                                                                     end_date)] +
                                                                                                 user_sku_action_type[
                                                                                                     'action_%s_%s_4' % (
                                                                                                     begin_date,
                                                                                                     end_date)] + 1e-6)
            user_sku_action_type['user_sku_buy_vs_dcart_ratio_%s_%s' % (begin_date, end_date)] = user_sku_action_type[
                                                                                                     'action_%s_%s_4' % (
                                                                                                     begin_date,
                                                                                                     end_date)] / \
                                                                                                 (user_sku_action_type[
                                                                                                      'action_%s_%s_3' % (
                                                                                                      begin_date,
                                                                                                      end_date)] +
                                                                                                  user_sku_action_type[
                                                                                                      'action_%s_%s_4' % (
                                                                                                      begin_date,
                                                                                                      end_date)] + 1e-6)
            user_sku_action_type['user_sku_buy_vs_attention_ratio_%s_%s' % (begin_date, end_date)] = \
            user_sku_action_type['action_%s_%s_4' % (begin_date, end_date)] / \
            (user_sku_action_type['action_%s_%s_5' % (begin_date, end_date)] + user_sku_action_type[
                'action_%s_%s_4' % (begin_date, end_date)] + 1e-6)
            user_sku_action_type['user_sku_buy_vs_click_ratio_%s_%s' % (begin_date, end_date)] = user_sku_action_type[
                                                                                                     'action_%s_%s_4' % (
                                                                                                     begin_date,
                                                                                                     end_date)] / \
                                                                                                 (user_sku_action_type[
                                                                                                      'action_%s_%s_1' % (
                                                                                                      begin_date,
                                                                                                      end_date)] +
                                                                                                  user_sku_action_type[
                                                                                                      'action_%s_%s_4' % (
                                                                                                      begin_date,
                                                                                                      end_date)] + 1e-6)
            user_sku_action_type['user_sku_buy_vs_surf_ratio_%s_%s' % (begin_date, end_date)] = user_sku_action_type[
                                                                                                    'action_%s_%s_4' % (
                                                                                                    begin_date,
                                                                                                    end_date)] / \
                                                                                                (user_sku_action_type[
                                                                                                     'action_%s_%s_6' % (
                                                                                                     begin_date,
                                                                                                     end_date)] +
                                                                                                 user_sku_action_type[
                                                                                                     'action_%s_%s_4' % (
                                                                                                     begin_date,
                                                                                                     end_date)] + 1e-6)
            user_sku_action_type['user_sku_cart_vs_dcart_ratio_%s_%s' % (begin_date, end_date)] = user_sku_action_type[
                                                                                                      'action_%s_%s_2' % (
                                                                                                      begin_date,
                                                                                                      end_date)] / \
                                                                                                  (user_sku_action_type[
                                                                                                       'action_%s_%s_2' % (
                                                                                                       begin_date,
                                                                                                       end_date)] +
                                                                                                   user_sku_action_type[
                                                                                                       'action_%s_%s_3' % (
                                                                                                       begin_date,
                                                                                                       end_date)] + 1e-6)
            user_sku_action_type[user_sku_action_type.columns[2:8]] = np.log10(
                user_sku_action_type[user_sku_action_type.columns[2:8]].values + 1)
            # final concatenation
            user_sku_window_feature = pd.concat([user_sku_first_action_ratio[['user_id', 'sku_id']], \
                                                 user_sku_first_action_ratio[user_sku_first_action_ratio.columns[-1]], \
                                                 user_sku_last_action_ratio[user_sku_last_action_ratio.columns[-1]], \
                                                 user_sku_action_time[user_sku_action_time.columns[-1]], \
                                                 user_sku_buy_range_stat[user_sku_buy_range_stat.columns[-1]], \
                                                 user_sku_action_type[user_sku_action_type.columns[-12:]]], axis=1)
            user_sku_window_feature.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return user_sku_window_feature

    def get_product_type_actions(self, start_date, end_date):
        data_path = './cache/product_type_actions_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            product_type = pd.get_dummies(actions['type'], prefix='product_type_actions_%s_%s' % (start_date, end_date))
            if product_type.shape[1] < 6:
                columns = []
                for rounds in range(1, 7):
                    columns.append('product_type_actions_%s_%s_%s' % (start_date, end_date, rounds))
                for item in columns:
                    if item not in product_type.columns:
                        product_type[item] = np.zeros((product_type.shape[0], 1))
                product_type = product_type[columns]
            actions = pd.concat([actions['sku_id'], product_type], axis=1)
            actions = actions.groupby(['sku_id'], as_index=False).sum()
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def get_user_sku_actions(self, start_date, end_date):
        data_path = './cache/user_sku_actions_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions[['user_id', 'sku_id', 'type']]
            user_sku_actions = pd.get_dummies(actions['type'], prefix='user_sku_actions_%s_%s' % (start_date, end_date))
            if user_sku_actions.shape[1] < 6:
                columns = []
                for rounds in range(1, 7):
                    columns.append('user_sku_actions_%s_%s_%s' % (start_date, end_date, rounds))
                for item in columns:
                    if item not in user_sku_actions.columns:
                        user_sku_actions[item] = np.zeros((user_sku_actions.shape[0], 1))
                user_sku_actions = user_sku_actions[columns]
            actions = pd.concat([actions, user_sku_actions], axis=1)
            actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
            del actions['type']
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def get_buy_cate8(self, begin_date, end_date):
        data_path = './cache/buy_cate8_%s_%s.hdf' % (begin_date, end_date)
        if os.path.exists(data_path):
            buy_cate8 = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(begin_date, end_date)
            buy_cate8 = actions[(actions.type == 4) & (actions.cate == 8)].groupby('user_id').count()
            buy_cate8.reset_index(level=0, inplace=True)
            buy_cate8['has_bought_cate_8'] = 1
            buy_cate8 = buy_cate8[['user_id', 'has_bought_cate_8']]
            buy_cate8.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return buy_cate8

    def get_user_first_action(self, begin_date, end_date):
        data_path = './cache/user_first_action_%s_%s.hdf' % (begin_date, end_date)
        if os.path.exists(data_path):
            user_first_action = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(begin_date, end_date)
            first_action = pd.to_datetime(end_date) - pd.to_datetime(actions.groupby('user_id').first().time)
            first_action = first_action / np.timedelta64(1, 'D')
            user = actions.groupby('user_id').count()
            user.reset_index(level=0, inplace=True)
            time_range = ((pd.to_datetime(end_date) - pd.to_datetime(begin_date)) / np.timedelta64(1, 'D')) + 1
            user['first_action'] = first_action.values
            user['first_action_within_3'] = (first_action.values < 3).astype(np.int)
            user['first_action_within_5'] = (first_action.values < 5).astype(np.int)
            user['first_action_within_7'] = (first_action.values < 7).astype(np.int)
            user['first_action'] = first_action.values / time_range
            user['first_action'] = 1 - user['first_action']
            self_data = pd.read_csv(user_path, encoding='gbk')
            user_info = self_data.set_index(keys='user_id').loc[user.user_id].reset_index(level=0)
            user[user.columns[-3:]] = user[user.columns[-3:]].values * user_info['user_lv_cd'].values[:, np.newaxis]
            user_first_action = user[
                ['user_id', 'first_action', 'first_action_within_3', 'first_action_within_5', 'first_action_within_7']]
            user_first_action.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return user_first_action

    def get_user_action_days(self, start_date, end_date):
        data_path = './cache/user_action_days_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            actions['time'] = actions.time.str[:10]
            actions = actions.groupby(['user_id', 'type', 'time'], as_index=False).count()
            actions = actions.groupby(['user_id', 'type'], as_index=False).count()
            user_action_days = pd.get_dummies(actions['type'],
                                              prefix='user_action_days_%s_%s' % (start_date, end_date)).mul(
                actions['time'], axis=0)
            actions = pd.concat([actions['user_id'], user_action_days], axis=1)
            actions = actions.groupby(['user_id'], as_index=False).sum()
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def get_user_product_type_actions(self, start_date, end_date):
        data_path = './cache/user_product_type_actions_%s_%s.hdf' % (start_date, end_date)
        if os.path.exists(data_path):
            actions = pd.read_hdf(data_path, 'w')
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions.groupby(['user_id', 'sku_id', 'type'], as_index=False).count()
            actions = actions.groupby(['user_id', 'type'], as_index=False).count()
            user_product_type = pd.get_dummies(actions['type'],
                                               prefix='user_product_type_actions_%s_%s' % (start_date, end_date)).mul(
                actions['time'], axis=0)
            if user_product_type.shape[1] < 6:
                columns = []
                for rounds in range(1, 7):
                    columns.append('user_product_type_actions_%s_%s_%s' % (start_date, end_date, rounds))
                for item in columns:
                    if item not in user_product_type.columns:
                        user_product_type[item] = np.zeros((user_product_type.shape[0], 1))
                user_product_type = user_product_type[columns]
            actions = pd.concat([actions['user_id'], user_product_type], axis=1)
            actions = actions.groupby(['user_id'], as_index=False).sum()
            actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
        return actions

    def train_test(self, train_start_date, train_end_date, test_start_date='2016-01-01', test_end_date='2016-01-01',
                   train=True):
        dump_path = './cache/train_test_%s_%s_%s_%s.hdf' % (
        train_start_date, train_end_date, test_start_date, test_end_date)
        if os.path.exists(dump_path):
            actions = pd.read_hdf(dump_path, 'w')
        else:
            start_date = "2016-02-01"
            # (1) no date
            basic_user_feature = self.get_basic_user_feature()
            basic_product_feature = self.get_basic_product_feature()
            register_distance = self.get_register_distance(train_end_date)

            # (2) begin with start_date
            user_action_ratio = self.get_user_action_ratio(start_date, train_end_date)
            product_action_ratio = self.get_product_action_ratio(start_date, train_end_date)
            user_window_feature = self.get_user_window_feature(start_date, train_end_date)
            sku_window_feature = self.get_sku_window_feature(start_date, train_end_date)
            user_sku_window_feature = self.get_user_sku_window_feature(start_date, train_end_date)
            buy_cate8 = self.get_buy_cate8(start_date, train_end_date)
            user_first_action1 = self.get_user_first_action(start_date, train_end_date)
            user_first_action2 = self.get_user_first_action(start_date, train_end_date)

            # (3) begin with train_start_date
            comment_feature = self.get_comment_feature(train_start_date, train_end_date)
            user_last_action_time = self.get_user_last_action_time(train_start_date, train_end_date)
            user_sku_last_action_time = self.get_user_sku_last_action_time(train_start_date, train_end_date)

            user_action_days = None
            user_product_type_actions = None
            product_type_actions = None
            actions = None
            for rounds in (1, 2, 3, 5, 7, 10, 15, 21, 30, 45, 60):
                start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=rounds)
                start_date = start_date.strftime('%Y-%m-%d')
                # ---1---
                if product_type_actions is None:
                    product_type_actions = self.get_product_type_actions(start_date, train_end_date)
                else:
                    product_type_actions = pd.merge(product_type_actions, \
                                                    self.get_product_type_actions(start_date, train_end_date), \
                                                    how='outer', on=['sku_id'])
                    # ---2---
                if actions is None:
                    actions = self.get_user_sku_actions(start_date, train_end_date)
                else:
                    actions = pd.merge(actions, \
                                       self.get_user_sku_actions(start_date, train_end_date), \
                                       how='outer', on=['user_id', 'sku_id'])
                # ---3---
                if rounds in (1, 3, 7, 10, 15, 30, 45, 60):
                    if user_product_type_actions is None:
                        user_product_type_actions = self.get_user_product_type_actions(start_date, train_end_date)
                    else:
                        user_product_type_actions = pd.merge(user_product_type_actions, \
                                                             self.get_user_product_type_actions(start_date,
                                                                                                train_end_date), \
                                                             how='outer', on=['user_id'])
                # ---4---
                if rounds in (7, 15, 30, 45, 60):
                    if user_action_days is None:
                        user_action_days = self.get_user_action_days(start_date, train_end_date)
                    else:
                        user_action_days = pd.merge(user_action_days, \
                                                    self.get_user_action_days(start_date, train_end_date), \
                                                    how='outer', on=['user_id'])

            # feature combination
            actions = pd.merge(actions, user_action_days, how='left', on='user_id')
            actions = pd.merge(actions, user_product_type_actions, how='left', on='user_id')
            actions = pd.merge(actions, basic_user_feature, how='left', on='user_id')
            actions = pd.merge(actions, user_action_ratio, how='left', on='user_id')
            actions = pd.merge(actions, register_distance, how='left', on='user_id')
            actions = pd.merge(actions, product_type_actions, how='left', on='sku_id')
            actions = pd.merge(actions, basic_product_feature, how='left', on='sku_id')
            actions = pd.merge(actions, product_action_ratio, how='left', on='sku_id')
            actions = pd.merge(actions, comment_feature, how='left', on='sku_id')
            actions = pd.merge(actions, user_last_action_time, how='left', on='user_id')
            actions = pd.merge(actions, user_sku_last_action_time, how='left', on=['user_id', 'sku_id'])
            actions = pd.merge(actions, user_window_feature, how='left', on='user_id')
            actions = pd.merge(actions, sku_window_feature, how='left', on='sku_id')
            actions = pd.merge(actions, user_sku_window_feature, how='left', on=['user_id', 'sku_id'])
            actions = pd.merge(actions, buy_cate8, how='left', on='user_id')
            actions = pd.merge(actions, user_first_action1, how='left', on='user_id')
            actions = pd.merge(actions, user_first_action2, how='left', on='user_id')
            actions = actions.fillna(0)
            actions.to_hdf(dump_path, 'w', complib='blosc', complevel=5)
        if train == True:
            if test_start_date != '2016-01-01' and test_end_date != '2016-01-01':
                labels = self.get_labels(test_start_date, test_end_date)
            else:
                raise ValueError('test date is missing!')
            actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
            actions = actions.fillna(0)
            user = actions[['user_id', 'sku_id']].copy()
            labels = actions['label'].copy()
            del actions['user_id'], actions['sku_id'], actions['label']
            return user, actions, labels
        else:
            actions = actions[actions['cate'] == 8]
            user = actions[['user_id', 'sku_id']].copy()
            del actions['user_id'], actions['sku_id']
            return user, actions

    def sampling(self, x_data, y_data, rate):
        positive_index = np.where(y_data == 1)[0]
        negative_index = np.where(y_data == 0)[0]
        positive_num = y_data[positive_index].shape[0]
        negative_num = y_data[negative_index].shape[0]
        sampling_num = positive_num * rate
        positive_data = x_data[positive_index, :]
        positive_label = y_data[positive_index]
        negative_data = x_data[negative_index, :]
        negative_label = y_data[negative_index]
        sampling_index = random.sample(range(negative_num), sampling_num)
        negative_data = negative_data[sampling_index, :]
        negative_label = negative_label[sampling_index]
        x_data = np.concatenate([negative_data, positive_data], axis=0)
        y_data = np.concatenate([negative_label, positive_label], axis=0)
        return x_data, y_data

    def make_submission(self, max_depth=3, xgb_rounds=700):
        # moving windows
        windows = []
        train_start_date = '2016-02-25';
        train_end_date = '2016-04-11';
        test_start_date = '2016-04-11';
        test_end_date = '2016-04-16';
        for rounds in xrange(26):
            windows.append({'train_start_date': train_start_date, \
                            'train_end_date': train_end_date, \
                            'test_start_date': test_start_date, \
                            'test_end_date': test_end_date})
            train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d') - timedelta(days=1)
            train_start_date = train_start_date.strftime('%Y-%m-%d')
            test_start_date = datetime.strptime(test_start_date, '%Y-%m-%d') - timedelta(days=1)
            test_start_date = test_start_date.strftime('%Y-%m-%d')
            train_end_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=1)
            train_end_date = train_end_date.strftime('%Y-%m-%d')
            test_end_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(days=1)
            test_end_date = test_end_date.strftime('%Y-%m-%d')

        # train_data
        print
        'Moving windows, rounds: ' + str(1)
        _, train_data, train_label = self.train_test(windows[0]['train_start_date'], windows[0]['train_end_date'], \
                                                     windows[0]['test_start_date'], windows[0]['test_end_date'])
        train_data, train_label = self.sampling(train_data.values, train_label.values, 30)
        for rounds in range(1, 26):
            print
            'Moving windows, rounds: ' + str(rounds + 1)
            _, train_data_temp, train_label_temp = self.train_test(windows[rounds]['train_start_date'],
                                                                   windows[rounds]['train_end_date'], \
                                                                   windows[rounds]['test_start_date'],
                                                                   windows[rounds]['test_end_date'])
            train_data_temp, train_label_temp = self.sampling(train_data_temp.values, train_label_temp.values, 30)
            train_data = np.concatenate([train_data, train_data_temp], axis=0)
            train_label = np.concatenate([train_label, train_label_temp], axis=0)

        # training process
        x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, test_size=0.2, random_state=0)
        x_train = xgb.DMatrix(x_train, label=y_train)
        x_valid = xgb.DMatrix(x_valid, label=y_valid)
        param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': max_depth, 'nthread': 15,
                 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
                 'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
        evalist = [(x_valid, 'evel'), (x_train, 'train')]
        bst = xgb.train(param.items(), x_train, xgb_rounds, evalist)

        # testing process
        user_test, x_test = self.train_test('2016-03-01', '2016-04-16', train=False)
        x_test = xgb.DMatrix(x_test.values)
        y_test = bst.predict(x_test)
        user_test['label'] = y_test
        if max_depth == 3 and xgb_rounds == 700:
            print
            'Output files: ui_model_1.csv'
            user_test.to_csv('ui_model_1.csv', index=False, index_label=False)
        elif max_depth == 5 and xgb_rounds == 500:
            print
            'Output files: ui_model_2.csv'
            user_test.to_csv('ui_model_2.csv', index=False, index_label=False)
        else:
            raise NotImplementedError('Please verify the input variables!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for submission")
    parser.add_argument('--depth', type=int, default=3, help='the max depth of trees')
    parser.add_argument('--rounds', type=int, default=700, help='the training rounds of xgboost')
    args = parser.parse_args()
    if [args.depth, args.rounds] not in [[3, 700], [5, 500]]:
        raise NotImplementedError('Please verify the input variables!')
    actions = data()
    actions.make_submission(args.depth, args.rounds)

data_path = r'C:\Users\csw\Desktop\test.hdf'
actions = pd.concat([actions, actions])
actions.to_hdf(data_path, 'w', complib='blosc', complevel=5)
