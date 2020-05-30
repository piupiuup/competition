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
from data import *

def main():
    # 计算第一次对cate8的行为在4天内的用户
    get_actions = data()
    user_data = get_actions.actions[get_actions.actions.cate==8]
    user = user_data.groupby('user_id').count().index
    result = np.zeros((len(user),2))
    result_first = ((pd.to_datetime('2016-04-16') -  pd.to_datetime(user_data.groupby('user_id').first()['time']))\
                     / np.timedelta64(1,'D')).values
    user = user_data.groupby('user_id').count().index
    user = pd.DataFrame(user)
    user['first'] = result_first
    user['first'] = (user['first'] <= 4).astype(np.int)
    # --------------------------------
    files = ['../u1_u5/probability/u_result_1.csv', '../u1_u5/probability/u_result_2.csv', '../u1_u5/probability/u_result_3.csv', \
             '../u1_u5/probability/u_result_4.csv', '../u1_u5/probability/u_result_5.csv']
    f_user = {}
    for rounds in range(len(files)):
        # (1) 加载两个ui模型，找出第一次对cate8的行为在4天内的用于预测
        ui1 = pd.read_csv('../ui_model_1.csv')
        ui2 = pd.read_csv('../ui_model_2.csv')
        ui = ui1.copy()
        ui['label'] = (ui1['label'] + ui2['label'])*0.5
        ui = pd.merge(ui, user, how='left', on='user_id')
        ui_select = ui[ui.label>0.3]
        ui_select = ui_select[ui_select['first'] == 1]
        user_bought = get_actions.actions[(get_actions.actions.type==4) & (get_actions.actions.cate==8)]
        user_bought = user_bought[user_bought.columns[[0,4]]]
        user_bought = user_bought.groupby('user_id', as_index=False).first()
        user_select = pd.merge(ui_select, user_bought, how='left', on=['user_id'])
        user_select = user_select[user_select.type==4]
        user_select = user_select[user_select.columns[[0,1,4]]]
        ui_select = pd.merge(ui_select, user_select, how='left', on=['user_id', 'sku_id'])
        ui_select = ui_select[ui_select.type!=4]
        ui_max_sku = ui_select.groupby('user_id', as_index=False).max()
        ui_max_sku = ui_max_sku[ui_max_sku.columns[[0,1,3]]]
        ui_max_sku = pd.merge(ui_select, ui_max_sku, how='left', on=['user_id', 'sku_id'])
        ui_select_sku = ui_max_sku[ui_max_sku['first_y'] == 1]
        # (2) 加载u模型，选出前800个高预测中第一次对cate8的行为在4天内的用户
        user_model = pd.read_csv(files[int(rounds)])
        user_select = user_model[:800]
        user_select = pd.merge(user_select, user, how='left', on='user_id')
        user_select = user_select[user_select['first']==1]
        # (3) 第一部分，两个七天内的用户取交集
        first_part_user = np.intersect1d(ui_select_sku.values[:,0], user_select.values[:,1])
        ui_max = ui.groupby('user_id',as_index=False).max()
        ui_max = pd.merge(ui, ui_max, how='left', on=['user_id','label'])
        ui_max = ui_max.dropna()
        first_part_user_sku = ui_max[ui_max.user_id.isin(first_part_user)]
        first_part_user_sku = first_part_user_sku.values[:,:2].astype(np.int)
        print (first_part_user_sku.shape[0])
        # (4) 第二部分，两个ui模型均值的最大预测(>0.9)
        ui_highest = ui[ui.label>=0.9]
        ui_highest = np.setdiff1d(ui_highest.values[:,0],ui_select_sku.values[:,0])
        ui_highest = ui[ui.user_id.isin(ui_highest)]
        ui_highest_max = ui_highest.groupby('user_id', as_index=False).max()
        ui_highest_max = ui_highest_max[ui_highest_max.columns[[0,2,3]]]
        ui_highest_max = pd.merge(ui_highest, ui_highest_max, how='left', on=['user_id', 'label'])
        ui_highest_max = ui_highest_max[~np.isnan(ui_highest_max['first_y'])]
        second_part_user_sku = ui_highest_max.values[:,:2].astype(np.int)
        print (second_part_user_sku.shape[0])
        # (5) 第三部分，u模型的非4天首交用户的前150
        user_select = user_model[:800]
        user_select = pd.merge(user_select, user, how='left', on='user_id')
        user_select = user_select[user_select['first']!=1]
        third_part_user = user_select[:150].values[:,1]
        third_part_user_sku = ui_max[ui_max.user_id.isin(third_part_user)]
        third_part_user_sku = third_part_user_sku.values[:,:2].astype(np.int)
        print (third_part_user_sku.shape[0])
        # (6) 第四部分，ui模型4天首交用户非第一部分中的前100
        ui_max_select = ui_select_sku[~ui_select_sku.user_id.isin(first_part_user)]
        ui_max_select.sort_values(by='label', ascending=False, inplace=True)
        fourth_part_user = ui_max_select[:100].values[:,0]
        fourth_part_user_sku = ui_select_sku[ui_select_sku.user_id.isin(fourth_part_user)]
        fourth_part_user_sku = fourth_part_user_sku.values[:100,:2].astype(np.int)
        print (fourth_part_user_sku.shape[0])
        # (7) 综合所有四个部分
        final_user = np.union1d(first_part_user, second_part_user_sku[:,0])
        final_user = np.union1d(final_user, third_part_user)
        final_user = np.union1d(final_user, fourth_part_user)
        final_user_sku = ui_max[ui_max.user_id.isin(final_user)]
        final_user_sku = final_user_sku.values[:,:2].astype(np.int)
        print (final_user_sku.shape[0])
        f_user[str(int(rounds))] = pd.DataFrame({'user_id':final_user})
        f_user[str(int(rounds))][str(int(rounds))] = 1
    # -------------------
    # 开始投票
    final_user = f_user['0']
    for rounds in range(1, len(files)):
        final_user = pd.merge(final_user, f_user[str(rounds)], how='outer', on='user_id')
    final_user = final_user.fillna(0)
    final_user['sum'] = 0
    for rounds in range(len(files)):
        final_user['sum'] += final_user[str(rounds)]
    final_user = final_user[final_user['sum']>=2]
    final_user = final_user.values[:,0]
    final_user_sku = ui_max[ui_max.user_id.isin(final_user)]
    final_user_sku = final_user_sku.values[:,:2].astype(np.int)
    print (final_user_sku.shape[0])
    submit = pd.DataFrame({'user_id':final_user_sku[:,0].astype(np.int),'sku_id':final_user_sku[:,1].astype(np.int)})
    submit = submit[['user_id', 'sku_id']]
    submit.to_csv(args.output,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for final submission")
    parser.add_argument('--output', type=str, default='./submission', help='the name of output file')
    args = parser.parse_args()
    args.output += '.csv'
    main()

