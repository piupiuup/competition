# -*- coding:UTF-8 -*-
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from sklearn.model_selection import KFold
from datetime import datetime,timedelta
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

data_root = "../../data"
save_root = "{}/feat".format(data_root)

user_path = "{}/t_user.csv".format(data_root)
loan_path = "{}/t_loan.csv".format(data_root)
click_path = "{}/t_click.csv".format(data_root)
order_path = "{}/t_order.csv".format(data_root)
loan_sum_path = "{}/t_loan_sum.csv".format(data_root)

last_month_end_train = "2016-11-01 00:00:00"
last_month_start_train = "2016-10-01 00:00:00"
last_two_month_start_train = "2016-09-01 00:00:00"
last_three_month_start_train = "2016-08-01 00:00:00"

last_month_end_test = "2016-12-01 00:00:00"
last_month_start_test = "2016-11-01 00:00:00"
last_two_month_start_test = "2016-10-01 00:00:00"
last_three_month_correct_test = "2016-09-01 00:00:00"
last_three_month_start_test = "2016-08-01 00:00:00"

last_month_end_for_click_train = last_month_end_train.split(' ')[0]
last_month_start_for_click_train = last_month_start_train.split(' ')[0]
last_two_month_start_for_click_train = last_two_month_start_train.split(' ')[0]
last_three_month_start_for_click_train = last_three_month_start_train.split(' ')[0]

last_month_end_for_click_test = last_month_end_test.split(' ')[0]
last_month_start_for_click_test = last_month_start_test.split(' ')[0]
last_two_month_start_for_click_test = last_two_month_start_test.split(' ')[0]
last_three_month_correct_for_click_test = last_three_month_correct_test.split(' ')[0]
last_three_month_start_for_click_test = last_three_month_start_test.split(' ')[0]

def rank(data, feat1, feat2, ascending, rank_name):
    # 这部分比自己的实现的要好非常多，好好学习，大概比我实现的快六十倍
    data.sort_values([feat1,feat2], inplace=True, ascending=ascending)
    data[rank_name] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)[rank_name].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data[rank_name] = data[rank_name] - data['min_rank']
    del data['min_rank']
    return data

'''*******************************************loan**********************************************'''
def has_loan():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time > last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time > last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid"]].drop_duplicates()
    test_slice = loan_df.ix[test_time, ["uid"]].drop_duplicates()

    train_slice["has_loan"] = 1
    test_slice["has_loan"] = 1

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_slice, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_slice, on="uid", how="left").fillna(0)
    train_feat.to_csv("{}/has_loan_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/has_loan_test.csv".format(save_root), index=False)

def last_three_month_loan_nums():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid"]]
    test_slice = loan_df.ix[test_time, ["uid"]]
    train_slice["count"] = 1
    test_slice["count"] = 1
    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index().rename(columns={"count": "last_three_month_loan_nums"})
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index().rename(columns={"count": "last_three_month_loan_nums"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_loan_nums_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_nums_test.csv".format(save_root), index=False)

def last_month_loan_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount"]]

    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index().rename(columns={"loan_amount": "last_month_loan_amount"})
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index().rename(columns={"loan_amount": "last_month_loan_amount"})

    print(train_loan_nums.head())
    print(test_loan_nums.head())

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_loan_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_amount_test.csv".format(save_root), index=False)

def last_month_plannum_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "plannum"]]
    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index().rename(columns={"plannum": "last_month_plannum_amount"})
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index().rename(columns={"plannum": "last_month_plannum_amount"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_plannum_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_plannum_amount_test.csv".format(save_root), index=False)

def last_two_month_plannum_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "plannum"]]
    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index().rename(columns={"plannum": "last_two_month_plannum_amount"})
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index().rename(columns={"plannum": "last_two_month_plannum_amount"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_plannum_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_plannum_amount_test.csv".format(save_root), index=False)

def last_three_month_plannum_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "plannum"]]
    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index().rename(columns={"plannum": "last_three_month_plannum_amount"})
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index().rename(columns={"plannum": "last_three_month_plannum_amount"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_plannum_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_plannum_amount_test.csv".format(save_root), index=False)

def last_three_month_loan_div_plannum_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index()
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index()

    train_loan_nums["last_three_month_loan_div_plannum_amount"] = train_loan_nums["loan_amount"] / (train_loan_nums["plannum"]+0)
    test_loan_nums["last_three_month_loan_div_plannum_amount"] = test_loan_nums["loan_amount"] / (test_loan_nums["plannum"]+0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums[["uid", "last_three_month_loan_div_plannum_amount"]], on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums[["uid", "last_three_month_loan_div_plannum_amount"]], on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_loan_div_plannum_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_div_plannum_amount_test.csv".format(save_root), index=False)

def last_month_loan_div_plannum_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]

    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index()
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index()

    train_loan_nums["last_month_loan_div_plannum_amount"] = train_loan_nums["loan_amount"] / (train_loan_nums["plannum"]+0)
    test_loan_nums["last_month_loan_div_plannum_amount"] = test_loan_nums["loan_amount"] / (test_loan_nums["plannum"]+0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums[["uid", "last_month_loan_div_plannum_amount"]], on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums[["uid", "last_month_loan_div_plannum_amount"]], on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_loan_div_plannum_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_div_plannum_amount_test.csv".format(save_root), index=False)

def last_three_month_loan_mean():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount"]]

    train_loan_nums = train_slice.groupby(["uid"]).mean().reset_index().rename(columns={"loan_amount": "last_three_month_loan_mean"})
    test_loan_nums = test_slice.groupby(["uid"]).mean().reset_index().rename(columns={"loan_amount": "last_three_month_loan_mean"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_loan_mean_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_mean_test.csv".format(save_root), index=False)

def last_two_month_plannum_mean():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "plannum"]]
    train_loan_nums = train_slice.groupby(["uid"]).mean().reset_index().rename(columns={"plannum": "last_two_month_plannum_mean"})
    test_loan_nums = test_slice.groupby(["uid"]).mean().reset_index().rename(columns={"plannum": "last_two_month_plannum_mean"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_plannum_mean_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_plannum_mean_test.csv".format(save_root), index=False)

def last_month_loan_mean():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount"]]

    train_loan_nums = train_slice.groupby(["uid"]).mean().reset_index().rename(columns={"loan_amount": "last_month_loan_mean"})
    test_loan_nums = test_slice.groupby(["uid"]).mean().reset_index().rename(columns={"loan_amount": "last_month_loan_mean"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_loan_mean_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_mean_test.csv".format(save_root), index=False)

def date2day_train(x):
    sd = datetime.strptime(last_month_end_train, '%Y-%m-%d %H:%M:%S')
    d = datetime.strptime(x['loan_time'], '%Y-%m-%d %H:%M:%S')
    x['last_loan_day_dis'] = (sd - d).days + 1
    return x
def date2day_test(x):
    sd = datetime.strptime(last_month_end_test, '%Y-%m-%d %H:%M:%S')
    d = datetime.strptime(x['loan_time'], '%Y-%m-%d %H:%M:%S')
    x['last_loan_day_dis'] = (sd - d).days + 1
    return x

def last_time_loan_features():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = (loan_time < last_month_end_train)

    train_df = loan_df.ix[train_time]
    train_slice = train_df[["uid", "loan_time"]]
    test_slice = loan_df[["uid", "loan_time"]]

    train_slice = train_slice.groupby(["uid"]).max().reset_index()
    test_slice = test_slice.groupby(["uid"]).max().reset_index()
    train_slice = train_slice.merge(train_df, on=["uid", "loan_time"], how="left")
    test_slice = test_slice.merge(loan_df, on=["uid", "loan_time"], how="left")

    train_slice["last_loan_month"] = train_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    train_slice["last_loan_day"] = train_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    train_slice["last_loan_hour"] = train_slice.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    test_slice["last_loan_month"] = test_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    test_slice["last_loan_day"] = test_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    test_slice["last_loan_hour"] = test_slice.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    train_slice["last_loan_month"] = train_slice["last_loan_month"].apply(lambda x: 11-x)
    test_slice["last_loan_month"] = test_slice["last_loan_month"].apply(lambda x: 12-x)

    train_slice = train_slice.rename(columns={"loan_amount": "last_loan_amount", "plannum": "last_plannum"})
    test_slice = test_slice.rename(columns={"loan_amount": "last_loan_amount", "plannum": "last_plannum"})

    train_slice = train_slice.apply(date2day_train, axis=1)
    test_slice = test_slice.apply(date2day_test, axis=1)
    print(train_slice.head())
    print(test_slice.head())

    train_slice[
        ["uid", "last_loan_amount", "last_plannum", "last_loan_month", "last_loan_day", "last_loan_hour", "last_loan_day_dis"]].to_csv(
        "{}/last_time_loan_features_train.csv".format(save_root), index=False)
    test_slice[
        ["uid", "last_loan_amount", "last_plannum", "last_loan_month", "last_loan_day", "last_loan_hour", "last_loan_day_dis"]].to_csv(
        "{}/last_time_loan_features_test.csv".format(save_root), index=False)

def first_time_loan_features():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = (loan_time < last_month_end_train)

    train_df = loan_df.ix[train_time]
    train_slice = train_df[["uid", "loan_time"]]
    test_slice = loan_df[["uid", "loan_time"]]

    train_slice = train_slice.groupby(["uid"]).min().reset_index()
    test_slice = test_slice.groupby(["uid"]).min().reset_index()

    train_slice = train_slice.merge(train_df, on=["uid", "loan_time"], how="left")
    test_slice = test_slice.merge(loan_df, on=["uid", "loan_time"], how="left")

    train_slice["first_loan_month"] = train_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    train_slice["first_loan_day"] = train_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    train_slice["first_loan_hour"] = train_slice.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    test_slice["first_loan_month"] = test_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    test_slice["first_loan_day"] = test_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    test_slice["first_loan_hour"] = test_slice.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    train_slice["first_loan_month"] = train_slice["first_loan_month"].apply(lambda x: 11-x)
    test_slice["first_loan_month"] = test_slice["first_loan_month"].apply(lambda x: 12-x)

    train_slice = train_slice.rename(columns={"loan_amount": "first_loan_amount", "plannum": "first_plannum"})
    test_slice = test_slice.rename(columns={"loan_amount": "first_loan_amount", "plannum": "first_plannum"})

    train_slice = train_slice.apply(date2day_train, axis=1)
    test_slice = test_slice.apply(date2day_test, axis=1)
    print(train_slice.head())
    print(test_slice.head())

    train_slice = train_slice.rename(columns={"last_loan_day_dis": "first_loan_day_dis"})
    test_slice = test_slice.rename(columns={"last_loan_day_dis": "first_loan_day_dis"})
    train_slice[
        ["uid", "first_loan_amount", "first_plannum", "first_loan_month", "first_loan_day", "first_loan_hour", "first_loan_day_dis"]].to_csv(
        "{}/first_time_loan_features_train.csv".format(save_root), index=False)
    test_slice[
        ["uid", "first_loan_amount", "first_plannum", "first_loan_month", "first_loan_day", "first_loan_hour", "first_loan_day_dis"]].to_csv(
        "{}/first_time_loan_features_test.csv".format(save_root), index=False)

def last_second_time_loan_features():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = (loan_time < last_month_end_train)

    train_df = loan_df.ix[train_time]
    train_slice = train_df[["uid", "loan_time"]]
    test_slice = loan_df[["uid", "loan_time"]]

    last_time_train_slice = train_slice[["uid", "loan_time"]].groupby(["uid"], as_index=False).max()
    last_time_test_slice = test_slice[["uid", "loan_time"]].groupby(["uid"], as_index=False).max()

    last_time_train_slice["flag"] = 1
    last_time_test_slice["flag"] = 1
    train_slice = train_slice.merge(last_time_train_slice, on=["uid", "loan_time"], how="left").fillna(0)
    test_slice = test_slice.merge(last_time_test_slice, on=["uid", "loan_time"], how="left").fillna(0)

    train_slice = train_slice.ix[train_slice.flag != 1, ["uid", "loan_time"]]
    test_slice = test_slice.ix[test_slice.flag != 1, ["uid", "loan_time"]]

    train_slice = train_slice.merge(train_df, on=["uid", "loan_time"], how="left")
    test_slice = test_slice.merge(loan_df, on=["uid", "loan_time"], how="left")

    train_slice = train_slice.groupby(["uid"]).max().reset_index()
    test_slice = test_slice.groupby(["uid"]).max().reset_index()

    train_slice["last_second_loan_month"] = train_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    train_slice["last_second_loan_day"] = train_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    train_slice["last_second_loan_hour"] = train_slice.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    test_slice["last_second_loan_month"] = test_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    test_slice["last_second_loan_day"] = test_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    test_slice["last_second_loan_hour"] = test_slice.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    train_slice["last_second_loan_month"] = train_slice["last_second_loan_month"].apply(lambda x: 11-x)
    test_slice["last_second_loan_month"] = test_slice["last_second_loan_month"].apply(lambda x: 12-x)

    train_slice = train_slice.rename(columns={"loan_amount": "last_second_loan_amount", "plannum": "last_second_plannum"})
    test_slice = test_slice.rename(columns={"loan_amount": "last_second_loan_amount", "plannum": "last_second_plannum"})

    train_slice = train_slice.apply(date2day_train, axis=1)
    test_slice = test_slice.apply(date2day_test, axis=1)
    train_slice = train_slice.rename(columns={"last_loan_day_dis": "last_second_loan_day_dis"})
    test_slice = test_slice.rename(columns={"last_loan_day_dis": "last_second_loan_day_dis"})

    print(train_slice.head())
    print(test_slice.head())

    train_slice[
        ["uid", "last_second_loan_amount", "last_second_plannum", "last_second_loan_month", "last_second_loan_day", "last_second_loan_hour", "last_second_loan_day_dis"]].to_csv(
        "{}/last_second_time_loan_features_train.csv".format(save_root), index=False)
    test_slice[
        ["uid", "last_second_loan_amount", "last_second_plannum", "last_second_loan_month", "last_second_loan_day", "last_second_loan_hour", "last_second_loan_day_dis"]].to_csv(
        "{}/last_second_time_loan_features_test.csv".format(save_root), index=False)

def last_loan_day_features():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = (loan_time < last_month_end_train)

    train_df = loan_df.ix[train_time]
    test_df = loan_df
    train_slice = train_df[["uid", "loan_time"]]
    test_slice = loan_df[["uid", "loan_time"]]

    train_slice = train_slice.groupby(["uid"]).max().reset_index()
    test_slice = test_slice.groupby(["uid"]).max().reset_index()
    train_slice["last_loan_month"] = train_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    test_slice["last_loan_month"] = test_slice.loan_time.str.split('-', expand=True)[1].astype(np.int)
    train_slice["last_loan_day"] = train_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    test_slice["last_loan_day"] = test_slice.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    train_slice = train_slice[["uid", "last_loan_month", "last_loan_day"]]
    test_slice = test_slice[["uid", "last_loan_month", "last_loan_day"]]

    train_df["last_loan_month"] = train_df.loan_time.str.split('-', expand=True)[1].astype(np.int)
    train_df["last_loan_day"] = train_df.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    train_df["last_loan_hour"] = train_df.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    test_df["last_loan_month"] = test_df.loan_time.str.split('-', expand=True)[1].astype(np.int)
    test_df["last_loan_day"] = test_df.loan_time.str.split(' ', expand=True)[0].str.split('-', expand=True)[2].astype(np.int)
    test_df["last_loan_hour"] = test_df.loan_time.str.split(' ', expand=True)[1].str.split(':', expand=True)[0].astype(np.int)

    train_slice = train_slice.merge(train_df, on=["uid", "last_loan_month", "last_loan_day"], how="left")
    test_slice = test_slice.merge(test_df, on=["uid", "last_loan_month", "last_loan_day"], how="left")

    train_slice["last_loan_month"] = train_slice["last_loan_month"].apply(lambda x: 11-x)
    test_slice["last_loan_month"] = test_slice["last_loan_month"].apply(lambda x: 12-x)

    train_slice["last_day_loan_times"] = 1
    test_slice["last_day_loan_times"] = 1
    train_time_slice = train_slice[["uid", "last_loan_month", "last_loan_day", "last_day_loan_times"]].groupby(["uid", "last_loan_month", "last_loan_day"], as_index=False).count()
    test_time_slice = test_slice[["uid", "last_loan_month", "last_loan_day", "last_day_loan_times"]].groupby(["uid", "last_loan_month", "last_loan_day"], as_index=False).count()

    last_day_loan_amount_sum_train = train_slice[["uid", "last_loan_month", "last_loan_day", "loan_amount"]].groupby(["uid", "last_loan_month", "last_loan_day"], as_index=False).sum().rename(columns={"loan_amount": "last_day_loan_amount_sum"})
    last_day_loan_amount_sum_test = test_slice[["uid", "last_loan_month", "last_loan_day", "loan_amount"]].groupby(["uid", "last_loan_month", "last_loan_day"], as_index=False).sum().rename(columns={"loan_amount": "last_day_loan_amount_sum"})
    train_time_slice = train_time_slice.merge(last_day_loan_amount_sum_train, on=["uid", "last_loan_month", "last_loan_day"], how="left")
    test_time_slice = test_time_slice.merge(last_day_loan_amount_sum_test, on=["uid", "last_loan_month", "last_loan_day"], how="left")

    last_day_plannum_sum_train = train_slice[["uid", "last_loan_month", "last_loan_day", "plannum"]].groupby(["uid", "last_loan_month", "last_loan_day"], as_index=False).sum().rename(columns={"plannum": "last_day_plannum_sum"})
    last_day_plannum_sum_test = test_slice[["uid", "last_loan_month", "last_loan_day", "plannum"]].groupby(["uid", "last_loan_month", "last_loan_day"], as_index=False).sum().rename(columns={"plannum": "last_day_plannum_sum"})
    train_time_slice = train_time_slice.merge(last_day_plannum_sum_train, on=["uid", "last_loan_month", "last_loan_day"], how="left")
    test_time_slice = test_time_slice.merge(last_day_plannum_sum_test, on=["uid", "last_loan_month", "last_loan_day"], how="left")
    print(train_time_slice.head())
    print(test_time_slice.head())

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_time_slice, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_time_slice, on="uid", how="left").fillna(0)
    print(train_feat.head())
    print(test_feat.head())

    train_feat[["uid", "last_day_loan_times", "last_day_loan_amount_sum", "last_day_plannum_sum"]].to_csv(
        "{}/last_loan_day_features_train.csv".format(save_root), index=False)
    test_feat[["uid", "last_day_loan_times", "last_day_loan_amount_sum", "last_day_plannum_sum"]].to_csv(
        "{}/last_loan_day_features_test.csv".format(save_root), index=False)

def last_two_month_loan_mean():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount"]]

    train_loan_nums = train_slice.groupby(["uid"]).mean().reset_index().rename(columns={"loan_amount": "last_two_month_loan_mean"})
    test_loan_nums = test_slice.groupby(["uid"]).mean().reset_index().rename(columns={"loan_amount": "last_two_month_loan_mean"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_loan_mean_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_loan_mean_test.csv".format(save_root), index=False)

def last_two_month_loan_quantile25():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ( (loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train) )
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount"]]

    train_loan_nums = train_slice.groupby(["uid"]).quantile(q=0.25).reset_index().rename(columns={"loan_amount": "last_two_month_loan_quantile25"})
    test_loan_nums = test_slice.groupby(["uid"]).quantile(q=0.25).reset_index().rename(columns={"loan_amount": "last_two_month_loan_quantile25"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_loan_quantile25_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_loan_quantile25_test.csv".format(save_root), index=False)

def amount_plannum_repeat_loan():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    train_slice["count"] = 1
    test_slice["count"] = 1
    train_slice = train_slice.groupby(["uid", "loan_amount", "plannum"], as_index=False).count()
    test_slice = test_slice.groupby(["uid", "loan_amount", "plannum"], as_index=False).count()

    train_slice = train_slice[train_slice["count"] > 1]
    test_slice = test_slice[test_slice["count"] > 1]
    train_slice["is_repeat_loan"] = 1
    test_slice["is_repeat_loan"] = 1
    train_slice = train_slice.rename(columns={"loan_amount": "repeat_loan_amount", "plannum": "repeat_plannum", "count": "repeat_count"})
    test_slice = test_slice.rename(columns={"loan_amount": "repeat_loan_amount", "plannum": "repeat_plannum", "count": "repeat_count"})

    train_max_slice = train_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).max()
    test_max_slice = test_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).max()
    train_max_slice = train_max_slice.merge(train_slice, on=["uid", "repeat_count"], how="left")
    test_max_slice = test_max_slice.merge(test_slice, on=["uid", "repeat_count"], how="left")

    choose_train_max_slice = train_max_slice[["uid", "is_repeat_loan"]].drop_duplicates()
    choose_test_max_slice = test_max_slice[["uid", "is_repeat_loan"]].drop_duplicates()

    repeat_count_train = train_max_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).sum()
    repeat_count_test = test_max_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).sum()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_count_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_count_test, on="uid", how="left")

    repeat_loan_amount_train = train_max_slice[["uid", "repeat_loan_amount"]].groupby(["uid"], as_index=False).mean()
    repeat_loan_amount_test = test_max_slice[["uid", "repeat_loan_amount"]].groupby(["uid"], as_index=False).mean()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_loan_amount_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_loan_amount_test, on="uid", how="left")

    repeat_plannum_train = train_max_slice[["uid", "repeat_plannum"]].groupby(["uid"], as_index=False).mean()
    repeat_plannum_test = test_max_slice[["uid", "repeat_plannum"]].groupby(["uid"], as_index=False).mean()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_plannum_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_plannum_test, on="uid", how="left")

    train_feat = user_df.copy(deep=True)[["uid"]]
    test_feat = user_df.copy(deep=True)[["uid"]]
    train_feat = train_feat.merge(choose_train_max_slice, on="uid", how="left").fillna(-1)
    test_feat = test_feat.merge(choose_test_max_slice, on="uid", how="left").fillna(-1)

    print(train_feat.columns.values)
    train_feat.to_csv("{}/amount_plannum_repeat_loan_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/amount_plannum_repeat_loan_test.csv".format(save_root), index=False)

def loan_weekend_static_by_day(days):
    user = pd.read_csv(user_path)[["uid"]]
    loan = pd.read_csv(loan_path)
    loan.loan_time = pd.to_datetime(loan.loan_time)
    loan["week"] = loan.loan_time.dt.weekday

    loan_time = loan.loan_time

    weekend_day = ((loan.week == 6) | (loan.week == 7))
    loan["is_weekend"] = 0
    loan.ix[weekend_day, "is_weekend"] = 1

    def date_minus_days(start_date, days):
        end_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') - timedelta(days=days)
        end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
        return end_date
    train_time = ((loan_time >= date_minus_days(last_month_end_train, days)) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= date_minus_days(last_month_end_test, days)) & (loan_time < last_month_end_test))

    loan_train = loan.ix[(train_time & weekend_day)]
    loan_test = loan.ix[(test_time & weekend_day)]

    loan_user_train = loan_train[["uid", "is_weekend"]].rename(columns={"is_weekend": "last_{}_days_weekend_has_loan".format(days)}).drop_duplicates()
    loan_user_test = loan_test[["uid", "is_weekend"]].rename(columns={"is_weekend": "last_{}_days_weekend_has_loan".format(days)}).drop_duplicates()

    loan_times_train = loan_train.groupby("uid", as_index=False)["is_weekend"].count().rename(columns={"is_weekend": "last_{}_days_weekend_loan_times".format(days)})
    loan_times_test = loan_test.groupby("uid", as_index=False)["is_weekend"].count().rename(columns={"is_weekend": "last_{}_days_weekend_loan_times".format(days)})
    loan_user_train = loan_user_train.merge(loan_times_train, on="uid", how="left")
    loan_user_test = loan_user_test.merge(loan_times_test, on="uid", how="left")

    loan_amount_sum_train = loan_train.groupby("uid", as_index=False)["loan_amount"].sum().rename(columns={"loan_amount": "last_{}_days_weekend_loan_amount_sum".format(days)})
    loan_amount_sum_test = loan_test.groupby("uid", as_index=False)["loan_amount"].sum().rename(columns={"loan_amount": "last_{}_days_weekend_loan_amount_sum".format(days)})
    loan_user_train = loan_user_train.merge(loan_amount_sum_train, on="uid", how="left")
    loan_user_test = loan_user_test.merge(loan_amount_sum_test, on="uid", how="left")

    loan_plannum_sum_train = loan_train.groupby("uid", as_index=False)["plannum"].sum().rename(columns={"plannum": "last_{}_days_weekend_plannum_sum".format(days)})
    loan_plannum_sum_test = loan_test.groupby("uid", as_index=False)["plannum"].sum().rename(columns={"plannum": "last_{}_days_weekend_plannum_sum".format(days)})
    loan_user_train = loan_user_train.merge(loan_plannum_sum_train, on="uid", how="left")
    loan_user_test = loan_user_test.merge(loan_plannum_sum_test, on="uid", how="left")

    loan_user_train["last_{}_days_weekend_loan_amount_mean".format(days)] = loan_user_train["last_{}_days_weekend_loan_amount_sum".format(days)] / loan_user_train["last_{}_days_weekend_loan_times".format(days)]
    loan_user_test["last_{}_days_weekend_loan_amount_mean".format(days)] = loan_user_test["last_{}_days_weekend_loan_amount_sum".format(days)] / loan_user_test["last_{}_days_weekend_loan_times".format(days)]
    loan_user_train["last_{}_days_weekend_plannum_mean".format(days)] = loan_user_train["last_{}_days_weekend_plannum_sum".format(days)] / loan_user_train["last_{}_days_weekend_loan_times".format(days)]
    loan_user_test["last_{}_days_weekend_plannum_mean".format(days)] = loan_user_test["last_{}_days_weekend_plannum_sum".format(days)] / loan_user_test["last_{}_days_weekend_loan_times".format(days)]

    train_feat = user.copy(deep=True)
    test_feat = user.copy(deep=True)

    train_feat = train_feat.merge(loan_user_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_user_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/loan_weekend_static_by_{}_day_train.csv".format(save_root, days), index=False)
    test_feat.to_csv("{}/loan_weekend_static_by_{}_day_test.csv".format(save_root, days), index=False)

def month_loan_sum():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)
    loan_sum = pd.read_csv(loan_sum_path)

    loan_time = loan_df.loan_time
    last_month_train_time_slice = ( (loan_time >= last_month_start_train) & (loan_time < last_month_end_train) )
    last_month_test_time_slice = ( (loan_time >= last_month_start_test) & (loan_time < last_month_end_test) )

    last_month_train_slice = loan_df.ix[last_month_train_time_slice, ["uid", "loan_amount"]]
    last_month_test_slice = loan_df.ix[last_month_test_time_slice, ["uid", "loan_amount"]]
    # print(train_slice.head())
    last_month_train_slice["loan_amount"] = last_month_train_slice["loan_amount"].apply(lambda x: 5**x-1)
    last_month_test_slice["loan_amount"] = last_month_test_slice["loan_amount"].apply(lambda x: 5**x-1)
    # print(train_slice.head())
    last_month_train_slice = last_month_train_slice.groupby("uid").sum().reset_index().rename(columns={"loan_amount": "loan_sum"})
    last_month_train_slice["loan_sum"] = last_month_train_slice["loan_sum"].apply(lambda x: math.log(x+1)/math.log(5))
    # print(train_slice.head())
    last_month_test_slice = last_month_test_slice.groupby("uid").sum().reset_index().rename(columns={"loan_amount": "loan_sum"})
    last_month_test_slice["loan_sum"] = last_month_test_slice["loan_sum"].apply(lambda x: math.log(x + 1) / math.log(5))
    last_month_train_slice = last_month_train_slice.rename(columns={"loan_sum": "last_month_loan_sum"})
    last_month_test_slice = last_month_test_slice.rename(columns={"loan_sum": "last_month_loan_sum"})

    last_second_month_train_time_slice = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    last_second_month_test_time_slice = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))
    last_second_month_train_slice = loan_df.ix[last_second_month_train_time_slice, ["uid", "loan_amount"]]
    last_second_month_test_slice = loan_df.ix[last_second_month_test_time_slice, ["uid", "loan_amount"]]
    # print(train_slice.head())
    last_second_month_train_slice["loan_amount"] = last_second_month_train_slice["loan_amount"].apply(lambda x: 5 ** x - 1)
    last_second_month_test_slice["loan_amount"] = last_second_month_test_slice["loan_amount"].apply(lambda x: 5 ** x - 1)
    # print(train_slice.head())
    last_second_month_train_slice = last_second_month_train_slice.groupby("uid").sum().reset_index().rename(
        columns={"loan_amount": "loan_sum"})
    last_second_month_train_slice["loan_sum"] = last_second_month_train_slice["loan_sum"].apply(
        lambda x: math.log(x + 1) / math.log(5))
    # print(train_slice.head())
    last_second_month_test_slice = last_second_month_test_slice.groupby("uid").sum().reset_index().rename(
        columns={"loan_amount": "loan_sum"})
    last_second_month_test_slice["loan_sum"] = last_second_month_test_slice["loan_sum"].apply(lambda x: math.log(x + 1) / math.log(5))
    last_second_month_train_slice = last_second_month_train_slice.rename(columns={"loan_sum": "last_second_month_loan_sum"})
    last_second_month_test_slice = last_second_month_test_slice.rename(columns={"loan_sum": "last_second_month_loan_sum"})

    last_third_month_train_time_slice = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    last_third_month_test_time_slice = ((loan_time >= last_three_month_correct_test) & (loan_time < last_month_end_test))
    last_third_month_train_slice = loan_df.ix[last_third_month_train_time_slice, ["uid", "loan_amount"]]
    last_third_month_test_slice = loan_df.ix[last_third_month_test_time_slice, ["uid", "loan_amount"]]
    # print(train_slice.head())
    last_third_month_train_slice["loan_amount"] = last_third_month_train_slice["loan_amount"].apply(
        lambda x: 5 ** x - 1)
    last_third_month_test_slice["loan_amount"] = last_third_month_test_slice["loan_amount"].apply(
        lambda x: 5 ** x - 1)
    # print(train_slice.head())
    last_third_month_train_slice = last_third_month_train_slice.groupby("uid").sum().reset_index().rename(
        columns={"loan_amount": "loan_sum"})
    last_third_month_train_slice["loan_sum"] = last_third_month_train_slice["loan_sum"].apply(
        lambda x: math.log(x + 1) / math.log(5))
    # print(train_slice.head())
    last_third_month_test_slice = last_third_month_test_slice.groupby("uid").sum().reset_index().rename(
        columns={"loan_amount": "loan_sum"})
    last_third_month_test_slice["loan_sum"] = last_third_month_test_slice["loan_sum"].apply(
        lambda x: math.log(x + 1) / math.log(5))
    last_third_month_train_slice = last_third_month_train_slice.rename(
        columns={"loan_sum": "last_third_month_loan_sum"})
    last_third_month_test_slice = last_third_month_test_slice.rename(
        columns={"loan_sum": "last_third_month_loan_sum"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(last_month_train_slice, on="uid", how="left")
    test_feat = test_feat.merge(last_month_test_slice, on="uid", how="left")
    train_feat = train_feat.merge(last_second_month_train_slice, on="uid", how="left")
    test_feat = test_feat.merge(last_second_month_test_slice, on="uid", how="left")
    train_feat = train_feat.merge(last_third_month_train_slice, on="uid", how="left")
    test_feat = test_feat.merge(last_third_month_test_slice, on="uid", how="left")
    print(train_feat.head())
    print(test_feat.head())

    train_feat.fillna(0).to_csv("{}/month_loan_sum_train.csv".format(save_root), index=False)
    test_feat.fillna(0).to_csv("{}/month_loan_sum_test.csv".format(save_root), index=False)

def last_two_month_loan_div_plannum_amount():
    user_df = pd.read_csv(user_path)[["uid"]]
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    train_loan_nums = train_slice.groupby(["uid"]).sum().reset_index()
    test_loan_nums = test_slice.groupby(["uid"]).sum().reset_index()

    train_loan_nums["last_two_month_loan_div_plannum_amount"] = train_loan_nums["loan_amount"] / (train_loan_nums["plannum"]+0)
    test_loan_nums["last_two_month_loan_div_plannum_amount"] = test_loan_nums["loan_amount"] / (test_loan_nums["plannum"]+0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_loan_nums[["uid", "last_two_month_loan_div_plannum_amount"]], on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_loan_nums[["uid", "last_two_month_loan_div_plannum_amount"]], on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_loan_div_plannum_amount_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_loan_div_plannum_amount_test.csv".format(save_root), index=False)

def last_month_repeat_loan():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    train_slice["count"] = 1
    test_slice["count"] = 1
    train_slice = train_slice.groupby(["uid", "loan_amount", "plannum"], as_index=False).count()
    test_slice = test_slice.groupby(["uid", "loan_amount", "plannum"], as_index=False).count()

    train_slice = train_slice[train_slice["count"] > 1]
    test_slice = test_slice[test_slice["count"] > 1]
    train_slice["is_repeat_loan"] = 1
    test_slice["is_repeat_loan"] = 1
    train_slice = train_slice.rename(columns={"loan_amount": "repeat_loan_amount", "plannum": "repeat_plannum", "count": "repeat_count"})
    test_slice = test_slice.rename(columns={"loan_amount": "repeat_loan_amount", "plannum": "repeat_plannum", "count": "repeat_count"})

    train_max_slice = train_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).max()
    test_max_slice = test_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).max()
    train_max_slice = train_max_slice.merge(train_slice, on=["uid", "repeat_count"], how="left")
    test_max_slice = test_max_slice.merge(test_slice, on=["uid", "repeat_count"], how="left")

    choose_train_max_slice = train_max_slice[["uid", "is_repeat_loan"]].drop_duplicates()
    choose_test_max_slice = test_max_slice[["uid", "is_repeat_loan"]].drop_duplicates()

    repeat_count_train = train_max_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).sum()
    repeat_count_test = test_max_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).sum()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_count_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_count_test, on="uid", how="left")

    repeat_loan_amount_train = train_max_slice[["uid", "repeat_loan_amount"]].groupby(["uid"], as_index=False).mean()
    repeat_loan_amount_test = test_max_slice[["uid", "repeat_loan_amount"]].groupby(["uid"], as_index=False).mean()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_loan_amount_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_loan_amount_test, on="uid", how="left")

    repeat_plannum_train = train_max_slice[["uid", "repeat_plannum"]].groupby(["uid"], as_index=False).mean()
    repeat_plannum_test = test_max_slice[["uid", "repeat_plannum"]].groupby(["uid"], as_index=False).mean()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_plannum_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_plannum_test, on="uid", how="left")

    train_feat = user_df.copy(deep=True)[["uid"]]
    test_feat = user_df.copy(deep=True)[["uid"]]
    train_feat = train_feat.merge(choose_train_max_slice, on="uid", how="left").fillna(-1)
    test_feat = test_feat.merge(choose_test_max_slice, on="uid", how="left").fillna(-1)

    train_feat = train_feat.rename(columns={"repeat_loan_amount": "last_month_repeat_loan_amount", "repeat_plannum": "last_month_repeat_plannum", "repeat_count": "last_month_repeat_count", "is_repeat_loan": "last_month_is_repeat_loan"})
    test_feat = test_feat.rename(columns={"repeat_loan_amount": "last_month_repeat_loan_amount", "repeat_plannum": "last_month_repeat_plannum", "repeat_count": "last_month_repeat_count", "is_repeat_loan": "last_month_is_repeat_loan"})
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_month_repeat_loan_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_repeat_loan_test.csv".format(save_root), index=False)

def last_two_month_repeat_loan():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    train_slice["count"] = 1
    test_slice["count"] = 1
    train_slice = train_slice.groupby(["uid", "loan_amount", "plannum"], as_index=False).count()
    test_slice = test_slice.groupby(["uid", "loan_amount", "plannum"], as_index=False).count()

    train_slice = train_slice[train_slice["count"] > 1]
    test_slice = test_slice[test_slice["count"] > 1]
    train_slice["is_repeat_loan"] = 1
    test_slice["is_repeat_loan"] = 1
    train_slice = train_slice.rename(columns={"loan_amount": "repeat_loan_amount", "plannum": "repeat_plannum", "count": "repeat_count"})
    test_slice = test_slice.rename(columns={"loan_amount": "repeat_loan_amount", "plannum": "repeat_plannum", "count": "repeat_count"})

    train_max_slice = train_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).max()
    test_max_slice = test_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).max()
    train_max_slice = train_max_slice.merge(train_slice, on=["uid", "repeat_count"], how="left")
    test_max_slice = test_max_slice.merge(test_slice, on=["uid", "repeat_count"], how="left")

    choose_train_max_slice = train_max_slice[["uid", "is_repeat_loan"]].drop_duplicates()
    choose_test_max_slice = test_max_slice[["uid", "is_repeat_loan"]].drop_duplicates()

    repeat_count_train = train_max_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).sum()
    repeat_count_test = test_max_slice[["uid", "repeat_count"]].groupby(["uid"], as_index=False).sum()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_count_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_count_test, on="uid", how="left")

    repeat_loan_amount_train = train_max_slice[["uid", "repeat_loan_amount"]].groupby(["uid"], as_index=False).mean()
    repeat_loan_amount_test = test_max_slice[["uid", "repeat_loan_amount"]].groupby(["uid"], as_index=False).mean()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_loan_amount_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_loan_amount_test, on="uid", how="left")

    repeat_plannum_train = train_max_slice[["uid", "repeat_plannum"]].groupby(["uid"], as_index=False).mean()
    repeat_plannum_test = test_max_slice[["uid", "repeat_plannum"]].groupby(["uid"], as_index=False).mean()
    choose_train_max_slice = choose_train_max_slice.merge(repeat_plannum_train, on="uid", how="left")
    choose_test_max_slice = choose_test_max_slice.merge(repeat_plannum_test, on="uid", how="left")

    train_feat = user_df.copy(deep=True)[["uid"]]
    test_feat = user_df.copy(deep=True)[["uid"]]
    train_feat = train_feat.merge(choose_train_max_slice, on="uid", how="left").fillna(-1)
    test_feat = test_feat.merge(choose_test_max_slice, on="uid", how="left").fillna(-1)

    train_feat = train_feat.rename(columns={"repeat_loan_amount": "last_two_month_repeat_loan_amount", "repeat_plannum": "last_two_month_repeat_plannum", "repeat_count": "last_two_month_repeat_count", "is_repeat_loan": "last_two_month_is_repeat_loan"})
    test_feat = test_feat.rename(columns={"repeat_loan_amount": "last_two_month_repeat_loan_amount", "repeat_plannum": "last_two_month_repeat_plannum", "repeat_count": "last_two_month_repeat_count", "is_repeat_loan": "last_two_month_is_repeat_loan"})
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_two_month_repeat_loan_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_repeat_loan_test.csv".format(save_root), index=False)

def last_month_continue_loan():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum", "loan_time"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum", "loan_time"]]

    train_continue = train_slice.groupby(["uid", "loan_amount", "plannum"]).count().rename(columns={"loan_time": "continue_loan"})
    test_continue = test_slice.groupby(["uid", "loan_amount", "plannum"]).count().rename(columns={"loan_time": "continue_loan"})
    train_continue["uid"] = train_continue.index.get_level_values("uid")
    train_continue["loan_amount"] = train_continue.index.get_level_values("loan_amount")
    train_continue["plannum"] = train_continue.index.get_level_values("plannum")
    train_continue["continue_loan"] = train_continue["continue_loan"].apply(lambda x: 1 if x >= 2 else 0)

    test_continue["uid"] = test_continue.index.get_level_values("uid")
    test_continue["loan_amount"] = test_continue.index.get_level_values("loan_amount")
    test_continue["plannum"] = test_continue.index.get_level_values("plannum")
    test_continue["continue_loan"] = test_continue["continue_loan"].apply(lambda x: 1 if x >= 2 else 0)

    train_slice = train_slice.merge(train_continue.reset_index(drop=True), on=["uid", "loan_amount", "plannum"], how="left")
    test_slice = test_slice.merge(train_continue.reset_index(drop=True), on=["uid", "loan_amount", "plannum"], how="left")
    train_slice = train_slice.ix[train_slice.continue_loan > 0, ["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0).drop_duplicates()
    test_slice = test_slice.ix[test_slice.continue_loan > 0, ["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0).drop_duplicates()

    continue_loan_train = train_slice[["uid", "continue_loan"]].drop_duplicates()
    continue_loan_test = test_slice[["uid", "continue_loan"]].drop_duplicates()
    continue_loan_amount_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().drop_duplicates()
    continue_loan_amount_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().drop_duplicates()
    continue_plannum_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().drop_duplicates()
    continue_plannum_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().drop_duplicates()

    train_feat = train_feat.merge(continue_loan_train, on="uid", how="left")
    train_feat = train_feat.merge(continue_loan_amount_train, on="uid", how="left")
    train_feat = train_feat.merge(continue_plannum_train, on="uid", how="left")

    test_feat = test_feat.merge(continue_loan_test, on="uid", how="left")
    test_feat = test_feat.merge(continue_loan_amount_test, on="uid", how="left")
    test_feat = test_feat.merge(continue_plannum_test, on="uid", how="left")

    train_feat = train_feat[["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0)
    test_feat = test_feat[["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0)
    train_feat = train_feat.rename(columns={"loan_amount": "last_month_continue_loan_amount", "plannum": "last_month_continue_plannum", "continue_loan": "last_month_continue_loan"})
    test_feat = test_feat.rename(columns={"loan_amount": "last_month_continue_loan_amount", "plannum": "last_month_continue_plannum", "continue_loan": "last_month_continue_loan"})

    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    print(len(set(train_feat.uid)))
    print(len(set(test_feat.uid)))
    # print(test_feat.columns.values)
    train_feat.to_csv("{}/last_month_continue_loan_train.csv".format(save_root))
    test_feat.to_csv("{}/last_month_continue_loan_test.csv".format(save_root))

def last_two_month_continue_loan():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum", "loan_time"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum", "loan_time"]]

    train_continue = train_slice.groupby(["uid", "loan_amount", "plannum"]).count().rename(columns={"loan_time": "continue_loan"})
    test_continue = test_slice.groupby(["uid", "loan_amount", "plannum"]).count().rename(columns={"loan_time": "continue_loan"})
    train_continue["uid"] = train_continue.index.get_level_values("uid")
    train_continue["loan_amount"] = train_continue.index.get_level_values("loan_amount")
    train_continue["plannum"] = train_continue.index.get_level_values("plannum")
    train_continue["continue_loan"] = train_continue["continue_loan"].apply(lambda x: 1 if x >= 2 else 0)

    test_continue["uid"] = test_continue.index.get_level_values("uid")
    test_continue["loan_amount"] = test_continue.index.get_level_values("loan_amount")
    test_continue["plannum"] = test_continue.index.get_level_values("plannum")
    test_continue["continue_loan"] = test_continue["continue_loan"].apply(lambda x: 1 if x >= 2 else 0)

    train_slice = train_slice.merge(train_continue.reset_index(drop=True), on=["uid", "loan_amount", "plannum"], how="left")
    test_slice = test_slice.merge(train_continue.reset_index(drop=True), on=["uid", "loan_amount", "plannum"], how="left")
    train_slice = train_slice.ix[train_slice.continue_loan > 0, ["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0).drop_duplicates()
    test_slice = test_slice.ix[test_slice.continue_loan > 0, ["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0).drop_duplicates()

    continue_loan_train = train_slice[["uid", "continue_loan"]].drop_duplicates()
    continue_loan_test = test_slice[["uid", "continue_loan"]].drop_duplicates()
    continue_loan_amount_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().drop_duplicates()
    continue_loan_amount_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().drop_duplicates()
    continue_plannum_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().drop_duplicates()
    continue_plannum_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().drop_duplicates()

    train_feat = train_feat.merge(continue_loan_train, on="uid", how="left")
    train_feat = train_feat.merge(continue_loan_amount_train, on="uid", how="left")
    train_feat = train_feat.merge(continue_plannum_train, on="uid", how="left")

    test_feat = test_feat.merge(continue_loan_test, on="uid", how="left")
    test_feat = test_feat.merge(continue_loan_amount_test, on="uid", how="left")
    test_feat = test_feat.merge(continue_plannum_test, on="uid", how="left")

    train_feat = train_feat[["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0)
    test_feat = test_feat[["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0)
    train_feat = train_feat.rename(columns={"loan_amount": "last_two_month_continue_loan_amount", "plannum": "last_two_month_continue_plannum", "continue_loan": "last_two_month_continue_loan"})
    test_feat = test_feat.rename(columns={"loan_amount": "last_two_month_continue_loan_amount", "plannum": "last_two_month_continue_plannum", "continue_loan": "last_two_month_continue_loan"})

    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    print(len(set(train_feat.uid)))
    print(len(set(test_feat.uid)))
    # print(test_feat.columns.values)
    train_feat.to_csv("{}/last_two_month_continue_loan_train.csv".format(save_root))
    test_feat.to_csv("{}/last_two_month_continue_loan_test.csv".format(save_root))

def last_three_month_continue_loan():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum", "loan_time"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum", "loan_time"]]

    train_continue = train_slice.groupby(["uid", "loan_amount", "plannum"]).count().rename(columns={"loan_time": "continue_loan"})
    test_continue = test_slice.groupby(["uid", "loan_amount", "plannum"]).count().rename(columns={"loan_time": "continue_loan"})
    train_continue["uid"] = train_continue.index.get_level_values("uid")
    train_continue["loan_amount"] = train_continue.index.get_level_values("loan_amount")
    train_continue["plannum"] = train_continue.index.get_level_values("plannum")
    train_continue["continue_loan"] = train_continue["continue_loan"].apply(lambda x: 1 if x >= 2 else 0)

    test_continue["uid"] = test_continue.index.get_level_values("uid")
    test_continue["loan_amount"] = test_continue.index.get_level_values("loan_amount")
    test_continue["plannum"] = test_continue.index.get_level_values("plannum")
    test_continue["continue_loan"] = test_continue["continue_loan"].apply(lambda x: 1 if x >= 2 else 0)

    train_slice = train_slice.merge(train_continue.reset_index(drop=True), on=["uid", "loan_amount", "plannum"], how="left")
    test_slice = test_slice.merge(train_continue.reset_index(drop=True), on=["uid", "loan_amount", "plannum"], how="left")
    train_slice = train_slice.ix[train_slice.continue_loan > 0, ["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0).drop_duplicates()
    test_slice = test_slice.ix[test_slice.continue_loan > 0, ["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0).drop_duplicates()

    continue_loan_train = train_slice[["uid", "continue_loan"]].drop_duplicates()
    continue_loan_test = test_slice[["uid", "continue_loan"]].drop_duplicates()
    continue_loan_amount_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().drop_duplicates()
    continue_loan_amount_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().drop_duplicates()
    continue_plannum_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().drop_duplicates()
    continue_plannum_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().drop_duplicates()

    train_feat = train_feat.merge(continue_loan_train, on="uid", how="left")
    train_feat = train_feat.merge(continue_loan_amount_train, on="uid", how="left")
    train_feat = train_feat.merge(continue_plannum_train, on="uid", how="left")

    test_feat = test_feat.merge(continue_loan_test, on="uid", how="left")
    test_feat = test_feat.merge(continue_loan_amount_test, on="uid", how="left")
    test_feat = test_feat.merge(continue_plannum_test, on="uid", how="left")

    train_feat = train_feat[["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0)
    test_feat = test_feat[["uid", "loan_amount", "plannum", "continue_loan"]].fillna(0)
    train_feat = train_feat.rename(columns={"loan_amount": "last_three_month_continue_loan_amount", "plannum": "last_three_month_continue_plannum", "continue_loan": "last_three_month_continue_loan"})
    test_feat = test_feat.rename(columns={"loan_amount": "last_three_month_continue_loan_amount", "plannum": "last_three_month_continue_plannum", "continue_loan": "last_three_month_continue_loan"})

    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    print(len(set(train_feat.uid)))
    print(len(set(test_feat.uid)))
    # print(test_feat.columns.values)
    train_feat.to_csv("{}/last_three_month_continue_loan_train.csv".format(save_root))
    test_feat.to_csv("{}/last_three_month_continue_loan_test.csv".format(save_root))

def get_loan_nday_count(n_day, period):
    def date_minus_days(start_date, days):
        end_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=days)
        end_date = end_date.strftime('%Y-%m-%d')
        return end_date
    def diff_of_days(day1, day2):
        days = (datetime.strptime(day1, '%Y-%m-%d') - datetime.strptime(day2, '%Y-%m-%d')).days
        return abs(days)

    user = pd.read_csv(user_path)
    loan = pd.read_csv(loan_path)
    train_feat = user[["uid"]].copy(deep=True)
    test_feat = user[["uid"]].copy(deep=True)

    loan_time = loan.loan_time
    train_time = ((loan_time >= date_minus_days(last_month_end_for_click_train, period)) & (loan_time < last_month_end_for_click_train))
    test_time = ((loan_time >= date_minus_days(last_month_end_for_click_test, period)) & (loan_time < last_month_end_for_click_test))

    train_slice = loan.ix[train_time]
    test_slice = loan.ix[test_time]
    train_slice["loan_time"] = train_slice["loan_time"].apply(lambda x: diff_of_days(last_month_end_for_click_train, x[:10])//n_day)
    test_slice["loan_time"] = test_slice["loan_time"].apply(lambda x: diff_of_days(last_month_end_for_click_test, x[:10]) // n_day)

    loan_day_count_train = train_slice.groupby('uid',as_index=False)['loan_time'].agg({'loan_{0}day_count{1}'.format(n_day, period):'nunique'})
    loan_day_count_test = test_slice.groupby('uid', as_index=False)['loan_time'].agg({'loan_{0}day_count{1}'.format(n_day, period): 'nunique'})
    train_feat = train_feat.merge(loan_day_count_train,on='uid',how='left').fillna(0)
    test_feat = test_feat.merge(loan_day_count_test, on='uid', how='left').fillna(0)
    print(len(train_feat))
    print(len(test_feat))
    print(train_feat.columns.values)
    name = train_feat.columns.values[-1]
    train_feat.to_csv("{}/{}_train.csv".format(save_root, name), index=False)
    test_feat.to_csv("{}/{}_test.csv".format(save_root, name), index=False)

def loan_left():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time
    '''********** last third month loan ***********'''
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_two_month_start_train))
    test_time = ((loan_time >= last_three_month_correct_test) & (loan_time < last_two_month_start_test))
    last_third_month_train_slice = loan_df.ix[train_time]
    last_third_month_test_slice = loan_df.ix[test_time]
    last_third_month_loan_left_train = last_third_month_train_slice.ix[last_third_month_train_slice.plannum > 2, ["uid", "loan_amount", "plannum"]]
    last_third_month_loan_left_test = last_third_month_test_slice.ix[last_third_month_test_slice.plannum > 2, ["uid", "loan_amount", "plannum"]]

    last_third_month_loan_left_train["third_month_left_loan"] = last_third_month_loan_left_train["loan_amount"] * (last_third_month_loan_left_train["plannum"]-2) / last_third_month_loan_left_train["plannum"]
    last_third_month_loan_left_test["third_month_left_loan"] = last_third_month_loan_left_test["loan_amount"] * (last_third_month_loan_left_test["plannum"] - 2) / last_third_month_loan_left_test["plannum"]
    '''********** last second month loan ***********'''
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_start_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_start_test))
    last_second_month_train_slice = loan_df.ix[train_time]
    last_second_month_test_slice = loan_df.ix[test_time]
    last_second_month_loan_left_train = last_second_month_train_slice.ix[
        last_second_month_train_slice.plannum > 1, ["uid", "loan_amount", "plannum"]]
    last_second_month_loan_left_test = last_second_month_test_slice.ix[
        last_second_month_test_slice.plannum > 1, ["uid", "loan_amount", "plannum"]]

    last_second_month_loan_left_train["second_month_left_loan"] = last_second_month_loan_left_train["loan_amount"] * (last_second_month_loan_left_train["plannum"]-1) / last_second_month_loan_left_train["plannum"]
    last_second_month_loan_left_test["second_month_left_loan"] = last_second_month_loan_left_test["loan_amount"] * (last_second_month_loan_left_test["plannum"] - 1) / last_second_month_loan_left_test["plannum"]
    '''********** last first month loan ***********'''
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))
    last_month_train_slice = loan_df.ix[train_time]
    last_month_test_slice = loan_df.ix[test_time]
    last_month_train_slice = last_month_train_slice.rename(columns={"loan_amount": "first_month_left_loan"})
    last_month_test_slice = last_month_test_slice.rename(columns={"loan_amount": "first_month_left_loan"})

    train_feat = train_feat.merge(last_third_month_loan_left_train[["uid", "third_month_left_loan"]], on="uid", how="left")
    train_feat = train_feat.merge(last_second_month_loan_left_train[["uid", "second_month_left_loan"]], on="uid", how="left")
    train_feat = train_feat.merge(last_month_train_slice[["uid", "first_month_left_loan"]], on="uid", how="left").fillna(0)

    test_feat = test_feat.merge(last_third_month_loan_left_test[["uid", "third_month_left_loan"]], on="uid", how="left")
    test_feat = test_feat.merge(last_second_month_loan_left_test[["uid", "second_month_left_loan"]], on="uid", how="left")
    test_feat = test_feat.merge(last_month_test_slice[["uid", "first_month_left_loan"]], on="uid", how="left").fillna(0)

    train_return = train_feat[["uid"]].copy(deep=True)
    test_return = test_feat[["uid"]].copy(deep=True)

    first_month_left_loan_sum_train = train_feat[["uid", "first_month_left_loan"]].fillna(0).groupby("uid", as_index=False).sum()
    second_month_left_loan_sum_train = train_feat[["uid", "second_month_left_loan"]].fillna(0).groupby("uid", as_index=False).sum()
    third_month_left_loan_sum_train = train_feat[["uid", "third_month_left_loan"]].fillna(0).groupby("uid", as_index=False).sum()
    train_return = train_return.merge(first_month_left_loan_sum_train, on="uid", how="left")
    train_return = train_return.merge(second_month_left_loan_sum_train, on="uid", how="left")
    train_return = train_return.merge(third_month_left_loan_sum_train, on="uid", how="left").drop_duplicates()

    first_month_left_loan_sum_test = test_feat[["uid", "first_month_left_loan"]].fillna(0).groupby("uid", as_index=False).sum()
    second_month_left_loan_sum_test = test_feat[["uid", "second_month_left_loan"]].fillna(0).groupby("uid", as_index=False).sum()
    third_month_left_loan_sum_test = test_feat[["uid", "third_month_left_loan"]].fillna(0).groupby("uid", as_index=False).sum()
    test_return = test_return.merge(first_month_left_loan_sum_test, on="uid", how="left")
    test_return = test_return.merge(second_month_left_loan_sum_test, on="uid", how="left")
    test_return = test_return.merge(third_month_left_loan_sum_test, on="uid", how="left").drop_duplicates()

    train_return["left_loan"] = train_return["first_month_left_loan"] + train_return["second_month_left_loan"] + train_return["third_month_left_loan"]
    test_return["left_loan"] = test_return["first_month_left_loan"] + test_return["second_month_left_loan"] + test_return["third_month_left_loan"]
    print(train_return.head())
    print(test_return.head())
    print(len(train_return))
    print(len(test_return))
    print(train_return.columns.values)
    train_return.to_csv("{}/loan_left_train.csv".format(save_root), index=False)
    test_return.to_csv("{}/loan_left_test.csv".format(save_root), index=False)

def last_second_month_loan_feat():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df[["uid"]].copy(deep=True)
    test_feat = user_df[["uid"]].copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_start_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_start_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    print(train_slice.head())
    print(test_slice.head())

    loan_sum_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).sum().rename(columns={"loan_amount": "loan_sum"})
    loan_mean_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_mean"})
    loan_median_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).median().rename(columns={"loan_amount": "loan_median"})
    loan_nums_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "loan_nums"})
    plannum_sum_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).sum().rename(columns={"plannum": "plannum_sum"})
    plannum_mean_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).mean().rename(columns={"plannum": "plannum_mean"})
    plannum_median_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().rename(columns={"plannum": "plannum_median"})

    loan_sum_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).sum().rename(columns={"loan_amount": "loan_sum"})
    loan_mean_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_mean"})
    loan_median_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).median().rename(columns={"loan_amount": "loan_median"})
    loan_nums_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_nums"})
    plannum_sum_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).sum().rename(columns={"plannum": "plannum_sum"})
    plannum_mean_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).mean().rename(columns={"plannum": "plannum_mean"})
    plannum_median_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().rename(columns={"plannum": "plannum_median"})

    train_feat = train_feat.merge(loan_sum_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_mean_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_median_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_nums_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_sum_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_mean_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_median_train, on="uid", how="left")

    test_feat = test_feat.merge(loan_sum_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_mean_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_median_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_nums_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_sum_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_mean_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_median_test, on="uid", how="left")

    train_feat["loan_sum_div_plannum_sum"] = train_feat["loan_sum"] / train_feat["plannum_sum"]
    train_feat["loan_median_div_plannum_median"] = train_feat["loan_median"] / train_feat["plannum_median"]

    test_feat["loan_sum_div_plannum_sum"] = test_feat["loan_sum"] / test_feat["plannum_sum"]
    test_feat["loan_median_div_plannum_median"] = test_feat["loan_median"] / test_feat["plannum_median"]

    train_feat = train_feat.rename(columns={"loan_sum": "last_second_month_loan_amount",
                                            "loan_mean": "last_second_month_loan_mean",
                                            "loan_median": "last_second_month_loan_median",
                                            "loan_nums": "last_second_month_loan_nums",
                                            "plannum_sum": "last_second_month_plannum_sum",
                                            "plannum_mean": "last_second_month_plannum_mean",
                                            "plannum_median": "last_second_month_plannum_median",
                                            "loan_sum_div_plannum_sum": "last_second_month_loan_sum_div_plannum_sum",
                                            "loan_median_div_plannum_median": "last_second_month_loan_median_div_plannum_median"})
    test_feat = test_feat.rename(columns={"loan_sum": "last_second_month_loan_amount",
                                            "loan_mean": "last_second_month_loan_mean",
                                            "loan_median": "last_second_month_loan_median",
                                            "loan_nums": "last_second_month_loan_nums",
                                            "plannum_sum": "last_second_month_plannum_sum",
                                            "plannum_mean": "last_second_month_plannum_mean",
                                            "plannum_median": "last_second_month_plannum_median",
                                            "loan_sum_div_plannum_sum": "last_second_month_loan_sum_div_plannum_sum",
                                            "loan_median_div_plannum_median": "last_second_month_loan_median_div_plannum_median"})
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_second_month_loan_feat_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_second_month_loan_feat_test.csv".format(save_root), index=False)

def last_third_month_loan_feat():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df[["uid"]].copy(deep=True)
    test_feat = user_df[["uid"]].copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_two_month_start_train))
    test_time = ((loan_time >= last_three_month_correct_test) & (loan_time < last_two_month_start_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    print(train_slice.head())
    print(test_slice.head())

    loan_sum_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).sum().rename(columns={"loan_amount": "loan_sum"})
    loan_mean_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_mean"})
    loan_median_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).median().rename(columns={"loan_amount": "loan_median"})
    loan_nums_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "loan_nums"})
    plannum_sum_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).sum().rename(columns={"plannum": "plannum_sum"})
    plannum_mean_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).mean().rename(columns={"plannum": "plannum_mean"})
    plannum_median_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().rename(columns={"plannum": "plannum_median"})

    loan_sum_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).sum().rename(columns={"loan_amount": "loan_sum"})
    loan_mean_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_mean"})
    loan_median_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).median().rename(columns={"loan_amount": "loan_median"})
    loan_nums_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_nums"})
    plannum_sum_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).sum().rename(columns={"plannum": "plannum_sum"})
    plannum_mean_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).mean().rename(columns={"plannum": "plannum_mean"})
    plannum_median_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().rename(columns={"plannum": "plannum_median"})

    train_feat = train_feat.merge(loan_sum_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_mean_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_median_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_nums_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_sum_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_mean_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_median_train, on="uid", how="left")

    test_feat = test_feat.merge(loan_sum_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_mean_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_median_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_nums_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_sum_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_mean_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_median_test, on="uid", how="left")

    train_feat["loan_sum_div_plannum_sum"] = train_feat["loan_sum"] / train_feat["plannum_sum"]
    train_feat["loan_median_div_plannum_median"] = train_feat["loan_median"] / train_feat["plannum_median"]

    test_feat["loan_sum_div_plannum_sum"] = test_feat["loan_sum"] / test_feat["plannum_sum"]
    test_feat["loan_median_div_plannum_median"] = test_feat["loan_median"] / test_feat["plannum_median"]

    train_feat = train_feat.rename(columns={"loan_sum": "last_third_month_loan_amount",
                                            "loan_mean": "last_third_month_loan_mean",
                                            "loan_median": "last_third_month_loan_median",
                                            "loan_nums": "last_third_month_loan_nums",
                                            "plannum_sum": "last_third_month_plannum_sum",
                                            "plannum_mean": "last_third_month_plannum_mean",
                                            "plannum_median": "last_third_month_plannum_median",
                                            "loan_sum_div_plannum_sum": "last_third_month_loan_sum_div_plannum_sum",
                                            "loan_median_div_plannum_median": "last_third_month_loan_median_div_plannum_median"})
    test_feat = test_feat.rename(columns={"loan_sum": "last_third_month_loan_amount",
                                            "loan_mean": "last_third_month_loan_mean",
                                            "loan_median": "last_third_month_loan_median",
                                            "loan_nums": "last_third_month_loan_nums",
                                            "plannum_sum": "last_third_month_plannum_sum",
                                            "plannum_mean": "last_third_month_plannum_mean",
                                            "plannum_median": "last_third_month_plannum_median",
                                            "loan_sum_div_plannum_sum": "last_third_month_loan_sum_div_plannum_sum",
                                            "loan_median_div_plannum_median": "last_third_month_loan_median_div_plannum_median"})
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_third_month_loan_feat_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_third_month_loan_feat_test.csv".format(save_root), index=False)

def last_month_loan_tsfresh():
    settings = EfficientFCParameters()
    user_df = pd.read_csv(user_path)[["uid"]]

    train_feat = user_df[["uid"]].copy(deep=True)
    test_feat = user_df[["uid"]].copy(deep=True)
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time]
    test_slice = loan_df.ix[test_time]

    tsfresh_train = extract_features(train_slice[['uid', "loan_time", 'loan_amount', 'plannum']], column_id="uid", column_sort="loan_time").reset_index()
    tsfresh_test = extract_features(test_slice[['uid', "loan_time", 'loan_amount', 'plannum']], column_id="uid", column_sort="loan_time").reset_index()
    # print(tsfresh_train.head())
    # print(tsfresh_test.head())
    train_feat = train_feat.merge(tsfresh_train, on="uid", how="left")
    test_feat = test_feat.merge(tsfresh_test, on="uid", how="left")
    # print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_month_loan_tsfresh_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_tsfresh_test.csv".format(save_root), index=False)

def last_month_loan_feat():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df[["uid"]].copy(deep=True)
    test_feat = user_df[["uid"]].copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_start_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_start_test))

    train_slice = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    test_slice = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    print(train_slice.head())
    print(test_slice.head())

    loan_sum_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).sum().rename(columns={"loan_amount": "loan_sum"})
    loan_mean_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_mean"})
    loan_median_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).median().rename(columns={"loan_amount": "loan_median"})
    loan_nums_train = train_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "loan_nums"})
    plannum_sum_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).sum().rename(columns={"plannum": "plannum_sum"})
    plannum_mean_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).mean().rename(columns={"plannum": "plannum_mean"})
    plannum_median_train = train_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().rename(columns={"plannum": "plannum_median"})

    loan_sum_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).sum().rename(columns={"loan_amount": "loan_sum"})
    loan_mean_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_mean"})
    loan_median_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).median().rename(columns={"loan_amount": "loan_median"})
    loan_nums_test = test_slice[["uid", "loan_amount"]].groupby(["uid"], as_index=False).mean().rename(columns={"loan_amount": "loan_nums"})
    plannum_sum_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).sum().rename(columns={"plannum": "plannum_sum"})
    plannum_mean_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).mean().rename(columns={"plannum": "plannum_mean"})
    plannum_median_test = test_slice[["uid", "plannum"]].groupby(["uid"], as_index=False).median().rename(columns={"plannum": "plannum_median"})

    train_feat = train_feat.merge(loan_sum_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_mean_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_median_train, on="uid", how="left")
    train_feat = train_feat.merge(loan_nums_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_sum_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_mean_train, on="uid", how="left")
    train_feat = train_feat.merge(plannum_median_train, on="uid", how="left")

    test_feat = test_feat.merge(loan_sum_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_mean_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_median_test, on="uid", how="left")
    test_feat = test_feat.merge(loan_nums_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_sum_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_mean_test, on="uid", how="left")
    test_feat = test_feat.merge(plannum_median_test, on="uid", how="left")

    train_feat["loan_sum_div_plannum_sum"] = train_feat["loan_sum"] / train_feat["plannum_sum"]
    train_feat["loan_median_div_plannum_median"] = train_feat["loan_median"] / train_feat["plannum_median"]

    test_feat["loan_sum_div_plannum_sum"] = test_feat["loan_sum"] / test_feat["plannum_sum"]
    test_feat["loan_median_div_plannum_median"] = test_feat["loan_median"] / test_feat["plannum_median"]

    train_feat = train_feat.rename(columns={"loan_sum": "last_month_loan_amount",
                                            "loan_mean": "last_month_loan_mean",
                                            "loan_median": "last_month_loan_median",
                                            "loan_nums": "last_month_loan_nums",
                                            "plannum_sum": "last_month_plannum_sum",
                                            "plannum_mean": "last_month_plannum_mean",
                                            "plannum_median": "last_month_plannum_median",
                                            "loan_sum_div_plannum_sum": "last_month_loan_sum_div_plannum_sum",
                                            "loan_median_div_plannum_median": "last_month_loan_median_div_plannum_median"})
    test_feat = test_feat.rename(columns={"loan_sum": "last_month_loan_amount",
                                            "loan_mean": "last_month_loan_mean",
                                            "loan_median": "last_month_loan_median",
                                            "loan_nums": "last_month_loan_nums",
                                            "plannum_sum": "last_month_plannum_sum",
                                            "plannum_mean": "last_month_plannum_mean",
                                            "plannum_median": "last_month_plannum_median",
                                            "loan_sum_div_plannum_sum": "last_month_loan_sum_div_plannum_sum",
                                            "loan_median_div_plannum_median": "last_month_loan_median_div_plannum_median"})
    train_feat = train_feat[["uid", "last_month_loan_median", "last_month_plannum_median", "last_month_loan_median_div_plannum_median"]]
    test_feat = test_feat[
        ["uid", "last_month_loan_median", "last_month_plannum_median", "last_month_loan_median_div_plannum_median"]]
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_month_loan_feat_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_feat_test.csv".format(save_root), index=False)

def last_three_month_plannum_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time]
    loan_test = loan_df.ix[test_time]

    loan_train = loan_train.groupby(["uid", "plannum"])["loan_time"].count().unstack().add_prefix('last_three_month_plannum_onehot_').reset_index().fillna(0)
    loan_test = loan_test.groupby(["uid", "plannum"])["loan_time"].count().unstack().add_prefix('last_three_month_plannum_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(loan_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_test, on="uid", how="left").fillna(0)

    print(train_feat.head())
    print(test_feat.head())
    print(len(train_feat))
    print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_three_month_plannum_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_plannum_count_test.csv".format(save_root), index=False)

def last_two_month_plannum_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time]
    loan_test = loan_df.ix[test_time]

    loan_train = loan_train.groupby(["uid", "plannum"])["loan_time"].count().unstack().add_prefix('last_two_month_plannum_onehot_').reset_index().fillna(0)
    loan_test = loan_test.groupby(["uid", "plannum"])["loan_time"].count().unstack().add_prefix('last_two_month_plannum_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(loan_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_test, on="uid", how="left").fillna(0)

    print(train_feat.head())
    print(test_feat.head())
    print(len(train_feat))
    print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_two_month_plannum_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_plannum_count_test.csv".format(save_root), index=False)

def last_month_plannum_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time]
    loan_test = loan_df.ix[test_time]

    loan_train = loan_train.groupby(["uid", "plannum"])["loan_time"].count().unstack().add_prefix('last_month_plannum_onehot_').reset_index().fillna(0)
    loan_test = loan_test.groupby(["uid", "plannum"])["loan_time"].count().unstack().add_prefix('last_month_plannum_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(loan_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_test, on="uid", how="left").fillna(0)

    print(train_feat.head())
    print(test_feat.head())
    print(len(train_feat))
    print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_month_plannum_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_plannum_count_test.csv".format(save_root), index=False)

def last_three_month_loan_amount_per_plannum():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    loan_test = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    loan_train["last_three_month_loan_amount_per_plannum"] = loan_train["loan_amount"] / loan_train["plannum"]
    loan_test["last_three_month_loan_amount_per_plannum"] = loan_test["loan_amount"] / loan_test["plannum"]

    loan_train = loan_train[["uid", "last_three_month_loan_amount_per_plannum"]]
    loan_train = loan_train.groupby("uid", as_index=False)["last_three_month_loan_amount_per_plannum"]\
        .agg({"last_three_month_loan_amount_per_plannum_sum": np.sum,
              "last_three_month_loan_amount_per_plannum_mean": np.mean,
              "last_three_month_loan_amount_per_plannum_median": np.median,
              "last_three_month_loan_amount_per_plannum_min": np.min,
              "last_three_month_loan_amount_per_plannum_max": np.max,
              "last_three_month_loan_amount_per_plannum_var": np.var,
              "last_three_month_loan_amount_per_plannum_count": "count"})
    loan_test = loan_test[["uid", "last_three_month_loan_amount_per_plannum"]]
    loan_test = loan_test.groupby("uid", as_index=False)["last_three_month_loan_amount_per_plannum"]\
        .agg({"last_three_month_loan_amount_per_plannum_sum": np.sum,
              "last_three_month_loan_amount_per_plannum_mean": np.mean,
              "last_three_month_loan_amount_per_plannum_median": np.median,
              "last_three_month_loan_amount_per_plannum_min": np.min,
              "last_three_month_loan_amount_per_plannum_max": np.max,
              "last_three_month_loan_amount_per_plannum_var": np.var,
              "last_three_month_loan_amount_per_plannum_count": "count"})
    train_feat = train_feat.merge(loan_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_test, on="uid", how="left").fillna(0)
    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_three_month_loan_amount_per_plannum_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_amount_per_plannum_test.csv".format(save_root), index=False)

def last_two_month_loan_amount_per_plannum():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    loan_test = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    loan_train["last_two_month_loan_amount_per_plannum"] = loan_train["loan_amount"] / loan_train["plannum"]
    loan_test["last_two_month_loan_amount_per_plannum"] = loan_test["loan_amount"] / loan_test["plannum"]

    loan_train = loan_train[["uid", "last_two_month_loan_amount_per_plannum"]]
    loan_train = loan_train.groupby("uid", as_index=False)["last_two_month_loan_amount_per_plannum"]\
        .agg({"last_two_month_loan_amount_per_plannum_sum": np.sum,
              "last_two_month_loan_amount_per_plannum_mean": np.mean,
              "last_two_month_loan_amount_per_plannum_median": np.median,
              "last_two_month_loan_amount_per_plannum_min": np.min,
              "last_two_month_loan_amount_per_plannum_max": np.max,
              "last_two_month_loan_amount_per_plannum_var": np.var,
              "last_two_month_loan_amount_per_plannum_count": "count"})
    loan_test = loan_test[["uid", "last_two_month_loan_amount_per_plannum"]]
    loan_test = loan_test.groupby("uid", as_index=False)["last_two_month_loan_amount_per_plannum"]\
        .agg({"last_two_month_loan_amount_per_plannum_sum": np.sum,
              "last_two_month_loan_amount_per_plannum_mean": np.mean,
              "last_two_month_loan_amount_per_plannum_median": np.median,
              "last_two_month_loan_amount_per_plannum_min": np.min,
              "last_two_month_loan_amount_per_plannum_max": np.max,
              "last_two_month_loan_amount_per_plannum_var": np.var,
              "last_two_month_loan_amount_per_plannum_count": "count"})
    train_feat = train_feat.merge(loan_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_test, on="uid", how="left").fillna(0)
    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_two_month_loan_amount_per_plannum_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_loan_amount_per_plannum_test.csv".format(save_root), index=False)

def last_month_loan_amount_per_plannum():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    loan_test = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]
    loan_train["last_month_loan_amount_per_plannum"] = loan_train["loan_amount"] / loan_train["plannum"]
    loan_test["last_month_loan_amount_per_plannum"] = loan_test["loan_amount"] / loan_test["plannum"]

    loan_train = loan_train[["uid", "last_month_loan_amount_per_plannum"]]
    loan_train = loan_train.groupby("uid", as_index=False)["last_month_loan_amount_per_plannum"]\
        .agg({"last_month_loan_amount_per_plannum_sum": np.sum,
              "last_month_loan_amount_per_plannum_mean": np.mean,
              "last_month_loan_amount_per_plannum_median": np.median,
              "last_month_loan_amount_per_plannum_min": np.min,
              "last_month_loan_amount_per_plannum_max": np.max,
              "last_month_loan_amount_per_plannum_var": np.var,
              "last_month_loan_amount_per_plannum_count": "count"})
    loan_test = loan_test[["uid", "last_month_loan_amount_per_plannum"]]
    loan_test = loan_test.groupby("uid", as_index=False)["last_month_loan_amount_per_plannum"]\
        .agg({"last_month_loan_amount_per_plannum_sum": np.sum,
              "last_month_loan_amount_per_plannum_mean": np.mean,
              "last_month_loan_amount_per_plannum_median": np.median,
              "last_month_loan_amount_per_plannum_min": np.min,
              "last_month_loan_amount_per_plannum_max": np.max,
              "last_month_loan_amount_per_plannum_var": np.var,
              "last_month_loan_amount_per_plannum_count": "count"})
    train_feat = train_feat.merge(loan_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(loan_test, on="uid", how="left").fillna(0)
    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_month_loan_amount_per_plannum_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_amount_per_plannum_test.csv".format(save_root), index=False)

def finish_repay_nums():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_correct_test) & (loan_time < last_month_end_test))

    loan_train = loan_df.ix[train_time, ["uid", "loan_amount", "plannum"]]
    loan_test = loan_df.ix[test_time, ["uid", "loan_amount", "plannum"]]

    '''********************* Third month finish repay **********************'''
    third_month_slice_train = ((loan_time >= last_three_month_start_train) & (loan_time < last_two_month_start_train) & (loan_df.plannum < 3))
    third_month_slice_test = ((loan_time >= last_three_month_correct_test) & (loan_time < last_two_month_start_test) & (loan_df.plannum < 3))
    '''********************* Second month finish repay **********************'''
    second_month_slice_train = (loan_time >= last_two_month_start_train) & (loan_time < last_month_start_train) & (loan_df.plannum < 2)
    second_month_slice_test = (loan_time >= last_two_month_start_test) & (loan_time < last_month_start_test) & (loan_df.plannum < 2)

    third_month_finish_repay_nums_train = loan_df.ix[third_month_slice_train, ["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "third_month_finish_repay_nums"})
    third_month_finish_repay_nums_test = loan_df.ix[third_month_slice_test, ["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "third_month_finish_repay_nums"})

    second_month_finish_repay_nums_train = loan_df.ix[second_month_slice_train, ["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "second_month_finish_repay_nums"})
    second_month_finish_repay_nums_test = loan_df.ix[second_month_slice_test, ["uid", "loan_amount"]].groupby(["uid"], as_index=False).count().rename(columns={"loan_amount": "second_month_finish_repay_nums"})

    train_feat = train_feat.merge(third_month_finish_repay_nums_train, on="uid", how="left")
    test_feat = test_feat.merge(third_month_finish_repay_nums_test, on="uid", how="left")
    train_feat = train_feat.merge(second_month_finish_repay_nums_train, on="uid", how="left")
    test_feat = test_feat.merge(second_month_finish_repay_nums_test, on="uid", how="left")

    train_feat["all_finish_repay_nums"] = train_feat["second_month_finish_repay_nums"] + train_feat["third_month_finish_repay_nums"]
    test_feat["all_finish_repay_nums"] = test_feat["second_month_finish_repay_nums"] + test_feat["third_month_finish_repay_nums"]

    print(len(train_feat))
    print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.fillna(0).to_csv("{}/finish_repay_nums_train.csv".format(save_root), index=False)
    test_feat.fillna(0).to_csv("{}/finish_repay_nums_test.csv".format(save_root), index=False)

def last_three_month_loan_hour():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_df['day'] = pd.to_datetime(loan_df.loan_time).dt.day
    loan_df['hour'] = pd.to_datetime(loan_df.loan_time).dt.hour
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "day", "hour"]].drop_duplicates()
    test_slice = loan_df.ix[test_time, ["uid", "day", "hour"]].drop_duplicates()
    train_slice = train_slice.groupby("uid", as_index=False)["uid"].agg({"last_three_month_loan_hour": "count"})
    test_slice = test_slice.groupby("uid", as_index=False)["uid"].agg({"last_three_month_loan_hour": "count"})
    train_feat = train_feat.merge(train_slice[["uid", "last_three_month_loan_hour"]], on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_slice[["uid", "last_three_month_loan_hour"]], on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_loan_hour_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_hour_test.csv".format(save_root), index=False)


def last_two_month_loan_hour():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_df['day'] = pd.to_datetime(loan_df.loan_time).dt.day
    loan_df['hour'] = pd.to_datetime(loan_df.loan_time).dt.hour
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "day", "hour"]].drop_duplicates()
    test_slice = loan_df.ix[test_time, ["uid", "day", "hour"]].drop_duplicates()
    train_slice = train_slice.groupby("uid", as_index=False)["uid"].agg({"last_two_month_loan_hour": "count"})
    test_slice = test_slice.groupby("uid", as_index=False)["uid"].agg({"last_two_month_loan_hour": "count"})
    train_feat = train_feat.merge(train_slice[["uid", "last_two_month_loan_hour"]], on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_slice[["uid", "last_two_month_loan_hour"]], on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_loan_hour_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_loan_hour_test.csv".format(save_root), index=False)

def last_month_loan_hour():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    loan_df = pd.read_csv(loan_path)
    loan_df['day'] = pd.to_datetime(loan_df.loan_time).dt.day
    loan_df['hour'] = pd.to_datetime(loan_df.loan_time).dt.hour
    print(len(loan_df))
    print(loan_df.head())
    loan_time = loan_df.loan_time
    train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time, ["uid", "day", "hour"]].drop_duplicates()
    test_slice = loan_df.ix[test_time, ["uid", "day", "hour"]].drop_duplicates()
    train_slice = train_slice.groupby("uid", as_index=False)["uid"].agg({"last_month_loan_hour": "count"})
    test_slice = test_slice.groupby("uid", as_index=False)["uid"].agg({"last_month_loan_hour": "count"})
    train_feat = train_feat.merge(train_slice[["uid", "last_month_loan_hour"]], on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_slice[["uid", "last_month_loan_hour"]], on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_loan_hour_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_loan_hour_test.csv".format(save_root), index=False)

def last_three_month_loan_tsfresh():
    settings = EfficientFCParameters()
    user_df = pd.read_csv(user_path)[["uid"]]

    train_feat = user_df[["uid"]].copy(deep=True)
    test_feat = user_df[["uid"]].copy(deep=True)
    loan_df = pd.read_csv(loan_path)

    loan_time = loan_df.loan_time.values
    train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    train_slice = loan_df.ix[train_time]
    test_slice = loan_df.ix[test_time]

    tsfresh_train = extract_features(train_slice[['uid', "loan_time", 'loan_amount', 'plannum']], column_id="uid", column_sort="loan_time").reset_index()
    tsfresh_test = extract_features(test_slice[['uid', "loan_time", 'loan_amount', 'plannum']], column_id="uid", column_sort="loan_time").reset_index()
    # print(tsfresh_train.columns.values)
    # print(tsfresh_test.head())
    tsfresh_train = tsfresh_train.rename(columns={"id": "uid"})
    tsfresh_test = tsfresh_test.rename(columns={"id": "uid"})
    train_feat = train_feat.merge(tsfresh_train, on="uid", how="left")
    test_feat = test_feat.merge(tsfresh_test, on="uid", how="left")
    # print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_three_month_loan_tsfresh_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_tsfresh_test.csv".format(save_root), index=False)

'''*******************************************click**********************************************'''
def last_month_click_times():
    user_df = pd.read_csv(user_path)[["uid"]]
    # print(user_df.head())
    click_df = pd.read_csv(click_path)
    print(len(click_df))
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    train_slice = click_df.ix[train_time, ["uid", "click_time"]]
    test_slice = click_df.ix[test_time, ["uid", "click_time"]]
    print(len(train_slice))
    print(len(test_slice))
    print(train_slice.head())
    print(test_slice.head())

    train_slice = train_slice.groupby(["uid"]).count().reset_index().rename(
        columns={"click_time": "last_month_click_times"})
    test_slice = test_slice.groupby(["uid"]).count().reset_index().rename(
        columns={"click_time": "last_month_click_times"})

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_slice, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_slice, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_click_times_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_click_times_test.csv".format(save_root), index=False)

def two_month_pid_max():
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    print(len(click_df))
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_two_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_two_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    train_slice = click_df.ix[train_time, ["uid", "pid"]]
    test_slice = click_df.ix[test_time, ["uid", "pid"]]
    train_slice["pid_times"] = 1
    test_slice["pid_times"] = 1

    train_slice = train_slice.groupby(["uid", "pid"]).count().reset_index()
    test_slice = test_slice.groupby(["uid", "pid"]).count().reset_index()

    train_times = train_slice[["uid", "pid_times"]].groupby(["uid"]).max().reset_index()
    test_times = test_slice[["uid", "pid_times"]].groupby(["uid"]).max().reset_index()

    train_times = train_times.merge(train_slice, on=["uid", "pid_times"], how="left")[["uid", "pid"]].rename(columns={"pid": "two_month_pid_max"}).drop_duplicates()
    test_times = test_times.merge(test_slice, on=["uid", "pid_times"], how="left")[["uid", "pid"]].rename(columns={"pid": "two_month_pid_max"}).drop_duplicates()
    train_times = train_times.groupby(["uid"]).max().reset_index()
    test_times = test_times.groupby(["uid"]).max().reset_index()
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(train_times, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(test_times, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/two_month_pid_max_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/two_month_pid_max_test.csv".format(save_root), index=False)

def last_month_pid_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train = click_train.groupby(["uid", "pid"])["click_time"].count().unstack().add_prefix('last_month_pid_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "pid"])["click_time"].count().unstack().add_prefix('last_month_pid_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_pid_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_pid_count_test.csv".format(save_root), index=False)

def last_two_month_pid_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_two_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_two_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train = click_train.groupby(["uid", "pid"])["click_time"].count().unstack().add_prefix('last_two_month_pid_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "pid"])["click_time"].count().unstack().add_prefix('last_two_month_pid_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_pid_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_pid_count_test.csv".format(save_root), index=False)

def last_three_month_pid_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_three_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_three_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train = click_train.groupby(["uid", "pid"])["click_time"].count().unstack().add_prefix('last_three_month_pid_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "pid"])["click_time"].count().unstack().add_prefix('last_three_month_pid_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_pid_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_pid_count_test.csv".format(save_root), index=False)

def last_month_param_count():
    user_df = pd.read_csv(user_path)[["uid"]]
    # print(user_df.head())
    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train = click_train.groupby(["uid", "param"])["click_time"].count().unstack().add_prefix('last_month_param_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "param"])["click_time"].count().unstack().add_prefix('last_month_param_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_month_param_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_param_count_test.csv".format(save_root), index=False)

def last_two_month_param_count():
    user_df = pd.read_csv(user_path)[["uid"]]
    # print(user_df.head())
    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_two_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_two_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train = click_train.groupby(["uid", "param"])["click_time"].count().unstack().add_prefix('last_two_month_param_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "param"])["click_time"].count().unstack().add_prefix('last_two_month_param_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_two_month_param_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_two_month_param_count_test.csv".format(save_root), index=False)

def last_three_month_param_count():
    user_df = pd.read_csv(user_path)[["uid"]]
    # print(user_df.head())
    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_three_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_three_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train = click_train.groupby(["uid", "param"])["click_time"].count().unstack().add_prefix('last_three_month_param_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "param"])["click_time"].count().unstack().add_prefix('last_three_month_param_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    train_feat.to_csv("{}/last_three_month_param_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_param_count_test.csv".format(save_root), index=False)

def last_month_pid_param_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train["pid_param"] = click_train.apply(lambda x: str(x.pid) + '_' + str(x.param), axis=1)
    click_test["pid_param"] = click_test.apply(lambda x: str(x.pid) + '_' + str(x.param), axis=1)
    click_train = click_train.groupby(["uid", "pid_param"])["click_time"].count().unstack().add_prefix('last_month_pid_param_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "pid_param"])["click_time"].count().unstack().add_prefix('last_month_pid_param_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    print(train_feat.head())
    print(test_feat.head())
    train_feat.to_csv("{}/last_month_pid_param_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_pid_param_count_test.csv".format(save_root), index=False)

def last_three_month_pid_param_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_three_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_three_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    click_train = click_df.ix[train_time]
    click_test = click_df.ix[test_time]

    click_train["pid_param"] = click_train.apply(lambda x: str(x.pid) + '_' + str(x.param), axis=1)
    click_test["pid_param"] = click_test.apply(lambda x: str(x.pid) + '_' + str(x.param), axis=1)
    click_train = click_train.groupby(["uid", "pid_param"])["click_time"].count().unstack().add_prefix('last_three_month_pid_param_onehot_').reset_index().fillna(0)
    click_test = click_test.groupby(["uid", "pid_param"])["click_time"].count().unstack().add_prefix('last_three_month_pid_param_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(click_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(click_test, on="uid", how="left").fillna(0)

    # print(train_feat.head())
    # print(test_feat.head())
    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_three_month_pid_param_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_pid_param_count_test.csv".format(save_root), index=False)

def three_month_pid_topK(k=3):
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    print(len(click_df))
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_three_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_three_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    train_slice = click_df.ix[train_time, ["uid", "pid"]]
    test_slice = click_df.ix[test_time, ["uid", "pid"]]
    train_slice["pid_times"] = 1
    test_slice["pid_times"] = 1

    train_slice = train_slice.groupby(["uid", "pid"]).count().reset_index()
    test_slice = test_slice.groupby(["uid", "pid"]).count().reset_index()

    train_slice = rank(train_slice, "uid", "pid_times", ascending=False, rank_name="pid_rank")
    test_slice = rank(test_slice, "uid", "pid_times", ascending=False, rank_name="pid_rank")

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    for i in range(k):
        max_pid_train = train_slice.ix[train_slice.pid_rank == i, ["uid", "pid"]]
        max_pid_test = test_slice.ix[test_slice.pid_rank == i, ["uid", "pid"]]
        train_feat = train_feat.merge(max_pid_train, on="uid", how="left").rename(columns={"pid": "pid_rank_{}".format(i)})
        test_feat = test_feat.merge(max_pid_test, on="uid", how="left").rename(columns={"pid": "pid_rank_{}".format(i)})

    print(train_feat.columns.values)
    train_feat.fillna(-1).to_csv("{}/three_month_pid_topK_train.csv".format(save_root), index=False)
    test_feat.fillna(-1).to_csv("{}/three_month_pid_topK_test.csv".format(save_root), index=False)


def two_month_pid_topK(k=3):
    user_df = pd.read_csv(user_path)[["uid"]]

    click_df = pd.read_csv(click_path)
    print(len(click_df))
    click_time = click_df.click_time.values
    train_time = ((click_time >= last_two_month_start_for_click_train) & (click_time < last_month_end_for_click_train))
    test_time = ((click_time >= last_two_month_start_for_click_test) & (click_time < last_month_end_for_click_test))

    train_slice = click_df.ix[train_time, ["uid", "pid"]]
    test_slice = click_df.ix[test_time, ["uid", "pid"]]
    train_slice["pid_times"] = 1
    test_slice["pid_times"] = 1

    train_slice = train_slice.groupby(["uid", "pid"]).count().reset_index()
    test_slice = test_slice.groupby(["uid", "pid"]).count().reset_index()

    train_slice = rank(train_slice, "uid", "pid_times", ascending=False, rank_name="pid_rank")
    test_slice = rank(test_slice, "uid", "pid_times", ascending=False, rank_name="pid_rank")

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    for i in range(k):
        max_pid_train = train_slice.ix[train_slice.pid_rank == i, ["uid", "pid"]]
        max_pid_test = test_slice.ix[test_slice.pid_rank == i, ["uid", "pid"]]
        train_feat = train_feat.merge(max_pid_train, on="uid", how="left").rename(columns={"pid": "two_month_pid_rank_{}".format(i)})
        test_feat = test_feat.merge(max_pid_test, on="uid", how="left").rename(columns={"pid": "two_month_pid_rank_{}".format(i)})

    print(train_feat.columns.values)
    train_feat.fillna(-1).to_csv("{}/two_month_pid_topK_train.csv".format(save_root), index=False)
    test_feat.fillna(-1).to_csv("{}/two_month_pid_topK_test.csv".format(save_root), index=False)

'''*******************************************user**********************************************'''
def date_prior():
    k = 5.
    f = 0.05
    g = 1.
    r_k = 0.02
    user_df = pd.read_csv(user_path)
    pred_base = pd.read_csv("{}/pred_base.csv".format(data_root))
    user_df["year"] = user_df.active_date.str.split('-', expand=True)[0]
    user_df["month"] = user_df.active_date.str.split('-', expand=True)[1]
    user_df["day"] = user_df.active_date.str.split('-', expand=True)[2]
    user_df["date"] = user_df["year"] + user_df["month"] + user_df["day"]
    print(user_df.head())
    loan_sum_df = pd.read_csv(loan_sum_path)
    # loan_sum_df["label"] = 1
    loan_sum_df = loan_sum_df.rename(columns={"loan_sum": "label"})
    user_df = user_df.merge(loan_sum_df, on="uid", how="left").fillna(0)
    user_df = user_df.merge(pred_base, on="uid", how="left")
    date_df = user_df[["uid", "date", "label"]]
    date_df["pred"] = -1

    def find_objects_with_only_one_record(feature_name):
        temp = date_df.reset_index()
        temp = temp.groupby(feature_name, as_index=False).count()
        return temp[temp['index'] == 1]  # 将出现次数==1的，全部置为同一类

    def reset_id_with_one_lot(high_cardinality):
        for c in high_cardinality:
            id_with_one_lot = find_objects_with_only_one_record(c)
            date_df.loc[date_df[c].isin(id_with_one_lot[c].ravel()), c] = -1

    def categorical_average(variable="date", y="label", pred_0="pred", feature_name="date_prior"):
        def calculate_average(sub1, sub2):
            sub1_group = sub1.groupby(variable, as_index=False)
            s = pd.DataFrame(data={
                variable: sub1_group.count()[variable],
                'sumy': sub1_group.sum()['y'],
                'avgY': sub1_group.mean()['y'],
                'cnt': sub1_group.count()['y']
            })
            # print("sub1_group: ", sub1_group.describe())
            tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable)
            del tmp['index']
            tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
            tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

            def compute_beta(row):
                cnt = row['cnt']
                rate = (cnt - k) / f
                rate = np.clip(rate, -10, 10)
                return 1.0 / (g + math.exp(rate))

            # print(tmp.head(10))
            tmp['beta'] = tmp.apply(compute_beta, axis=1)

            tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred'],
                                       axis=1) #  / (tmp['cnt']+1)

            tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred']
            tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred']
            tmp['random'] = np.random.uniform(size=len(tmp))
            tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] * (1 + (row['random'] - 0.5) * r_k), axis=1)

            return tmp['adj_avg'].ravel()

        # cv for training set
        nfolds = 5
        k_fold = KFold(nfolds, shuffle=True, random_state=2011)
        date_df[feature_name] = -1
        for (train_index, val_index) in k_fold.split(np.zeros(len(date_df)), date_df['label'].ravel()):
            sub = pd.DataFrame(data={
                variable: date_df[variable],
                'y': date_df[y],
                'pred': date_df[pred_0]
            })
            sub1 = sub.iloc[train_index]
            sub2 = sub.iloc[val_index]

            date_df.loc[val_index, feature_name] = calculate_average(sub1, sub2)

        # print(age_df.head())
        date_df[["uid", "date_prior"]].to_csv("{}/date_prior_train.csv".format(save_root), index=False)
        date_df[["uid", "date_prior"]].to_csv("{}/date_prior_test.csv".format(save_root), index=False)

    reset_id_with_one_lot(["date"])
    categorical_average(variable="date", y="label", pred_0="pred", feature_name="date_prior")

def regist_usr_count():
    user = pd.read_csv(user_path)
    regist_date_user_count = user[["uid", "active_date"]].groupby("active_date", as_index=False).count().rename(columns={"uid": "regist_date_user_count"})
    regist_date_limit_sum = user[["limit", "active_date"]].groupby("active_date", as_index=False).sum().rename(columns={"limit": "regist_date_limit_sum"})
    regist_date_limit_mean = user[["limit", "active_date"]].groupby("active_date", as_index=False).mean().rename(columns={"limit": "regist_date_limit_mean"})
    regist_date_limit_max = user[["limit", "active_date"]].groupby("active_date", as_index=False).max().rename(columns={"limit": "regist_date_limit_max"})
    regist_date_limit_min = user[["limit", "active_date"]].groupby("active_date", as_index=False).min().rename(columns={"limit": "regist_date_limit_min"})
    regist_date_limit_median = user[["limit", "active_date"]].groupby("active_date", as_index=False).median().rename(columns={"limit": "regist_date_limit_median"})

    user = user.merge(regist_date_user_count, on="active_date", how="left")
    user = user.merge(regist_date_limit_sum, on="active_date", how="left")
    user = user.merge(regist_date_limit_mean, on="active_date", how="left")
    user = user.merge(regist_date_limit_max, on="active_date", how="left")
    user = user.merge(regist_date_limit_min, on="active_date", how="left")
    user = user.merge(regist_date_limit_median, on="active_date", how="left")
    user = user[["uid", "regist_date_user_count", "regist_date_limit_sum", "regist_date_limit_mean", "regist_date_limit_max", "regist_date_limit_min", "regist_date_limit_median"]]
    print(user.head())
    user.to_csv("{}/regist_usr_count_train.csv".format(save_root), index=False)
    user.to_csv("{}/regist_usr_count_test.csv".format(save_root), index=False)

def date_prior_change():
    k = 5.
    f = 0.05
    g = 1.
    r_k = 0.02
    user_df = pd.read_csv(user_path)
    loan = pd.read_csv(loan_path)
    '''********** Get train test feat label **********'''
    loan_time = loan.loan_time.values
    train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))
    train_label = loan.ix[train_time, ["uid", "loan_amount"]]
    test_label = loan.ix[test_time, ["uid", "loan_amount"]]
    train_label["loan_amount"] = train_label["loan_amount"].apply(lambda x: 5 ** x - 1)
    test_label["loan_amount"] = test_label["loan_amount"].apply(lambda x: 5 ** x - 1)
    train_label = train_label.groupby("uid", as_index=False).sum().rename(columns={"loan_amount": "label"})
    test_label = test_label.groupby("uid", as_index=False).sum().rename(columns={"loan_amount": "label"})
    train_label["label"] = train_label["label"].apply(lambda x: math.log(x+1)/math.log(5))
    test_label["label"] = test_label["label"].apply(lambda x: math.log(x+1)/math.log(5))

    user_df["year"] = user_df.active_date.str.split('-', expand=True)[0]
    user_df["month"] = user_df.active_date.str.split('-', expand=True)[1]
    user_df["day"] = user_df.active_date.str.split('-', expand=True)[2]
    user_df["date"] = user_df["year"] + user_df["month"] + user_df["day"]
    print(user_df.head())

    train_date_df = user_df[["uid", "date"]].copy(deep=True)
    test_date_df = user_df[["uid", "date"]].copy(deep=True)
    train_date_df = train_date_df.merge(train_label, on="uid", how="left").fillna(0)
    test_date_df = test_date_df.merge(test_label, on="uid", how="left").fillna(0)

    train_date_df["pred"] = -1
    test_date_df["pred"] = -1

    def reset_id_with_one_lot(high_cardinality, date_df):
        for c in high_cardinality:
            temp = date_df.reset_index()
            temp = temp.groupby(c, as_index=False).count()
            id_with_one_lot = temp[temp['index'] == 1]
            date_df.loc[date_df[c].isin(id_with_one_lot[c].ravel()), c] = -1
        return date_df

    def categorical_average(date_df, variable="date", y="label", pred_0="pred", feature_name="date_prior"):
        def calculate_average(sub1, sub2):
            sub1_group = sub1.groupby(variable, as_index=False)
            s = pd.DataFrame(data={
                variable: sub1_group.count()[variable],
                'sumy': sub1_group.sum()['y'],
                'avgY': sub1_group.mean()['y'],
                'cnt': sub1_group.count()['y']
            })
            # print("sub1_group: ", sub1_group.describe())
            tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable)
            del tmp['index']
            tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
            tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

            def compute_beta(row):
                cnt = row['cnt']
                rate = (cnt - k) / f
                rate = np.clip(rate, -10, 10)
                return 1.0 / (g + math.exp(rate))

            # print(tmp.head(10))
            tmp['beta'] = tmp.apply(compute_beta, axis=1)

            tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred'],
                                       axis=1) #  / (tmp['cnt']+1)

            tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred']
            tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred']
            tmp['random'] = np.random.uniform(size=len(tmp))
            tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] * (1 + (row['random'] - 0.5) * r_k), axis=1)

            return tmp['adj_avg'].ravel()

        # cv for training set
        nfolds = 5
        k_fold = KFold(nfolds, shuffle=True, random_state=2011)
        date_df[feature_name] = -1
        for (train_index, val_index) in k_fold.split(np.zeros(len(date_df)), date_df['label'].ravel()):
            sub = pd.DataFrame(data={
                variable: date_df[variable],
                'y': date_df[y],
                'pred': date_df[pred_0]
            })
            sub1 = sub.iloc[train_index]
            sub2 = sub.iloc[val_index]

            date_df.loc[val_index, feature_name] = calculate_average(sub1, sub2)

        # print(age_df.head())
        # date_df[["uid", "date_prior"]].to_csv("{}/date_prior_change_train.csv".format(save_root), index=False)
        # date_df[["uid", "date_prior"]].to_csv("{}/date_prior_change_test.csv".format(save_root), index=False)
        return date_df[["uid", "date_prior"]]

    train_date_df = reset_id_with_one_lot(["date"], train_date_df)
    test_date_df = reset_id_with_one_lot(["date"], test_date_df)
    train_date_df = categorical_average(train_date_df, variable="date", y="label", pred_0="pred", feature_name="date_prior")
    test_date_df = categorical_average(test_date_df, variable="date", y="label", pred_0="pred", feature_name="date_prior")
    train_date_df = train_date_df.rename(columns={"date_prior": "date_prior_change"})
    test_date_df = test_date_df.rename(columns={"date_prior": "date_prior_change"})
    train_date_df.to_csv("{}/date_prior_change_train.csv".format(save_root), index=False)
    test_date_df.to_csv("{}/date_prior_change_test.csv".format(save_root), index=False)

def limit_loan_count():
    user_df = pd.read_csv(user_path)[["uid", "limit"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)
    loan = pd.read_csv(loan_path)
    loan = loan.merge(user_df, on="uid", how="left")
    loan_time = loan.loan_time.values
    last_month_train_time = ((loan_time >= last_month_start_train) & (loan_time < last_month_end_train))
    last_month_test_time = ((loan_time >= last_month_start_test) & (loan_time < last_month_end_test))
    last_two_month_train_time = ((loan_time >= last_two_month_start_train) & (loan_time < last_month_end_train))
    last_two_month_test_time = ((loan_time >= last_two_month_start_test) & (loan_time < last_month_end_test))
    last_three_month_train_time = ((loan_time >= last_three_month_start_train) & (loan_time < last_month_end_train))
    last_three_month_test_time = ((loan_time >= last_three_month_start_test) & (loan_time < last_month_end_test))

    last_month_loan_train = loan.ix[last_month_train_time, ["limit", "loan_amount"]].groupby("limit", as_index=False)["loan_amount"]\
        .agg({"last_month_limit_loan_mean": "mean",
              "last_month_limit_loan_median": "median"})
    last_month_loan_test = loan.ix[last_month_test_time, ["limit", "loan_amount"]].groupby("limit", as_index=False)["loan_amount"]\
        .agg({"last_month_limit_loan_mean": "mean",
              "last_month_limit_loan_median": "median"})
    last_month_plannum_train = loan.ix[last_month_train_time, ["limit", "plannum"]].groupby("limit", as_index=False)["plannum"]\
        .agg({"last_month_limit_plannum_mean": "mean",
              "last_month_limit_plannum_median": "median"})
    last_month_plannum_test = loan.ix[last_month_test_time, ["limit", "plannum"]].groupby("limit", as_index=False)["plannum"] \
        .agg({"last_month_limit_plannum_mean": "mean",
              "last_month_limit_plannum_median": "median"})
    last_two_month_loan_train = loan.ix[last_two_month_train_time, ["limit", "loan_amount"]].groupby("limit", as_index=False)["loan_amount"] \
        .agg({"last_two_month_limit_loan_mean": "mean",
              "last_two_month_limit_loan_median": "median"})
    last_two_month_loan_test = loan.ix[last_two_month_test_time, ["limit", "loan_amount"]].groupby("limit", as_index=False)["loan_amount"] \
        .agg({"last_two_month_limit_loan_mean": "mean",
              "last_two_month_limit_loan_median": "median"})
    last_two_month_plannum_train = loan.ix[last_two_month_train_time, ["limit", "plannum"]].groupby("limit", as_index=False)["plannum"] \
        .agg({"last_two_month_limit_plannum_mean": "mean",
              "last_two_month_limit_plannum_median": "median"})
    last_two_month_plannum_test = loan.ix[last_two_month_test_time, ["limit", "plannum"]].groupby("limit", as_index=False)["plannum"] \
        .agg({"last_two_month_limit_plannum_mean": "mean",
              "last_two_month_limit_plannum_median": "median"})
    last_three_month_loan_train = loan.ix[last_three_month_train_time, ["limit", "loan_amount"]].groupby("limit", as_index=False)["loan_amount"] \
        .agg({"last_three_month_limit_loan_mean": "mean",
              "last_three_month_limit_loan_median": "median"})
    last_three_month_loan_test = loan.ix[last_three_month_test_time, ["limit", "loan_amount"]].groupby("limit", as_index=False)["loan_amount"] \
        .agg({"last_three_month_limit_loan_mean": "mean",
              "last_three_month_limit_loan_median": "median"})
    last_three_month_plannum_train = loan.ix[last_three_month_train_time, ["limit", "plannum"]].groupby("limit", as_index=False)["plannum"] \
        .agg({"last_three_month_limit_plannum_mean": "mean",
              "last_three_month_limit_plannum_median": "median"})
    last_three_month_plannum_test = loan.ix[last_three_month_test_time, ["limit", "plannum"]].groupby("limit", as_index=False)["plannum"] \
        .agg({"last_three_month_limit_plannum_mean": "mean",
              "last_three_month_limit_plannum_median": "median"})
    train_feat = train_feat.merge(last_month_loan_train, on="limit", how="left")
    train_feat = train_feat.merge(last_month_plannum_train, on="limit", how="left")
    train_feat = train_feat.merge(last_two_month_loan_train, on="limit", how="left")
    train_feat = train_feat.merge(last_two_month_plannum_train, on="limit", how="left")
    train_feat = train_feat.merge(last_three_month_loan_train, on="limit", how="left")
    train_feat = train_feat.merge(last_three_month_plannum_train, on="limit", how="left")

    test_feat = test_feat.merge(last_month_loan_test, on="limit", how="left")
    test_feat = test_feat.merge(last_month_plannum_test, on="limit", how="left")
    test_feat = test_feat.merge(last_two_month_loan_test, on="limit", how="left")
    test_feat = test_feat.merge(last_two_month_plannum_test, on="limit", how="left")
    test_feat = test_feat.merge(last_three_month_loan_test, on="limit", how="left")
    test_feat = test_feat.merge(last_three_month_plannum_test, on="limit", how="left")

    train_feat = train_feat.drop("limit", axis=1)
    test_feat = test_feat.drop("limit", axis=1)
    print(len(train_feat))
    print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.to_csv("{}/limit_loan_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/limit_loan_count_test.csv".format(save_root), index=False)

'''*******************************************order**********************************************'''
def is_usr_order_every_cate():
    user_df = pd.read_csv(user_path)[["uid"]]
    # print(user_df.head())
    order_df = pd.read_csv(order_path)
    print(len(order_df))
    buy_time = order_df.buy_time.values
    train_time = (buy_time < last_month_end_for_click_train)
    test_time = (buy_time < last_month_end_for_click_test)
    train_slice = order_df.ix[train_time, ["uid", "cate_id"]]
    test_slice = order_df.ix[test_time, ["uid", "cate_id"]]

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    cate_set = sorted(set(test_slice["cate_id"].values))
    print(cate_set)
    for cate in cate_set:
        train_slice_cate = train_slice.ix[train_slice.cate_id == cate].drop_duplicates()
        test_slice_cate = test_slice.ix[test_slice.cate_id == cate].drop_duplicates()
        train_slice_cate = train_slice_cate.groupby(["uid"]).count().reset_index().rename(columns={"cate_id": "cate_{}".format(cate)})
        test_slice_cate = test_slice_cate.groupby(["uid"]).count().reset_index().rename(columns={"cate_id": "cate_{}".format(cate)})
        train_feat = train_feat.merge(train_slice_cate, on="uid", how="left").fillna(0)
        test_feat = test_feat.merge(test_slice_cate, on="uid", how="left").fillna(0)
    print(train_feat.head())
    print(test_feat.head())
    train_feat.to_csv("{}/is_usr_order_every_cate_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/is_usr_order_every_cate_test.csv".format(save_root), index=False)

def last_month_cate_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    order_df = pd.read_csv(order_path)
    order_time = order_df.buy_time.values
    train_time = ((order_time >= last_month_start_for_click_train) & (order_time < last_month_end_for_click_train))
    test_time = ((order_time >= last_month_start_for_click_test) & (order_time < last_month_end_for_click_test))

    order_train = order_df.ix[train_time]
    order_test = order_df.ix[test_time]

    order_train = order_train.groupby(["uid", "cate_id"])["buy_time"].count().unstack().add_prefix('last_month_cate_onehot_').reset_index().fillna(0)
    order_test = order_test.groupby(["uid", "cate_id"])["buy_time"].count().unstack().add_prefix('last_month_cate_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(order_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(order_test, on="uid", how="left").fillna(0)

    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_month_cate_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_month_cate_count_test.csv".format(save_root), index=False)

def last_three_month_cate_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    order_df = pd.read_csv(order_path)

    # order_df.dropna(axis=0, inplace=True)
    # order_df = order_df.ix[order_df.qty > 0.]
    # order_df["price"] = order_df["price"].apply(lambda x: np.round(5 ** x - 1, 2))
    # order_df["discount"] = order_df["discount"].apply(lambda x: np.round(5 ** x - 1, 2))
    # order_df["loan_amount"] = order_df["price"] * order_df["qty"] - order_df["discount"]
    # order_df = order_df.ix[order_df.loan_amount > 0.]

    order_time = order_df.buy_time.values
    train_time = ((order_time >= last_three_month_start_for_click_train) & (order_time < last_month_end_for_click_train))
    test_time = ((order_time >= last_three_month_start_for_click_test) & (order_time < last_month_end_for_click_test))

    order_train = order_df.ix[train_time]
    order_test = order_df.ix[test_time]

    order_train = order_train.groupby(["uid", "cate_id"])["buy_time"].count().unstack().add_prefix('last_three_month_cate_onehot_').reset_index().fillna(0)
    order_test = order_test.groupby(["uid", "cate_id"])["buy_time"].count().unstack().add_prefix('last_three_month_cate_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(order_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(order_test, on="uid", how="left").fillna(0)

    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_three_month_cate_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_cate_count_test.csv".format(save_root), index=False)

def last_three_month_qty_count():
    user_df = pd.read_csv(user_path)[["uid"]]

    order_df = pd.read_csv(order_path)
    order_df.ix[order_df.qty > 100., "qty"] = 100.
    # print(order_df.qty.quantile(0.998))
    order_time = order_df.buy_time.values
    train_time = ((order_time >= last_three_month_start_for_click_train) & (order_time < last_month_end_for_click_train))
    test_time = ((order_time >= last_three_month_start_for_click_test) & (order_time < last_month_end_for_click_test))

    order_train = order_df.ix[(train_time & (order_df.qty > 0.))] #
    order_test = order_df.ix[(test_time & (order_df.qty > 0.))]

    order_train = order_train.groupby(["uid", "qty"])["buy_time"].count().unstack().add_prefix('last_three_month_qty_onehot_').reset_index().fillna(0)
    order_test = order_test.groupby(["uid", "qty"])["buy_time"].count().unstack().add_prefix('last_three_month_qty_onehot_').reset_index().fillna(0)

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    train_feat = train_feat.merge(order_train, on="uid", how="left").fillna(0)
    test_feat = test_feat.merge(order_test, on="uid", how="left").fillna(0)

    print(train_feat.columns.values)
    train_feat.to_csv("{}/last_three_month_qty_count_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_qty_count_test.csv".format(save_root), index=False)


def last_three_month_loan_match_buy():
    user_df = pd.read_csv(user_path)[["uid"]]
    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)
    loan_df = pd.read_csv(loan_path)
    loan_df["loan_amount"] = loan_df["loan_amount"].apply(lambda x: np.round(x, 2))
    loan_df["buy_time"] = loan_df["loan_time"].str.split(' ', expand=True)[0]
    loan_df = loan_df[["uid", "buy_time", "loan_amount"]].drop_duplicates()
    print(loan_df.head())
    print(len(set(loan_df.uid)))

    order_df = pd.read_csv(order_path)
    order_df.dropna(axis=0, inplace=True)
    order_df = order_df.ix[order_df.qty > 0.]
    order_df["price"] = order_df["price"].apply(lambda x: np.round(5**x - 1, 2))
    order_df["discount"] = order_df["discount"].apply(lambda x: np.round(5 ** x - 1, 2))
    order_df["loan_amount"] = order_df["price"] * order_df["qty"] - order_df["discount"]
    order_df = order_df.ix[order_df.loan_amount > 0.]
    order_df["loan_amount"] = order_df["loan_amount"].apply(lambda x: np.round((math.log(x+1)/math.log(5)), 2))
    order_df = order_df[["uid", "buy_time", "loan_amount", "cate_id"]].drop_duplicates()
    print(loan_df.head())
    order_df["match"] = 1
    print(order_df.head())

    match_df = loan_df.merge(order_df, on=["uid", "buy_time", "loan_amount"], how="left").fillna(0)
    del loan_df
    del order_df
    match_df = match_df.ix[match_df.match == 1, ["uid", "buy_time", "loan_amount", "match"]].drop_duplicates()

    match_time = match_df.buy_time.values
    train_time = ((match_time >= last_three_month_start_for_click_train) & (match_time < last_month_end_for_click_train))
    test_time = ((match_time >= last_three_month_start_for_click_test) & (match_time < last_month_end_for_click_test))

    def diff_of_days(day1, day2):
        days = (datetime.strptime(day1, '%Y-%m-%d') - datetime.strptime(day2, '%Y-%m-%d')).days
        return abs(days)

    match_train = match_df.ix[train_time].rename(columns={"buy_time": "match_buy_time", "loan_amount": "match_loan_amount"})
    match_test = match_df.ix[test_time].rename(columns={"buy_time": "match_buy_time", "loan_amount": "match_loan_amount"})

    match_train["match_day_dis"] = match_train["match_buy_time"].apply(lambda x: diff_of_days(last_month_end_for_click_train, x))
    match_test["match_day_dis"] = match_test["match_buy_time"].apply(lambda x: diff_of_days(last_month_end_for_click_test, x))
    match_train = match_train.drop("match_buy_time", axis=1).drop_duplicates()
    match_test = match_test.drop("match_buy_time", axis=1).drop_duplicates()

    train_feat = train_feat.merge(match_train, on="uid", how="left").fillna(0).drop_duplicates("uid")
    test_feat = test_feat.merge(match_test, on="uid", how="left").fillna(0).drop_duplicates("uid")
    print(train_feat.columns.values)
    print(len(train_feat))
    print(len(test_feat))
    train_feat.to_csv("{}/last_three_month_loan_match_buy_train.csv".format(save_root), index=False)
    test_feat.to_csv("{}/last_three_month_loan_match_buy_test.csv".format(save_root), index=False)

def three_month_cate_topK(k=3):
    user_df = pd.read_csv(user_path)[["uid"]]

    order_df = pd.read_csv(order_path)
    print(len(order_df))
    order_time = order_df.buy_time.values
    train_time = ((order_time >= last_three_month_start_for_click_train) & (order_time < last_month_end_for_click_train))
    test_time = ((order_time >= last_three_month_start_for_click_test) & (order_time < last_month_end_for_click_test))

    train_slice = order_df.ix[train_time, ["uid", "cate_id"]]
    test_slice = order_df.ix[test_time, ["uid", "cate_id"]]
    train_slice["cate_times"] = 1
    test_slice["cate_times"] = 1

    train_slice = train_slice.groupby(["uid", "cate_id"]).count().reset_index()
    test_slice = test_slice.groupby(["uid", "cate_id"]).count().reset_index()

    train_slice = rank(train_slice, "uid", "cate_times", ascending=False, rank_name="cate_rank")
    test_slice = rank(test_slice, "uid", "cate_times", ascending=False, rank_name="cate_rank")

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    for i in range(k):
        max_pid_train = train_slice.ix[train_slice.cate_rank == i, ["uid", "cate_id"]]
        max_pid_test = test_slice.ix[test_slice.cate_rank == i, ["uid", "cate_id"]]
        train_feat = train_feat.merge(max_pid_train, on="uid", how="left").rename(columns={"cate_id": "cate_rank_{}".format(i)})
        test_feat = test_feat.merge(max_pid_test, on="uid", how="left").rename(columns={"cate_id": "cate_rank_{}".format(i)})

    # print(train_feat.head())
    # print(test_feat.head())
    # print(len(train_feat))
    # print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.fillna(-1).to_csv("{}/three_month_cate_topK_train.csv".format(save_root), index=False)
    test_feat.fillna(-1).to_csv("{}/three_month_cate_topK_test.csv".format(save_root), index=False)

def two_month_cate_topK(k=3):
    user_df = pd.read_csv(user_path)[["uid"]]

    order_df = pd.read_csv(order_path)
    print(len(order_df))
    order_time = order_df.buy_time.values
    train_time = ((order_time >= last_two_month_start_for_click_train) & (order_time < last_month_end_for_click_train))
    test_time = ((order_time >= last_two_month_start_for_click_test) & (order_time < last_month_end_for_click_test))

    train_slice = order_df.ix[train_time, ["uid", "cate_id"]]
    test_slice = order_df.ix[test_time, ["uid", "cate_id"]]
    train_slice["cate_times"] = 1
    test_slice["cate_times"] = 1

    train_slice = train_slice.groupby(["uid", "cate_id"]).count().reset_index()
    test_slice = test_slice.groupby(["uid", "cate_id"]).count().reset_index()

    train_slice = rank(train_slice, "uid", "cate_times", ascending=False, rank_name="cate_rank")
    test_slice = rank(test_slice, "uid", "cate_times", ascending=False, rank_name="cate_rank")

    train_feat = user_df.copy(deep=True)
    test_feat = user_df.copy(deep=True)

    for i in range(k):
        max_pid_train = train_slice.ix[train_slice.cate_rank == i, ["uid", "cate_id"]]
        max_pid_test = test_slice.ix[test_slice.cate_rank == i, ["uid", "cate_id"]]
        train_feat = train_feat.merge(max_pid_train, on="uid", how="left").rename(columns={"cate_id": "two_month_cate_rank_{}".format(i)})
        test_feat = test_feat.merge(max_pid_test, on="uid", how="left").rename(columns={"cate_id": "two_month_cate_rank_{}".format(i)})

    # print(train_feat.head())
    # print(test_feat.head())
    # print(len(train_feat))
    # print(len(test_feat))
    print(train_feat.columns.values)
    train_feat.fillna(-1).to_csv("{}/two_month_cate_topK_train.csv".format(save_root), index=False)
    test_feat.fillna(-1).to_csv("{}/two_month_cate_topK_test.csv".format(save_root), index=False)

if __name__ == "__main__":
    print("\nGain all features..\n")

    # has_loan()
    # last_three_month_loan_nums()
    # last_month_loan_amount()
    # last_month_plannum_amount()
    # last_two_month_plannum_amount()
    # last_three_month_plannum_amount()
    # last_three_month_loan_div_plannum_amount()
    # last_month_loan_div_plannum_amount()
    # last_three_month_loan_mean()
    # last_two_month_plannum_mean()
    # last_month_loan_mean()
    # last_time_loan_features()
    # first_time_loan_features()
    # last_second_time_loan_features()
    # last_loan_day_features()
    # last_two_month_loan_mean()
    # last_two_month_loan_quantile25()
    # amount_plannum_repeat_loan()
    # loan_weekend_static_by_day(days=90)
    # loan_weekend_static_by_day(days=120)
    # month_loan_sum()
    # last_two_month_loan_div_plannum_amount()
    # loan_left()
    # last_second_month_loan_feat()
    # last_third_month_loan_feat()
    # last_month_loan_tsfresh()
    # last_month_loan_feat()
    # last_month_plannum_count()
    # last_two_month_plannum_count()
    # last_three_month_plannum_count()
    # finish_repay_nums()
    # last_month_loan_hour()
    # last_two_month_loan_hour()
    # last_three_month_loan_hour()

    # last_month_click_times()
    # two_month_pid_max()
    # last_month_pid_count()
    # last_three_month_pid_count()
    # last_month_param_count()
    # last_three_month_param_count()
    # last_month_pid_param_count()
    # last_three_month_pid_param_count()
    # last_two_month_pid_count()
    # last_two_month_param_count()
    # three_month_pid_topK(k=5)
    # two_month_pid_topK(k=5)

    # date_prior()
    # regist_usr_count()
    # date_prior_change()
    # limit_loan_count()

    # is_usr_order_every_cate()
    # last_month_cate_count()
    # last_three_month_cate_count()
    # last_three_month_loan_match_buy()
    # three_month_cate_topK(k=10)
    # two_month_cate_topK(k=10)
    # last_three_month_cate_money()
