from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from com_util import *
import gc

df_train = pd.read_csv('../input/air_visit_data.csv')
df_test = pd.read_csv("../input/sample_submission.csv")
store_id_relation = pd.read_csv("../input/store_id_relation.csv")

df_test["air_store_id"]=df_test["id"].apply(lambda x:"_".join(x.split("_")[:2]))
df_test["visit_date"]=df_test["id"].apply(lambda x:x.split("_")[2])
del df_test["visitors"]

air_reserve = pd.read_csv("../input/air_reserve.csv")
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")
hpg_reserve = pd.merge(hpg_reserve, store_id_relation, how='inner', on=['hpg_store_id'])
air_reserve["reserve_date"]=air_reserve["reserve_datetime"].apply(lambda x:x.split(" ")[0])
air_reserve["visit_date"]=air_reserve["visit_datetime"].apply(lambda x:x.split(" ")[0])
hpg_reserve["reserve_date"]=hpg_reserve["reserve_datetime"].apply(lambda x:x.split(" ")[0])
hpg_reserve["visit_date"]=hpg_reserve["visit_datetime"].apply(lambda x:x.split(" ")[0])
air_reserve['reserve_datetime_diff'] = (pd.to_datetime(air_reserve['visit_date']) - pd.to_datetime(air_reserve['reserve_date'])).dt.days
hpg_reserve['reserve_datetime_diff'] = (pd.to_datetime(hpg_reserve['visit_date']) - pd.to_datetime(hpg_reserve['reserve_date'])).dt.days
air_reserve['reserve_hour'] = pd.to_datetime(air_reserve['reserve_datetime']).dt.day
hpg_reserve['reserve_hour'] = pd.to_datetime(hpg_reserve['reserve_datetime']).dt.day

df_train=df_train.merge(store_id_relation,on=["air_store_id"],how="left")

df_date_info= pd.read_csv("../input/date_info.csv")
df_date_info.columns=["visit_date","day_of_week","holiday_flg"]
df_date_info["day_of_week"]=df_date_info["day_of_week"].replace({"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7})
df_date_info["holiday"]= map(lambda a, b: 1 if a in [6, 7] or b == 1 else 0, df_date_info["day_of_week"], df_date_info["holiday_flg"])
del df_date_info["day_of_week"]

air_info=pd.read_csv("../input/air_store_info.csv")
air_info=encode_count(air_info,"air_genre_name")
air_info=encode_count(air_info,"air_area_name")
hpg_info=pd.read_csv("../input/hpg_store_info.csv")
hpg_info=encode_count(hpg_info,"hpg_genre_name")
hpg_info=encode_count(hpg_info,"hpg_area_name")
#构造数据

def date_gap(x,y):
    a,b,c=x.split("-")
    return (date(int(a),int(b),int(c))-y).days
def date_handle(df):
    df_visit_date=pd.to_datetime(df["visit_date"])
    df["weekday"]=df_visit_date.dt.weekday
    df["day"]=df_visit_date.dt.day
    df["day"]=df["day"].apply(lambda x:0 if x<=5 else 2 if x>=25 else 1)
    df = df.merge(air_info, on="air_store_id", how="left").fillna(-1)
    #df = df.merge(hpg_info, on="hpg_store_id", how="left").fillna(-1)
    df = df.merge(df_date_info, on="visit_date", how="left").fillna(-1)
    #df["holiday"] = map(lambda a, b: 1 if a in [5, 6] or b == 1 else 0, df["weekday"], df["holiday_flg"])
    return df

def create_features(df_label,df_train,df_air_reserve,df_hpg_reserve):
    df_train=date_handle(df_train)
    df_label=date_handle(df_label)

    #预定信息
    df_label=feat_sum(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_datetime_diff","air_reserve_datetime_diff_sum")
    df_label=feat_mean(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_datetime_diff","air_reserve_datetime_diff_mean")
    df_label=feat_sum(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_visitors","air_reserve_visitors_sum")
    df_label=feat_mean(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_visitors","air_reserve_visitors_mean")
    df_label=feat_sum(df_label,df_air_reserve,["visit_date"],"reserve_visitors","air_date_reserve_visitors_sum")
    df_label=feat_mean(df_label,df_air_reserve,["visit_date"],"reserve_visitors","air_date_reserve_visitors_mean")

    df_label=feat_sum(df_label,df_hpg_reserve,["air_store_id","visit_date"],"reserve_datetime_diff","hpg_reserve_datetime_diff_sum")
    df_label=feat_mean(df_label,df_hpg_reserve,["air_store_id","visit_date"],"reserve_datetime_diff","hpg_reserve_datetime_diff_mean")
    df_label=feat_sum(df_label,df_hpg_reserve,["air_store_id","visit_date"],"reserve_visitors","hpg_reserve_visitors_sum")
    df_label=feat_mean(df_label,df_hpg_reserve,["air_store_id","visit_date"],"reserve_visitors","hpg_reserve_visitors_mean")
    df_label=feat_sum(df_label,df_hpg_reserve,["visit_date"],"reserve_visitors","hpg_date_reserve_visitors_sum")
    df_label=feat_mean(df_label,df_hpg_reserve,["visit_date"],"reserve_visitors","hpg_date_reserve_visitors_mean")

    #df_label['total_reserv_sum'] = df_label['air_reserve_visitors_sum'] + df_label['hpg_reserve_visitors_sum']

    for i in [35,63,140]:
        df_air_reserve_select=df_air_reserve[df_air_reserve.day_gap>=-i].copy()
        df_hpg_reserve_select=df_hpg_reserve[df_hpg_reserve.day_gap>=-i].copy()
        date_air_reserve=pd.DataFrame(df_air_reserve_select.groupby(["air_store_id","visit_date"]).reserve_visitors.sum()).reset_index()
        date_air_reserve.columns=["air_store_id","visit_date","reserve_visitors_sum"]
        date_air_reserve=feat_count(date_air_reserve,df_air_reserve_select,["air_store_id","visit_date"],"reserve_visitors","reserve_visitors_count")
        date_air_reserve=feat_mean(date_air_reserve,df_air_reserve_select,["air_store_id","visit_date"],"reserve_visitors","reserve_visitors_mean")

        date_hpg_reserve=pd.DataFrame(df_hpg_reserve_select.groupby(["air_store_id","visit_date"]).reserve_visitors.sum()).reset_index()
        date_hpg_reserve.columns=["air_store_id","visit_date","reserve_visitors_sum"]
        date_hpg_reserve=feat_count(date_hpg_reserve,df_hpg_reserve_select,["air_store_id","visit_date"],"reserve_visitors","reserve_visitors_count")
        date_hpg_reserve=feat_mean(date_hpg_reserve,df_hpg_reserve_select,["air_store_id","visit_date"],"reserve_visitors","reserve_visitors_mean")

        date_air_reserve=date_handle(date_air_reserve)
        date_hpg_reserve=date_handle(date_hpg_reserve)
        date_air_reserve["holiday"] = map(lambda a, b: 1 if a in [5, 6] or b == 1 else 0, date_air_reserve["weekday"], date_air_reserve["holiday_flg"])
        date_hpg_reserve["holiday"] = map(lambda a, b: 1 if a in [5, 6] or b == 1 else 0, date_hpg_reserve["weekday"], date_hpg_reserve["holiday_flg"])

        df_label=feat_mean(df_label,date_air_reserve,["air_store_id","weekday"],"reserve_visitors_sum","air_reserve_visitors_sum_weekday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_store_id","weekday"],"reserve_visitors_sum","hpg_reserve_visitors_sum_weekday_mean_%s"%i)
        df_label=feat_mean(df_label,date_air_reserve,["air_store_id","weekday"],"reserve_visitors_mean","air_reserve_visitors_mean_weekday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_store_id","weekday"],"reserve_visitors_mean","hpg_reserve_visitors_mean_weekday_mean_%s"%i)
        df_label=feat_mean(df_label,date_air_reserve,["air_store_id","weekday"],"reserve_visitors_count","air_reserve_visitors_count_weekday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_store_id","weekday"],"reserve_visitors_count","hpg_reserve_visitors_count_weekday_mean_%s"%i)


        df_label=feat_mean(df_label,date_air_reserve,["air_store_id","holiday"],"reserve_visitors_sum","air_reserve_visitors_sum_holiday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_store_id","holiday"],"reserve_visitors_sum","hpg_reserve_visitors_sum_holiday_mean_%s"%i)
        df_label=feat_mean(df_label,date_air_reserve,["air_store_id","holiday"],"reserve_visitors_mean","air_reserve_visitors_mean_holiday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_store_id","holiday"],"reserve_visitors_mean","hpg_reserve_visitors_mean_holiday_mean_%s"%i)
        df_label=feat_mean(df_label,date_air_reserve,["air_store_id","holiday"],"reserve_visitors_count","air_reserve_visitors_count_holiday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_store_id","holiday"],"reserve_visitors_count","hpg_reserve_visitors_count_holiday_mean_%s"%i)

        df_label=feat_mean(df_label,date_air_reserve,["air_genre_name","air_area_name","holiday"],"reserve_visitors_sum","air_genre_area_holiday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_genre_name","air_area_name","holiday"],"reserve_visitors_sum","hpg_genre_area_holiday_mean_%s"%i)

        df_label=feat_mean(df_label,date_air_reserve,["air_genre_name","air_area_name","weekday"],"reserve_visitors_sum","air_genre_area_weekday_mean_%s"%i)
        df_label=feat_mean(df_label,date_hpg_reserve,["air_genre_name","air_area_name","weekday"],"reserve_visitors_sum","hpg_genre_area_weekday_mean_%s"%i)




    #月初月中月末
    df_label = feat_mean(df_label, df_train, ["air_store_id","day","weekday"], "visitors", "air_day_mean")
    df_label = feat_mean(df_label, df_train, ["air_store_id","day","holiday"], "visitors", "air_holiday_mean")
    for i in [21,35,63,140,280,350,420]:
        df_select=df_train[df_train.day_gap>=-i].copy()
        df_label=feat_mean(df_label,df_select,["air_store_id"],"visitors","air_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_store_id"],"visitors","air_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_store_id"],"visitors","air_min_%s"%i)
        df_label=feat_std(df_label,df_select,["air_store_id"],"visitors","air_std_%s"%i)
        df_label=feat_count(df_label,df_select,["air_store_id"],"visitors","air_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_min_%s"%i)
        df_label=feat_std(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_std_%s"%i)
        df_label=feat_count(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_min_%s"%i)
        df_label=feat_count(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_min_%s"%i)
        df_label=feat_count(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_min_%s"%i)
        df_label=feat_count(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_min_%s"%i)
        df_label=feat_count(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_mean_%s"%i)
        df_label=feat_max(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_max_%s"%i)
        df_label=feat_min(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_min_%s"%i)
        df_label=feat_count(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_count_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_area_name"],"visitors","air_area_name_mean_%s"%i)
        df_label=feat_sum(df_label,df_select,["air_area_name"],"visitors","air_area_name_sum_%s"%i)

        df_label=feat_mean(df_label,df_select,["air_genre_name"],"visitors","air_genre_name_mean_%s"%i)
        df_label=feat_sum(df_label,df_select,["air_genre_name"],"visitors","air_genre_name_sum_%s"%i)

        df_label = feat_mean(df_label, df_select, ["air_genre_name","air_area_name"], "visitors", "air_genre_area_name_mean_%s" % i)
        df_label = feat_sum(df_label, df_select, ["air_genre_name","air_area_name"], "visitors", "air_genre_area_name_sum_%s" % i)


    #add features
    #df_label['var_max_lat'] = df_label['latitude'].max() - df_label['latitude']
    #df_label['var_max_long'] = df_label['longitude'].max() - df_label['longitude']
    #df_label['lon_plus_lat'] = df_label['longitude'] + df_label['latitude']

    return df_label

t2017 = date(2017, 4, 23)
nday=14
#构造训练集
all_data=[]
for i in range(nday*1,nday*(30+1),nday):
    delta = timedelta(days=i)
    t_begin=t2017 - delta
    print(t_begin)
    df_train["day_gap"]=df_train["visit_date"].apply(lambda x:date_gap(x,t_begin))
    air_reserve["day_gap"]=air_reserve["reserve_date"].apply(lambda x:date_gap(x,t_begin))
    hpg_reserve["day_gap"]=hpg_reserve["reserve_date"].apply(lambda x:date_gap(x,t_begin))
    df_feature=df_train[df_train.day_gap<0].copy()
    df_air_reserve=air_reserve[air_reserve.day_gap<0].copy()
    df_hpg_reserve=hpg_reserve[hpg_reserve.day_gap<0].copy()
    df_label=df_train[(df_train.day_gap>=0)&(df_train.day_gap<nday)][["air_store_id","hpg_store_id","visit_date","day_gap","visitors"]].copy()
    train_data_tmp=create_features(df_label,df_feature,df_air_reserve,df_hpg_reserve)
    all_data.append(train_data_tmp)
train=pd.concat(all_data)


#构造线上测试集
t_begin=date(2017, 4, 23)
print(t_begin)
df_label=df_test.merge(store_id_relation,on="air_store_id",how="left")
df_label["day_gap"]=df_label["visit_date"].apply(lambda x:date_gap(x,t_begin))
df_train["day_gap"]=df_train["visit_date"].apply(lambda x:date_gap(x,t_begin))
air_reserve["day_gap"] = air_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))
hpg_reserve["day_gap"] = hpg_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))
df_label=df_label[["air_store_id","hpg_store_id","visit_date","day_gap"]].copy()
test=create_features(df_label,df_train,air_reserve,hpg_reserve)


#增加天气信息
weather=pd.read_csv("../input/weather.csv").fillna(0)
weather=weather.rename(columns={"calendar_date":"visit_date"})
#昨天的天气
#weather["precipitation_yes"]=weather.groupby("air_store_id")["precipitation"].shift(1)
#print weather[["air_store_id","visit_date","precipitation","precipitation_yes"]]
#weather[["air_store_id","visit_date","precipitation","precipitation_yes"]].to_csv("oooo.csv",index=None)
train = train.merge(weather, on=["air_store_id", "visit_date"], how="left").fillna(0)
test = test.merge(weather, on=["air_store_id", "visit_date"], how="left").fillna(0)

#train.to_csv("../stacking/train.csv",index=None)
#test.to_csv("../stacking/test.csv",index=None)

#取log
train["visitors"]=train["visitors"].apply(lambda x:np.log1p(float(x)) if float(x) > 0 else 0)

#训练预测
def stacking(clf,train_data,test_data,clf_name,class_num=1):
    train=np.zeros((train_data.shape[0],class_num))
    test=np.zeros((test_data.shape[0],class_num))
    test_pre=np.empty((folds,test_data.shape[0],class_num))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr=train_data.iloc[train_index]
        te=train_data.iloc[test_index]
        #分别测试分数
        te_1=te[te.day_gap<=6].copy()
        te_2=te[te.day_gap>6].copy()
        te_1_x=te_1.drop(["visitors"], axis=1)
        te_2_x=te_2.drop(["visitors"], axis=1)
        te_1_y=te_1["visitors"].values
        te_2_y=te_2["visitors"].values
        print(te_1.shape)
        print(te_2.shape)

        tr_x=tr.drop(["visitors"], axis=1)
        tr_y=tr['visitors'].values
        te_x=te.drop(["visitors"], axis=1)
        te_y=te['visitors'].values

        weight_train=weight_df.iloc[train_index]
        weight_test=weight_df.iloc[test_index]

        train_matrix = clf.Dataset(tr_x, label=tr_y,weight=weight_train["weight"])
        test_matrix = clf.Dataset(te_x, label=te_y,weight=weight_test["weight"])

        params = {
            'num_leaves': 2 ** 7 - 1,
            'objective': 'regression_l2',
            'max_depth': 8,
            'min_data_in_leaf': 50,
            'learning_rate': 0.01,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.75,
            'bagging_freq': 1,
            'metric': 'rmse',
            'num_threads': 4,
            'seed': 2018
        }

        num_round = 5000
        early_stopping_rounds = 300
        if test_matrix:
            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )
            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i, :]= model.predict(test_data, num_iteration=model.best_iteration).reshape((test_data.shape[0],1))
            pre_1=model.predict(te_1_x,num_iteration=model.best_iteration).reshape((te_1_x.shape[0],1))
            pre_2=model.predict(te_2_x,num_iteration=model.best_iteration).reshape((te_2_x.shape[0],1))
            cv_scores.append((mean_squared_error(te_y, pre)**0.5,mean_squared_error(te_1_y, pre_1)**0.5,mean_squared_error(te_2_y, pre_2)**0.5))

        print("%s now score is:"%clf_name,cv_scores)
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))

    score_split=(str(round(np.mean([i[0] for i in cv_scores]),6)),str(round(np.mean([i[1] for i in cv_scores]),6)),str(round(np.mean([i[2] for i in cv_scores]),6)))
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
        f.write("score_split:"+str(score_split)+"\n")

    return train.reshape(-1,class_num),test.reshape(-1,class_num),score_split


def lgb(train, valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm,train,valid,"lgb")
    return xgb_train, xgb_test,cv_scores


import lightgbm
from sklearn.cross_validation import KFold
folds = 5
seed = 2018

#生成数据
train_data = train.drop(["air_store_id","hpg_store_id","visit_date"], axis=1)
test_data = test.drop(["air_store_id","hpg_store_id","visit_date"], axis=1)

weight_df=train[["day_gap"]].copy()
weight_df["weight"]=weight_df["day_gap"].apply(lambda x:1 if x<=6 else 1)

kf = KFold(train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test,m=lgb(train_data,test_data)

#生成线下
train["visitors_pre"]=lgb_train
score_result=mean_squared_error(train["visitors"], train["visitors_pre"])**0.5
train["visitors"] = np.clip(np.expm1(train["visitors"]), 0, 1000)
train["visitors_pre"] = np.clip(np.expm1(train["visitors_pre"]), 0, 1000)
train[["air_store_id","visit_date","visitors","visitors_pre"]].to_csv("../offline/offline_cv_%s_%s_%s.csv"%m,index=None)
with open("score_cv.txt", "a") as f:
    f.write("result score is:" + str(score_result) + "\n")
#生成提交
df_test["visitors"]=lgb_test
df_test["visitors"] = np.clip(np.expm1(df_test["visitors"]), 0, 1000)
df_test[["id","visitors"]].to_csv("../sub/sub_plantsgo_cv_14_weight_%s_%s_%s.csv"%m,index=None)
