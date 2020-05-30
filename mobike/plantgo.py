#encoding=utf8
import pandas as pd
import geohash
import numpy as np
import gc
import lightgbm as lgb

train_path = r'C:\Users\csw\Desktop\python\mobike\data\train.csv'
test_path = r'C:\Users\csw\Desktop\python\mobike\data\test.csv'

def data_process():  #数据预处理
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    test["geohashed_end_loc"] = np.nan
    all=train.append(test)

    all["starttime"]=pd.to_datetime(all["starttime"])
    all["day"]=all["starttime"].dt.day
    all["hour"]=all["starttime"].dt.hour
    all["minute"]=all["starttime"].dt.minute
    all["minute_from"]=all["hour"]*60+all["minute"]
    all["work_day"]=all["day"].apply(lambda x:1 if x in [13,14,20,21,28,29,30] else 0)
    all["lon_start"]=all["geohashed_start_loc"].map(lambda x:round(geohash.decode_exactly(x)[1],7))
    all["lat_start"]=all["geohashed_start_loc"].map(lambda x:round(geohash.decode_exactly(x)[0],7))

    def split(x):
        if x<5:return 0
        if x<10:return 1
        if x<14:return 2
        if x<18:return 3
        if x<21:return 4
        return 5
    all["period"]=all["hour"].apply(split)
    return all

def top(df,names,pre,col,n=3):  #生成候选集
    columns=names+[col]
    start_end = pd.DataFrame(df.groupby(columns).orderid.count()).reset_index()
    start_end.columns = columns+["end_count"]
    print (start_end.head())
    start_end = start_end.sort_values(names+["end_count"], ascending=False)
    print (start_end.head())
    if n>0:
        user_end = pd.DataFrame(start_end.groupby(names)[col].apply(lambda x: ",".join(list(x)[:n]))).reset_index()
    else:
        user_end = pd.DataFrame(start_end.groupby(names)[col].apply(lambda x: ",".join(list(x)))).reset_index()
    user_end.columns = names+[pre]
    return user_end

def create_sample(df):  #生成样本
    order_list = []
    end_list = []
    i = 0
    for row in df.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order = row.orderid
        end = row.predict
        order_list += [order] * len(end)
        end_list += end
    create_df = pd.DataFrame({'orderid':order_list,'geohashed_end_loc':end_list})
    return create_df

def predict(a,b,c):
    #start=["wx4f9mk","wx4f9ky","wx4f9ms"]
    start=[]
    for ii in [a,b]:
        ii=ii.split(",")
        for jj in ii:
            if jj:
                start.append(jj)
    result=list(set(start))

    if len(result)<4:
        for _ in c:   #如果用户交互过少才添加起点候选
            if _ not in result:
                result.append(_)

    if len(result)<4:
        for _ in ["wx4f9mk","wx4f9ky","wx4f9ms"]:   #瞎填充
            if _ not in result:
                result.append(_)
    return result

#添加特征
def num(df,names,pre):
    columns=names+["geohashed_end_loc"]
    start_end = pd.DataFrame(df.groupby(columns).orderid.count()).reset_index()
    start_end.columns = columns+[pre]
    return start_end

def feat(df_ori,df_feat,names,col):
    add=num(df_feat,names,col)
    df_ori=df_ori.merge(add,on=names+["geohashed_end_loc"],how="left").fillna(0)
    return df_ori
###############################
def feat_count(df, df_feature, fe,value):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_median(df, df_feature, fe,value):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

all=data_process()

#排除需要统计的那一天，一天天构造样本
def train_set(day,label=True):
    select=all[all.day==day].copy()
    if label:
        true = select[["orderid", "geohashed_end_loc"]].copy()  # 用来打标签和做线下评分
        true.columns = ["orderid", "label"]
    else:
        true=None
    select = select[["orderid", "userid", "day", "hour", "minute", "work_day", "period", "geohashed_start_loc", "lon_start","lat_start"]].copy()
    select_out=all[all.day!=day].copy()
    user_end_loc=top(select_out,["userid"],"user_end","geohashed_end_loc",-1)  #用户交互过的起点
    user_start_loc=top(select_out,["userid"],"user_start","geohashed_start_loc",-1)    #用户交互过的终点
    start_day_period=top(select_out,["geohashed_start_loc","work_day","period"],"start_day_period_end","geohashed_end_loc")  #日阶段起点最多的top3
    hot=pd.read_csv("loc_matrix.csv")
    start_hot = pd.DataFrame(hot.groupby("geohashed_start_loc")["geohashed_end_loc"].apply(lambda x: ",".join(list(x)))).reset_index()

    select=select.merge(user_end_loc,on="userid",how="left").fillna("")
    select=select.merge(user_start_loc,on="userid",how="left").fillna("")
    select=select.merge(start_day_period,on=["geohashed_start_loc","work_day","period"],how="left").fillna("")
    select["predict"] = map(lambda a,b,c:predict(a,b,c),select["user_end"], select["user_start"], select["start_day_period_end"])
    sample = select[["orderid", "predict"]].copy()
    sample = create_sample(sample)
    sample = sample.merge(select, on="orderid", how="left")

    #删除start==end
    sample = sample[sample.geohashed_end_loc!= sample.geohashed_start_loc]

    #构造特征
    end_loc = sample[["geohashed_end_loc"]].copy().drop_duplicates()
    end_loc["lon_end"] = end_loc["geohashed_end_loc"].map(lambda x: round(geohash.decode_exactly(x)[1], 7))
    end_loc["lat_end"] = end_loc["geohashed_end_loc"].map(lambda x: round(geohash.decode_exactly(x)[0], 7))
    sample = sample.merge(end_loc, on="geohashed_end_loc", how="left")
    #打标签
    if label:
        sample = sample.merge(true, on="orderid", how="left")
        sample["label"] = map(lambda a, b: 1 if a == b else 0, sample["label"], sample["geohashed_end_loc"])

    #####################################################转化率特征
    sample = feat(sample, select_out, ["userid", "work_day", "period"], "num_0")
    sample = feat(sample, select_out, ["userid", "geohashed_start_loc", "work_day", "hour"], "num_1")
    sample = feat(sample, select_out, ["userid", "geohashed_start_loc", "work_day"], "num_2")
    sample = feat(sample, select_out, ["userid", "geohashed_start_loc"], "num_3")
    sample = feat(sample, select_out, ["userid", "period"], "num_4")
    sample = feat(sample, select_out, ["userid", "hour"], "num_5")
    sample = feat(sample, select_out, ["geohashed_start_loc", "period"], "num_6")
    sample = feat(sample, select_out, ["geohashed_start_loc", "hour"], "num_7")
    sample = feat(sample, select_out, ["userid", "geohashed_start_loc", "period"], "num_8")

    #####################################################用户特征
    sample = feat_count(sample, select_out, ["userid"], "geohashed_end_loc")
    sample = feat_nunique(sample, select_out, ["userid"], "geohashed_start_loc")
    sample = feat_nunique(sample, select_out, ["userid"], "geohashed_end_loc")

    #################################################起点特征
    # 和start的交互次数2倍
    select_out_1 = select_out[["orderid", "userid", "geohashed_start_loc", "geohashed_end_loc"]].copy()
    select_out_2 = select_out_1.copy()
    select_out_2.columns = ["orderid", "userid", "geohashed_end_loc", "geohashed_start_loc"]
    select_out_times = select_out_1.append(select_out_2)
    sample = feat(sample, select_out_times, ["geohashed_start_loc"], "times_0")
    sample = feat(sample, select_out_times, ["userid", "geohashed_start_loc"], "times_1")  # 用户和该点交互的次数
    sample = feat(sample, select_out_2, ["userid", "geohashed_start_loc"], "times_2")  # 用户从改点出发的次数
    del select_out_1
    del select_out_2
    del select_out_times
    gc.collect()

    #################################################时空特征
    # 该地该时刻的总的车辆数--线上需要除以总天数
    #"""
    sample=feat_count(sample,select_out,["work_day","hour","geohashed_start_loc"],"geohashed_end_loc")  #起始点该时刻的发车数
    sample=feat_count(sample,select_out,["work_day","hour","geohashed_end_loc"],"geohashed_start_loc")  #终点此刻的入车数
    sample=feat_count(sample,select_out,["work_day","period","geohashed_start_loc"],"geohashed_end_loc")  #起始点该阶段的发车数
    sample=feat_count(sample,select_out,["work_day","period","geohashed_end_loc"],"geohashed_start_loc")  #终点该阶段的入车数

    select_out_1=select_out[["work_day","period","hour","geohashed_start_loc","geohashed_end_loc"]].copy()
    select_out_1.columns=["work_day","period","hour","geohashed_end_loc","geohashed_start_loc"]
    sample=feat_count(sample,select_out_1,["work_day","hour","geohashed_start_loc"],"geohashed_end_loc")  #起始点该时刻的入车数
    sample=feat_count(sample,select_out_1,["work_day","hour","geohashed_end_loc"],"geohashed_start_loc")  #终点此刻的出车数
    sample=feat_count(sample,select_out_1,["work_day","period","geohashed_start_loc"],"geohashed_end_loc")  #起始点该时刻的入车数
    sample=feat_count(sample,select_out_1,["work_day","period","geohashed_end_loc"],"geohashed_start_loc")  #终点此刻的出车数

    del select_out_1
    gc.collect()



    #"""
    index=sample[["orderid","geohashed_end_loc"]].copy()
    del sample["geohashed_end_loc"]
    del sample["geohashed_start_loc"]
    del sample["user_end"]
    del sample["user_start"]
    del sample["start_day_period_end"]
    del sample["predict"]
    gc.collect()
    return sample,true,index

test_18,true_18,index_18=train_set(18)
test_19,true_19,index_19=train_set(19)
test_20,true_20,index_20=train_set(20)
test_21,true_21,index_21=train_set(21)
test_22,true_22,index_22=train_set(22)
test_23,true_23,index_23=train_set(23)
test_24,true_24,index_24=train_set(24)

train=pd.concat([test_18,test_19,test_20,test_21])
test=pd.concat([test_22,test_23,test_24])
true=pd.concat([true_22,true_23,true_24])
index=pd.concat([index_22,index_23,index_24])

for i in [test_18,true_18,index_18,
           test_19,true_19,index_19,
           test_20,true_20,index_20,
           test_21,true_21,index_21,
           test_22,true_22,index_22,
           test_23,true_23,index_23,
           test_24,true_24,index_24
          ]:
    del i
gc.collect()

train_y=train["label"]
del train["label"]

test_y=test["label"]
del test["label"]
test_pre=test.copy()

#lgb算法
train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    #'metric': 'auc',
    'min_child_weight': 1.5,
    'num_leaves': 2 ** 5,
    'lambda_l2': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'learning_rate': 0.2,
    'tree_method': 'exact',
    'seed': 2017,
    'nthread': 12,
    'silent': True,
}
num_round = 3000
early_stopping_rounds = 100
model = lgb.train(params, train, num_round, valid_sets=test,
                      early_stopping_rounds=early_stopping_rounds,
                      )
pre = model.predict(test_pre, num_iteration=model.best_iteration)
index["predict"]=pre


result=index.sort_values(["orderid","predict"], ascending=False)
pre=pd.DataFrame(result.groupby(["orderid"])["geohashed_end_loc"].apply(lambda x:list(x)[:3])).reset_index()
pre.columns=["orderid","predict"]

result=pre.merge(true,on="orderid",how="left")
result["label"]=result["label"].apply(lambda x:[x])

actual=list(result["label"])
predicted=list(result["predict"])
s= mapk(actual,predicted,3)
print s
with open("score_list.txt","a") as f:
    f.write(str(s)+"\n")

del train
del test
gc.collect()

#预测线上
valid=[]
inx=[]
for d in range(25,32):
    v, t, i = train_set(d,False)
    valid.append(v)
    inx.append(i)
valid=pd.concat(valid)
inx=pd.concat(inx)

pre = model.predict(valid, num_iteration=model.best_iteration)
inx["predict"]=pre

result=inx.sort_values(["orderid","predict"], ascending=False)
pre=pd.DataFrame(result.groupby(["orderid"])["geohashed_end_loc"].apply(lambda x:list(x)[:3])).reset_index()
pre.columns=["orderid","predict"]
pre["p1"]=pre["predict"].apply(lambda x:x[0])
pre["p2"]=pre["predict"].apply(lambda x:x[1])
pre["p3"]=pre["predict"].apply(lambda x:x[2])
save=pre[["orderid","p1","p2","p3"]].copy()
save.to_csv("../sub/model_v4.csv",index=None,header=False)

