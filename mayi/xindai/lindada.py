import numpy as np
import pandas as pd
import datetime

data_root = "C:/Users/csw/Desktop/python/JD/xindai/data"
save_root = "{}/feat".format(data_root)

user_path = "{}/t_user.csv".format(data_root)
order_path = "{}/t_order.csv".format(data_root)

user_df = pd.read_csv(user_path)[["uid"]]
order_df = pd.read_csv(order_path)
print(len(order_df))
order_df["day"] = order_df.buy_time.str.split('-', expand=True)[2]
order_time = order_df.buy_time.values
train_time = ((order_time >= "2016-08-01") & (order_time <= "2016-10-31"))
test_time = ((order_time >= "2016-08-01") & (order_time <= "2016-11-30"))

train_buy_times = order_df.ix[train_time, ["uid", "day"]]
test_buy_times = order_df.ix[test_time, ["uid", "day"]]

train_buy_times = train_buy_times.groupby(["uid"], as_index=False).count()
test_buy_times = test_buy_times.groupby(["uid"], as_index=False).count()
train_buy_times = train_buy_times.groupby(["uid"], as_index=False).max().rename(columns={"day": "high_feq_buy_day"})
test_buy_times = test_buy_times.groupby(["uid"], as_index=False).max().rename(columns={"day": "high_feq_buy_day"})
train_feat = user_df.copy(deep=True)[["uid"]]
test_feat = user_df.copy(deep=True)[["uid"]]
train_feat = train_feat.merge(train_buy_times, on="uid", how="left").fillna(-1)
test_feat = test_feat.merge(test_buy_times, on="uid", how="left").fillna(-1)

train_feat.to_csv("{}/high_feq_buy_day_train.csv".format(save_root), index=False)
test_feat.to_csv("{}/high_feq_buy_day_test.csv".format(save_root), index=False)






