import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from multiprocessing import Pool
import pickle
import gc


def lgb_modelfit_nocv(params, dtrain_x, dtrain_y, dvalid_x, dvalid_y, predictors, objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10,
                      categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.01,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 20,
        'verbose': 0,
        'metric': metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain_x[predictors].values, label=dtrain_y,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid_x[predictors].values, label=dvalid_y,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics + ":", evals_results['valid'][metrics][n_estimators - 1])

    return bst1, evals_results['valid'][metrics][n_estimators - 1]


path = '../'

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

try:
    print("Load train...")
    with open(path + "pickle/offline_train_6hour.pkl", "rb") as fp:
        train_df = pickle.load(fp)
        len_train = len(train_df)
    print("Load valid...")
    with open(path + "pickle/offline_valid_6hour.pkl", "rb") as fp:
        train_df = train_df.append(pickle.load(fp))
        len_valid = len(train_df) - len_train
    print("Load test...")
    with open(path + "pickle/test.pkl", "rb") as fp:
        test_df = pickle.load(fp)
        len_test = len(test_df)
    train_y = train_df['is_attributed'].values
    test_id = test_df['click_id'].values
    train_df.drop('is_attributed', axis=1, inplace=True)
    test_df.drop('click_id', axis=1, inplace=True)
    print("Data split:", len_train, len_valid, len_test)
    train_df = train_df.append(test_df)
    train_df.reset_index(drop=True, inplace=True)
    del test_df
    gc.collect()
except:
    raise ValueError("Cannot find pickle data! Please generate them by running pickle_data.py!")

from com_util import *

fe_add = []
attributed_data = pd.read_csv("../features/attributed_data.csv")
train_df = feat_mean(train_df, attributed_data, ["ip", "day"], "attributed_gap", "fe1")
train_df = feat_mean(train_df, attributed_data, ["ip", "os", "device", "day"], "attributed_gap", "fe2")


def multi_granularity_gap(train_df, outer_key, inner_key, fill_na=999):
    try:
        assert train_df.index.nunique() == len(train_df)
    except:
        raise ValueError("Indexs of the data should be unique!")
    assert type(outer_key) is list and type(inner_key) is list

    filename = "_".join(['inout_'] + outer_key + inner_key + [str(len(train_df))]) + ".pkl"
    try:
        with open(path + "pickle/" + filename, "rb") as fp:
            print("Load inout from pickle file, key: {}...".format(outer_key + inner_key))
            col = pickle.load(fp)
        train_df = pd.concat([train_df, col], axis=1)
        gc.collect()
    except:
        c1 = 'last_gap_O_{}_I_{}'.format('_'.join(outer_key), '_'.join(inner_key))
        c2 = 'next_gap_O_{}_I_{}'.format('_'.join(outer_key), '_'.join(inner_key))
        train_df['tmp'] = 1
        tmp_1 = train_df.groupby(outer_key)[['tmp']].cumsum()
        train_df.drop('tmp', axis=1, inplace=True)
        train_df["zx"] = tmp_1['tmp']  # 正序

        gp = train_df.groupby(outer_key + inner_key)["zx"]
        last_time, next_time = gp.shift(1), gp.shift(-1)
        tmp = train_df["zx"] - last_time - 1
        tmp.fillna(fill_na, inplace=True)
        train_df[c1] = tmp.astype("uint16")
        tmp = -(train_df["zx"] - next_time) - 1
        tmp.fillna(fill_na, inplace=True)
        train_df[c2] = tmp.astype("uint16")
        train_df.drop(['zx'], axis=1, inplace=True)
        with open(path + "pickle/" + filename, "wb") as fp:
            col = train_df[[c1, c2]]
            pickle.dump(col, fp)
    return train_df


# train_df=multi_granularity_gap(train_df,outer_key=["ip","device","os"],inner_key=["app"])
# train_df=multi_granularity_gap(train_df,outer_key=["ip","device","os"],inner_key=["channel"])


def merge_id(list_of_col):
    ans = list_of_col[0]
    for t in list_of_col[1:]:
        ans = ans * t % 2000000000
    return ans.astype('uint32')


new_id = ['ip_os', 'ip_app_device']
train_df[new_id[0]] = merge_id([train_df['ip'], train_df['os']])
train_df[new_id[1]] = merge_id([train_df['ip'], train_df['app'], train_df['device']])


def get_id_cnt(train_df, id_col):
    cnt = train_df[id_col].value_counts().reset_index()
    cnt.columns = [id_col, id_col + '_cnt']
    return train_df.merge(cnt, how='left', on=id_col)


def get_cnt_feature(train_df, key, on, rank=False):
    filename = "_".join(["cnt_features", key, on, str(len(train_df)), str(rank)]) + ".pkl"
    try:
        with open(path + "pickle/" + filename, "rb") as fp:
            print("load count feature from pickle file: key: {}, on: {}...".format(key, on))
            col = pickle.load(fp)
        train_df = pd.concat([train_df, col], axis=1)
        gc.collect()
    except:
        print('get count features, key: {}, on: {}...'.format(key, on))
        cnt = train_df[[key, on, 'click_time']].groupby(by=[key, on])[['click_time']].count().reset_index().rename(
            index=str, columns={'click_time': key + '_cnt_' + on})
        cnt_all = train_df[[key, on]].groupby([on])[[key]].count().reset_index().rename(index=str,
                                                                                        columns={key: 'all_cnt_' + on})
        cnt[key + '_cnt_' + on] = cnt[key + '_cnt_' + on].astype('uint16')
        cnt[key + '_percent_' + on] = (cnt[key + '_cnt_' + on] * 100 / cnt_all['all_cnt_' + on]).astype(np.float32)
        new_col = [key + '_cnt_' + on, key + '_percent_' + on]
        if rank:
            new_col.append(key + '_rank_' + on)
            tmps = []
            for k in cnt[on].drop_duplicates():
                tmp = cnt[cnt[on] == k][[key, key + '_percent_' + on]].drop_duplicates(key)
                tmp.sort_values(key + '_percent_' + on, inplace=True)
                tmp[key + '_rank_' + on] = (np.arange(len(tmp)) / len(tmp)).astype(np.float32)
                # tmp[key+'_quantile_'+on] = np.cumsum(tmp[key+'_percent_'+on])
                tmp.drop(key + '_percent_' + on, axis=1, inplace=True)
                tmp[on] = k
                tmps.append(tmp)
            tmp = pd.concat(tmps)
            cnt = cnt.merge(tmp, on=[key, on], how='left')
        train_df = train_df.merge(cnt, on=[key, on], how='left', copy=False)
        with open(path + "pickle/" + filename, "wb") as fp:
            col = train_df[new_col]
            pickle.dump(col, fp)

    return train_df


train_df['dayhour'] = (train_df['day'] * 24 + train_df['hour']).astype('uint16')
train_df = get_cnt_feature(train_df, 'ip', 'dayhour', rank=True)
train_df = get_cnt_feature(train_df, 'ip', 'day', rank=True)
train_df['ten_min'] = train_df['day'] * 24 * 6 + train_df['hour'] * 6 + train_df['ten_min']
train_df = get_cnt_feature(train_df, 'ip', 'ten_min', rank=True)
train_df = get_cnt_feature(train_df, 'ip', 'app', rank=False)


def get_timegaps(train_df, key=[], time_col='timestamp', fill_na=99999):
    assert type(key) is list
    filename = "_".join(['timegaps_'] + key + [str(len(train_df))]) + ".pkl"
    try:
        with open(path + "pickle/" + filename, "rb") as fp:
            print("Load time gaps from pickle file, key: {}...".format(key))
            col = pickle.load(fp)
        train_df = pd.concat([train_df, col], axis=1)
        gc.collect()
    except:
        print("get time gaps on {}...".format(key))
        gp = train_df.groupby(key)[time_col]
        last_time, next_time = gp.shift(1), gp.shift(-1)
        tmp = train_df[time_col] - last_time
        tmp.fillna(fill_na, inplace=True)
        train_df['last_timegap_' + '_'.join(key)] = tmp.astype(np.float32)
        tmp = -(train_df[time_col] - next_time)
        tmp.fillna(fill_na, inplace=True)
        train_df['next_timegap_' + '_'.join(key)] = tmp.astype(np.float32)

        with open(path + "pickle/" + filename, "wb") as fp:
            col = train_df[['last_timegap_' + '_'.join(key), 'next_timegap_' + '_'.join(key)]]
            pickle.dump(col, fp)
    return train_df


train_df["timestamp"] = train_df["click_time"].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
train_df = get_timegaps(train_df, key=["ip"], time_col="timestamp")
# train_df=get_timegaps(train_df,key=["ip","device"],time_col="timestamp")
# train_df=get_timegaps(train_df,key=["ip","os"],time_col="timestamp")
train_df = get_timegaps(train_df, key=["ip", "device", "os"], time_col="timestamp")
train_df = get_timegaps(train_df, key=["ip", "device", "os", "channel"], time_col="timestamp")
train_df = get_timegaps(train_df, key=["ip", "device", "os", "app"], time_col="timestamp")


# train_df=get_timegaps(train_df,key=["ip","app"],time_col="timestamp")
# train_df=get_timegaps(train_df,key=["ip","channel"],time_col="timestamp")

def get_inout(train_df, key=[], time_col='', fill_na=-1):
    assert type(key) is list
    filename = "_".join(['inout_'] + key + [time_col] + [str(len(train_df))]) + ".pkl"
    try:
        with open(path + "pickle/" + filename, "rb") as fp:
            print("Load inout from pickle file, key: {}...".format(key + [time_col]))
            col = pickle.load(fp)
        train_df = pd.concat([train_df, col], axis=1)
        gc.collect()
    except:
        print("get inout on {}...".format(key + [time_col]))
        gp = train_df.groupby(key)[time_col]
        last_time, next_time = gp.shift(1).fillna(fill_na), gp.shift(-1).fillna(fill_na)
        # tmp = train_df[time_col] - last_time
        # tmp.fillna(fill_na, inplace=True)
        train_df['last_inout_' + '_'.join(key + [time_col])] = last_time.astype(
            'uint16')  # apply(lambda x:0 if x==0 else 1).astype('uint8')
        # tmp = -(train_df[time_col] - next_time)
        # tmp.fillna(fill_na, inplace=True)
        train_df['next_inout_' + '_'.join(key + [time_col])] = next_time.astype(
            'uint16')  # apply(lambda x:0 if x==0 else 1).astype('uint8')

        with open(path + "pickle/" + filename, "wb") as fp:
            col = train_df[['last_inout_' + '_'.join(key + [time_col]), 'next_inout_' + '_'.join(key + [time_col])]]
            pickle.dump(col, fp)
    return train_df


for col in new_id:
    train_df = get_cnt_feature(train_df, col, 'dayhour', rank=True)
    train_df = get_cnt_feature(train_df, col, 'day', rank=True)
    train_df = get_cnt_feature(train_df, col, 'ten_min', rank=True)
    gc.collect()


def get_appear_feature(train_df, key, on):
    filename = "_".join(["appear_features", key, on, str(len(train_df))]) + ".pkl"
    try:
        with open(path + "pickle/" + filename, "rb") as fp:
            print("load count feature from pickle file: key: {}, on: {}...".format(key, on))
            col = pickle.load(fp)
        train_df = pd.concat([train_df, col], axis=1)
        gc.collect()
    except:
        print('get appearing feature, key: {}, on: {}'.format(key, on))
        tmp = train_df[[key, on]].groupby(key)[[on]].nunique()
        tmp = tmp.reset_index().rename(index=str, columns={on: key + '_appear_' + on})
        tmp = tmp[[key, key + '_appear_' + on]]
        tmp[key + '_appear_' + on] = tmp[key + '_appear_' + on].astype('uint16')
        train_df = train_df.merge(tmp, on=key, how='left', copy=False)
        with open(path + "pickle/" + filename, "wb") as fp:
            col = train_df[[key + '_appear_' + on]]
            pickle.dump(col, fp)

    return train_df


train_df = get_appear_feature(train_df, 'ip', 'dayhour')
train_df = get_appear_feature(train_df, 'ip', 'ten_min')
train_df = get_appear_feature(train_df, 'ip', 'device')
train_df = get_appear_feature(train_df, 'ip', 'app')

gc.collect()
for col in new_id:
    # train_df = get_appear_feature(train_df, col, 'ten_min')
    gc.collect();


def get_rolling_count(train_df, key=[]):
    filename = "_".join(['rolling_'] + key + [str(len(train_df))]) + ".pkl"
    try:
        with open(path + "pickle/" + filename, "rb") as fp:
            print("Load rolling count from pickle file, key: {}...".format(key))
            col = pickle.load(fp)
        train_df = pd.concat([train_df, col], axis=1)
        gc.collect()
    except:
        print("get rolling count on {}...".format(key))
        train_df['tmp'] = 1
        tmp_1 = train_df.groupby(key)[['tmp']].cumsum()
        tmp_2 = train_df[::-1].groupby(key)[['tmp']].cumsum()[::-1]
        train_df.drop('tmp', axis=1, inplace=True)
        train_df['rolling_cnt_' + '_'.join(key)] = tmp_1['tmp'].astype('uint16')
        train_df['rolling_cnt_r_' + '_'.join(key)] = tmp_2['tmp'].astype('uint16')
        with open(path + "pickle/" + filename, "wb") as fp:
            col = train_df[['rolling_cnt_' + '_'.join(key), 'rolling_cnt_r_' + '_'.join(key)]]
            pickle.dump(col, fp)
    return train_df


# train_df = get_rolling_count(train_df, ['ip', 'device'])
# train_df = get_rolling_count(train_df, ['ip', 'os'])
train_df = get_rolling_count(train_df, ['ip', 'os', 'device'])
train_df = get_rolling_count(train_df, ['ip', 'os', 'device', 'hour'])
train_df = get_rolling_count(train_df, ['ip', 'app', 'device', 'day'])
train_df = get_rolling_count(train_df, ['ip', 'os', 'device', 'day', 'hour'])
train_df = get_rolling_count(train_df, ['ip', 'os', 'device', 'day', 'hour', 'minute'])
train_df = get_rolling_count(train_df, ['ip', 'os', 'device', 'day', 'hour', 'ten_min'])
train_df = get_rolling_count(train_df, ["ip", "device", "os", "app", "day", "hour", "minute"])
train_df = get_rolling_count(train_df, ["ip", "device", "os", "channel", "day", "hour", "minute"])
train_df = get_rolling_count(train_df, ["ip", "device", "os", "app", "day", "hour"])
train_df = get_rolling_count(train_df, ["ip", "device", "os", "channel", "day", "hour"])
train_df = get_rolling_count(train_df, ["ip", "device", "os", "app", "day", "hour", "ten_min"])
train_df = get_rolling_count(train_df, ["ip", "device", "os", "channel", "day", "hour", "ten_min"])

train_df.info()

target = 'is_attributed'
predictors = ['app', 'device', 'os', 'channel', 'hour',
              'ip_cnt_dayhour', 'ip_percent_dayhour', 'ip_rank_dayhour',
              'ip_cnt_app',  # 'ip_app_os_count',
              'ip_cnt_ten_min', 'ip_rank_ten_min',
              'ip_percent_day', 'ip_rank_day',
              'ip_appear_device', 'ip_appear_ten_min', 'ip_appear_app',
              ] + sum([[
    new_id[c] + '_cnt_dayhour', new_id[c] + '_percent_dayhour', new_id[c] + '_rank_dayhour',
    new_id[c] + '_percent_day', new_id[c] + '_rank_day',
    new_id[c] + '_cnt_ten_min', new_id[c] + '_rank_ten_min',
] for c in [0, 1]], []) + [
                 # 'rolling_cnt_ip_device', 'rolling_cnt_r_ip_device',
                 # 'rolling_cnt_ip_os', 'rolling_cnt_r_ip_os',
                 'rolling_cnt_ip_os_device', 'rolling_cnt_r_ip_os_device',
                 'rolling_cnt_ip_os_device_hour', 'rolling_cnt_r_ip_os_device_hour',
                 'rolling_cnt_ip_app_device_day', 'rolling_cnt_r_ip_app_device_day',
                 'rolling_cnt_ip_device_os_app_day_hour', 'rolling_cnt_r_ip_device_os_app_day_hour',
                 'rolling_cnt_ip_device_os_channel_day_hour', 'rolling_cnt_r_ip_device_os_channel_day_hour',
                 'rolling_cnt_ip_device_os_app_day_hour_ten_min', 'rolling_cnt_r_ip_device_os_app_day_hour_ten_min',
                 'rolling_cnt_ip_device_os_channel_day_hour_ten_min',
                 'rolling_cnt_r_ip_device_os_channel_day_hour_ten_min',
                 'rolling_cnt_ip_os_device_day_hour', 'rolling_cnt_r_ip_os_device_day_hour',
                 'rolling_cnt_ip_os_device_day_hour_minute', 'rolling_cnt_r_ip_os_device_day_hour_minute',
                 'rolling_cnt_ip_os_device_day_hour_ten_min', 'rolling_cnt_r_ip_os_device_day_hour_ten_min',
                 'rolling_cnt_ip_device_os_app_day_hour_minute', 'rolling_cnt_r_ip_device_os_app_day_hour_minute',
                 'rolling_cnt_ip_device_os_channel_day_hour_minute',
                 'rolling_cnt_r_ip_device_os_channel_day_hour_minute',

                 'last_timegap_ip', 'next_timegap_ip',
                 'last_timegap_ip_device_os', 'next_timegap_ip_device_os',
                 'last_timegap_ip_device_os_channel', 'next_timegap_ip_device_os_channel',
                 'last_timegap_ip_device_os_app', 'next_timegap_ip_device_os_app',

             ] + fe_add
categorical = ['app', 'device', 'os', 'channel', 'hour']

test_df = train_df[-len_test:]
train_df = train_df[:-len_test]
sub = pd.DataFrame()
sub['click_id'] = test_id


def stacking(clf, train_x, train_y, test_x, clf_name, class_num=1):
    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))
    test_pre = np.empty((folds, test_x.shape[0], class_num))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf):
        if i != 4:
            continue
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        train_matrix = clf.Dataset(tr_x, label=tr_y)
        test_matrix = clf.Dataset(te_x, label=te_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'min_child_weight': 1.5,
            'num_leaves': 2 ** 5,
            'lambda_l2': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'colsample_bylevel': 0.5,
            'learning_rate': 0.2,
            'seed': 2017,
            'nthread': 16,
            # 'scale_pos_weight':99, # feiyang: fixing unbalanced data
            'silent': True,
        }

        num_round = 15000
        early_stopping_rounds = 100
        if test_matrix:
            model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )
            pre = model.predict(te_x, num_iteration=model.best_iteration).reshape((te_x.shape[0], 1))
            train[test_index] = pre
            test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0], 1))
            cv_scores.append(roc_auc_score(te_y, pre))

        print("%s now score is:" % clf_name, cv_scores)
        # break
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:" % clf_name + str(np.mean(cv_scores)) + "\n")
    return train.reshape(-1, class_num), test.reshape(-1, class_num), np.mean(cv_scores)


def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test, cv_scores = stacking(lightgbm, x_train, y_train, x_valid, "lgb")
    return xgb_train, xgb_test, cv_scores


import lightgbm
from sklearn.cross_validation import KFold

folds = 5
seed = 2018

kf = KFold(train_df.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test, m = lgb(train_df[predictors].values, train_y, test_df[predictors].values)
print(lgb_test)
# train=pd.DataFrame()
# train["is_attributed"]=lgb_train
# train.to_csv("../stacking/train_model_v8.csv",index=None)

sub["is_attributed"] = lgb_test
sub[["click_id", "is_attributed"]].to_csv("../stacking/test_model_v8.csv", index=None)