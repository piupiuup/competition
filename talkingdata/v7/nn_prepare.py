from talkingdata.v7.feat import *
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import pickle, gc, warnings, os, time
warnings.filterwarnings('ignore')
from talkingdata.v7.common import *
from talkingdata.v7.nn_utils import *
from argparse import ArgumentParser

def get_data():

    data = read(['click_id', 'date', 'hour', 'ip', 'os', 'device', 'app', 'channel', 'click_time', 'is_attributed'])
    train_feat = pd.DataFrame()
    for date in [0, 1, 2]:
        data_temp = data[(data['date'] == date) & (~data['is_attributed'].isnull())].copy()
        data_temp = data_temp[data_temp['is_attributed'] == 1].append(
            data_temp[data_temp['is_attributed'] == 0].sample(frac=0.05, random_state=66))
        train_feat_sub = make_feat(data_temp, '{}_7_all'.format(date))
        train_feat = train_feat.append(train_feat_sub)
        del train_feat_sub
        gc.collect()

    test_data = data[data['click_id'] > -1].copy()
    test_feat = make_feat(test_data, '3_7_6')

    predictors = [c for c in train_feat.columns if
                  c not in ['click_id', 'click_time',  'is_attributed', 'date', 'ip_os_rate']]
    index = list(range(len(train_feat)))
    import random
    random.shuffle(index)
    train_feat = train_feat.iloc[index].reset_index(drop=True)
    return train_feat[predictors],train_feat['is_attributed'],test_feat[predictors],test_feat['click_id']


parser = ArgumentParser()
parser.add_argument("--mode", default='cut', choices=['cut', 'all'])
parser.add_argument("--n_cuts", type=int, default=32)
args = parser.parse_args()
mode = args.mode

# path =
timer = Timer()

N_THREADS = 20
N_ITER = 1
N_FOLDS = 10
NEG_SAMPLE_RATE = 0.05
N_CUTS = args.n_cuts
if mode=='cut':
    # try:
    print("Load the training and testing data...")
    # with pd.HDFStore(path+'piupiu的四个变量.h5', "r") as hdf:

    train_df,train_y,test_df,test_id = get_data()
    train_df.fillna(-1,inplace=True)
    test_df.fillna(-1,inplace=True)
    gc.collect();
    train_df.info()
    timer.get_eclipse()
    # except:
    #     raise ValueError("Cannot find hdf5 data! \nPlease run lgb.py first to generate the data!")
    # new_id = ['ip_os', 'ip_app_device']
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    numerical = [x for x in train_df.columns if x not in categorical] #我这里认为 train_df 里面只有整数型特征和实数型特征，没有乱七八糟的别的，如果有的话在numerical里把它去掉
    important_features = ['ip_device_os_app_diff_time-1',
 'os_app_rate',
 'channel_app_rate',
 'minute15_attributed_mean',
 'ip_device_os_app_diff_time-2',
 'os_channel_rate',
 'ip&minute15_var',
 'os_app_count',
 'device_app_rate',
 'ip_device_os_channel_app_diff_time-1',
 'ip_device_rate',
 'ip_device_os_app_diff_time-3',
 'ip_rate',
 'ip&device_tfidf2',
 'app_attributed_mean',
 'a',
 'minute15_attributed_std',
 'device&app_tfidf2',
 'ipchannel_gini',
 'ip&app_tfidf2',
 'ip_attributed_std',
 'ip_os_plantsgo_19',
 'app_attributed_std',
 'os_app_count/os_count_rate',
 'app_rate',
 'app&hour_tfidf2',
 'minute15_rate',
 'os_attributed_mean',
 'ip_os_plantsgo_22',
 'ip_device_plantsgo_9',
 'ip_device_os_diff_time-1',
 'ip_device_os_app_diff_time1',
 'ip_app_diff_time-1',
 'ip_channel_rate',
 'channel_app_count/channel_count_rate',
 'ip_attributed_mean',
 'ip_channel_diff_time-1',
 'ip_app_plantsgo_9',
 'ipos_gini',
 'os_device_rate',
 'channel_attributed_mean',
 'ip_os_device_attributed_mean',
 'ip_device_os_diff_time-3',
 'ip&channel_tfidf2',
 'ip_os_device_attributed_std',
 'app',
 'ip_device_os_app_diff_time2',
 'date_minute15_count/minute15_count_rate',
 'ip_os_plantsgo_13',
 'date_hour_ip_device_count/date_hour_ip_count_rate']
    predictors = categorical + numerical

    print('Transform the data into a DL-friendly version...')

    print("Label encoding....")
    cate_dict = {}
    for col in categorical:
        tmp = train_df[col].append(test_df[col])
        tmp = tmp.astype('category').cat.codes
        cate_dict[col] = tmp.max()+1
        train_df[col], test_df[col] = tmp[:len(train_df)], tmp[-len(test_df):]
    timer.get_eclipse()
    print("Starting to encode and normalize numerical features... ")
    print("N_CUTS:", N_CUTS)

    for col in numerical:

        if col in important_features:
            print("Encoding numerical feature {}...".format(col))
            train_df[col+'_encoded'], test_df[col+'_encoded'] = get_quantile_encoding([train_df[col], test_df[col]], N_CUTS)
        cate_dict[col] = N_CUTS
        print("Normalizing {}...".format(col))
        tmp = np.concatenate([train_df[col].values, test_df[col].values])
        tmp = rank_gauss_normalization(tmp)
        train_df[col], test_df[col] = tmp[:len(train_df)], tmp[-len(test_df):]
    timer.get_eclipse()

    # with pd.HDFStore(data_path+'nn_data_piupiu.h5'.format(N_CUTS), "w") as hdf:
    #     hdf.put('train_df', train_df)
    #     hdf.put('train_y', train_y)
    #     hdf.put('test_df', test_df)
    #     hdf.put('test_id', test_id)
timer.get_eclipse()
print('Done!')
