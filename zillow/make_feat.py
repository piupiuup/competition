import os
import pickle
import geohash
import numpy as np
import pandas as pd

cache_path = 'F:/zillow_cache/'
data_path = 'C:/Users/csw/Desktop/python/zillow/data/'
prop_path = data_path + 'properties_2016.csv'
sample_path = data_path + 'sample_submission.csv'
train_path = data_path + 'train_2016_v2.csv'


#获取parcel对应的地点
def get_parcel_geohash_dict(precision=7):
    result_path = cache_path + 'parcel_geohash_dict.pkl'
    if os.path.exists(result_path):
        parcel_geohash_dict = pickle.load(open(result_path, 'rb+'))
    else:
        parcel_geohash_dict = {}
        prop = pd.read_csv(prop_path)
        for row in prop.itertuples():
            lat = row.latitude/1000000
            lon = row.longitude/1000000
            try:
                parcel_geohash_dict[row.parcelid] = geohash.encode(lat,lon,precision)
            except:
                pass
        pickle.dump(parcel_geohash_dict,open(result_path, 'wb+'))
    return parcel_geohash_dict

# 周围的8个地点加自己DataFrame
def get_near_loc():
    result_path = cache_path + 'near_loc.hdf'
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        import geohash
        prop = pd.read_csv(prop_path)
        parcel_geohash_dict = get_parcel_geohash_dict(7)
        prop['geohash'] = prop['parcelid'].map(parcel_geohash_dict)
        loc_list = prop['geohashed_start_loc'].tolist()
        loc_list = np.unique(loc_list)
        result = []
        for loc in loc_list:
            if loc is np.nan:
                continue
            nlocs = geohash.neighbors(loc)
            nlocs.append(loc)
            for nloc in nlocs:
                result.append([loc, nloc])
        result = pd.DataFrame(result, columns=['loc', 'near_loc'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 附近地点logerror
def get_ex_near_loc_error(data):
    key = data.memory_usage().sum()
    result_path = cache_path + 'near_loc_error{}.hdf'.format(key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        prop = pd.read_csv(prop_path)
        parcel_geohash_dict = get_parcel_geohash_dict(7)
        data_temp = data[['parcelid','logerror']].copy()
        data_temp['geohash'] = data_temp['parcelid'].map(parcel_geohash_dict)
        prop['geohash'] = prop['parcelid'].map(parcel_geohash_dict)
        loc_error = data_temp.groupby('geohash',as_index=False)['logerror'].agg({'sum_loc_error': 'sum',
                                                                                     'loc_count': 'count'})
        near_loc = get_near_loc()
        loc_error = loc_error.merge(near_loc,left_on='geohash',right_on='loc',how='left')
        loc_error['geohash'] = loc_error['near_loc']
        loc_error = loc_error.groupby('geohash',as_index=False).sum()
        loc_size = prop.groupby('geohash', as_index=False)['geohash'].agg({'loc_size': 'size'})
        loc_size = loc_size.merge(near_loc, left_on='geohash', right_on='loc', how='left')
        loc_size['geohash'] = loc_size['near_loc']
        loc_size = loc_size.groupby('geohash', as_index=False).sum()
        parcel_error_df = data_temp[data_temp['transactiondate'].isnull()][['parcelid', 'transactiondate', 'logerror']]
        parcel_error_df.rename(columns={'logerror': 'parcel-logerror'}, inplace=True)
        result = data_temp.merge(loc_error, on='geohash', how='left')
        result = result.merge(loc_size, on='geohash', how='left')
        result = result.merge(parcel_error_df, on=['parcelid', 'transactiondate'], how='left')
        result['sum_loc_error'] = result['sum_loc_error'] - result['parcel-logerror'].fillna(0)
        result['loc_count'] = result['loc_count'] - (~result['parcel-logerror'].isnull())
        result['loc_mean_error'] = result['sum_loc_error']/result['loc_count']
        result['saled_rate'] = result['loc_count']/(result['loc_size']-1)
        result = result[['loc_mean_error3','saled_rate3']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 附近地点logerror
def get_near_loc_error(data,precision=7):
    key = data.memory_usage().sum()
    result_path = cache_path + 'near_loc_error{0}_{1}.hdf'.format(precision,key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        prop = pd.read_csv(prop_path)
        parcel_geohash_dict = get_parcel_geohash_dict(precision)
        data_temp = data[['parcelid','transactiondate','logerror']].copy()
        data_temp['geohash'] = data_temp['parcelid'].map(parcel_geohash_dict)
        prop['geohash'] = prop['parcelid'].map(parcel_geohash_dict)
        loc_error = data_temp.groupby('geohash',as_index=False)['logerror'].agg({'sum_loc_error': 'sum',
                                                                                     'loc_count': 'count'})
        loc_size = prop.groupby('geohash', as_index=False)['geohash'].agg({'loc_size': 'size'})
        parcel_error_df = data_temp[data_temp['transactiondate'].isnull()][['parcelid','transactiondate','logerror']]
        parcel_error_df.rename(columns={'logerror':'parcel-logerror'},inplace=True)
        result = data_temp.merge(loc_error,on='geohash',how='left')
        result = result.merge(loc_size, on='geohash', how='left')
        result = result.merge(parcel_error_df, on=['parcelid','transactiondate'], how='left')
        result['sum_loc_error'] = result['sum_loc_error'] - result['parcel-logerror'].fillna(0)
        result['loc_count'] = result['loc_count'] - (~result['parcel-logerror'].isnull())
        result['loc_mean_error'] = result['sum_loc_error']/(result['loc_count']+0.0000001)
        result['saled_rate'] = result['loc_count']/(result['loc_size']-1+0.0000001)
        result = result[['loc_mean_error','saled_rate']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



#
# # 构造特征
# def make_feat_set(data):
#     key = data.memory_usage().sum()
#     result_path = cache_path + 'feat_set{}.hdf'.format(key)
#     if os.path.exists(result_path):
#         result = pd.read_hdf(result_path, 'w')
#     else:
#         result = None
#         train_df[['loc_mean_error', 'saled_rate']] = get_near_loc_error(train_df, 7)
#         train_df[['loc_mean_error2', 'saled_rate2']] = get_near_loc_error(train_df, 6)
#
#     return result

