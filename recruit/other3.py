import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor

data_path = 'C:/Users/csw/Desktop/python/recruit/data/'

data = {
    'air_visit': pd.read_csv(data_path + 'air_visit_data.csv'),
    'air_store': pd.read_csv(data_path + 'air_store_info.csv'),
    'hpg_store': pd.read_csv(data_path + 'hpg_store_info.csv'),
    'air_reserve': pd.read_csv(data_path + 'air_reserve.csv'),
    'hpg_reserve': pd.read_csv(data_path + 'hpg_reserve.csv'),
    'store_id': pd.read_csv(data_path + 'store_id_relation.csv'),
    'submission': pd.read_csv(data_path + 'sample_submission.csv'),
    'date_info': pd.read_csv(data_path + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}
air_reserve = pd.read_csv(data_path + 'air_reserve.csv')
hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv')
air_store = pd.read_csv(data_path + 'air_store_info.csv')
hpg_store = pd.read_csv(data_path + 'hpg_store_info.csv')
air_visit = pd.read_csv(data_path + 'air_visit_data.csv')
store_id = pd.read_csv(data_path + 'store_id_relation.csv').set_index('hpg_store_id',drop=False)
date_info = pd.read_csv(data_path + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'})
submission = pd.read_csv(data_path + 'sample_submission.csv')

air_reserve['visit_date'] = air_reserve['visit_datetime'].str[:10]
air_reserve['reserve_date'] = air_reserve['reserve_datetime'].str[:10]
hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].str[:10]
hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].str[:10]


hpg_reserve = pd.merge(hpg_reserve, store_id, how='inner', on=['hpg_store_id'])

hpg_reserve['reserve_datetime_diff'] = (pd.to_datetime(hpg_reserve['visit_datetime'])-pd.to_datetime(hpg_reserve['reserve_datetime'])).dt.days
air_reserve['reserve_datetime_diff'] = (pd.to_datetime(air_reserve['visit_datetime'])-pd.to_datetime(air_reserve['reserve_datetime'])).dt.days
temp1 = air_reserve.groupby(['air_store_id', 'visit_date'],).agg({'reserve_datetime_diff':{'rs1':'sum',
                                                                                           'rs2':'mean'},
                                                                  'reserve_visitors':{'rv1':'sum',
                                                                                      'rv2':'mean'}})
temp1.columns = temp1.columns.drop_levels(0)
temp1.reset_index(inplace=True)
air_reserve = air_reserve.merge(temp1,on=['air_store_id', 'visit_date'],how='inner')
temp1 = hpg_reserve.groupby(['air_store_id', 'visit_date'],).agg({'reserve_datetime_diff':{'rs1':'sum',
                                                                                           'rs2':'mean'},
                                                                  'reserve_visitors':{'rv1':'sum',
                                                                                      'rv2':'mean'}})
temp1.columns = temp1.columns.drop_levels(0)
temp1.reset_index(inplace=True)
hpg_reserve = hpg_reserve.merge(temp1,on=['air_store_id', 'visit_date'],how='inner')

# for df in ['air_reserve', 'hpg_reserve']:
#     data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime']).dt.date
#     data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime']).dt.date
#     data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
#     tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
#         ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
#         columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
#     tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
#         ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
#         columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
#     data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])

air_visit['visit_date'] = pd.to_datetime(data['air_visit']['visit_date'])
air_visit['dow'] = air_visit['visit_date'].dt.dayofweek
air_visit['year'] = air_visit['visit_date'].dt.year
air_visit['month'] = air_visit['visit_date'].dt.month
air_visit['visit_date'] = air_visit['visit_date'].dt.date


# data['air_visit']['visit_date'] = pd.to_datetime(data['air_visit']['visit_date'])
# data['air_visit']['dow'] = data['air_visit']['visit_date'].dt.dayofweek
# data['air_visit']['year'] = data['air_visit']['visit_date'].dt.year
# data['air_visit']['month'] = data['air_visit']['visit_date'].dt.month
# data['air_visit']['visit_date'] = data['air_visit']['visit_date'].dt.date

submission['visit_date'] = pd.to_datetime(submission['visit_date'])
submission['dow'] = submission['visit_date'].dt.dayofweek
submission['year'] = submission['visit_date'].dt.year
submission['month'] = submission['visit_date'].dt.month
submission['visit_date'] = submission['visit_date'].dt.date

# data['submission']['visit_date'] = data['submission']['id'].str[-10:]
# data['submission']['air_store_id'] = data['submission']['id'].str[:-11]
# data['submission']['visit_date'] = pd.to_datetime(data['submission']['visit_date'])
# data['submission']['dow'] = data['submission']['visit_date'].dt.dayofweek
# data['submission']['year'] = data['submission']['visit_date'].dt.year
# data['submission']['month'] = data['submission']['visit_date'].dt.month
# data['submission']['visit_date'] = data['submission']['visit_date'].dt.date


unique_stores = submission['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': i}) for i in range(7)],
                   axis=0, ignore_index=True).reset_index(drop=True)
# unique_stores = data['submission']['air_store_id'].unique()
# stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': i}) for i in range(7)],
#                    axis=0, ignore_index=True).reset_index(drop=True)

# sure it can be compressed...
tmp = air_visit.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].agg({'min_visitors':'min',
                                                                                  'mean_visitors':'mean',
                                                                                  'median_visitors':'median',
                                                                                  'max_visitors':'max',
                                                                                  'count_observations':'count',
                                                                                  'std_observations':'std',
                                                                                  'skew_observations':'skew'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
# tmp = data['air_visit'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].agg({'min_visitors':'min',
#                                                                                           'mean_visitors':'mean',
#                                                                                           'median_visitors':'median',
#                                                                                           'max_visitors':'max',
#                                                                                           'count_observations':'count',
#                                                                                           'std_observations':'std',
#                                                                                           'skew_observations':'skew'})
# stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

stores = pd.merge(stores, air_store, how='left', on=['air_store_id'])
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
lbl = preprocessing.LabelEncoder()
for i in range(10):
    stores['air_genre_name' + str(i)] = lbl.fit_transform(
        stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
    stores['air_area_name' + str(i)] = lbl.fit_transform(
        stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

date_info['day_of_week'] = lbl.fit_transform(pd.to_datetime(date_info['visit_date']))
train = pd.merge(air_visit, date_info, how='left', on=['visit_date'])
test = pd.merge(submission, date_info, how='left', on=['visit_date'])

train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

train = pd.merge(train, air_reserve, how='left', on=['air_store_id', 'visit_date'])
test = pd.merge(test, air_reserve, how='left', on=['air_store_id', 'visit_date'])
train = pd.merge(train, hpg_reserve, how='left', on=['air_store_id', 'visit_date'])
test = pd.merge(test, hpg_reserve, how='left', on=['air_store_id', 'visit_date'])

train['holiday'] = train['air_store_id'] + '_' + train['visit_date']

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

# NEW FEATURES FROM JMBULL
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude']
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [c for c in train if c not in ['holiday', 'air_store_id', 'visit_date', 'visitors']]
train = train.fillna(-1)
test = test.fillna(-1)


def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5


model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, n_estimators=200, subsample=0.8, max_depth=10)
model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
model3 = XGBRegressor(learning_rate=0.2, n_estimators=200, subsample=0.8, colsample_bytree=0.8, max_depth=10)

model1.fit(train[col], np.log1p(train['visitors'].values))
model2.fit(train[col], np.log1p(train['visitors'].values))
model3.fit(train[col], np.log1p(train['visitors'].values))

preds1 = model1.predict(train[col])
preds2 = model2.predict(train[col])
preds3 = model3.predict(train[col])

print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds1))
print('RMSE KNeighborsRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds2))
print('RMSE XGBRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds3))
preds1 = model1.predict(test[col])
preds2 = model2.predict(test[col])
preds3 = model3.predict(test[col])

test['visitors'] = 0.3 * preds1 + 0.3 * preds2 + 0.4 * preds3
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id', 'visitors']].copy()
del train;
del data;

# from hklee
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st/code
dfs = {re.search('\\\\([^/\.]*)\.csv', fn).group(1):
           pd.read_csv(fn) for fn in glob.glob(data_path+'*.csv')}

for k, v in dfs.items(): locals()[k] = v

wkend_holidays = date_info.apply(
    (lambda x: (x.day_of_week == 'Sunday' or x.day_of_week == 'Saturday') and x.holiday_flg == 1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

wmean = lambda x: ((x.weight * x.visitors).sum() / x.weight.sum())
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0: 'visitors'}, inplace=True)  # cumbersome, should be better ways.

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=[
    'air_store_id', 'day_of_week', 'holiday_flg'], how='left')

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg == 0], on=('air_store_id', 'day_of_week'),
    how='left')['visitors_y'].values

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(),
    on='air_store_id', how='left')['visitors_y'].values

sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sub2 = sample_submission[['holiday', 'visitors']].copy()
sub_merge = pd.merge(sub1, sub2, on='holiday', how='inner')

sub_merge['visitors'] = 0.7 * sub_merge['visitors_x'] + 0.3 * sub_merge['visitors_y'] * 1.1
sub_merge[['holiday', 'visitors']].to_csv('submission.csv', index=False)