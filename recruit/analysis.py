import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = 'C:/Users/csw/Desktop/python/recruit/data/'

air_reserve = pd.read_csv(data_path + 'air_reserve.csv')
hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv')
air_store = pd.read_csv(data_path + 'air_store_info.csv')
hpg_store = pd.read_csv(data_path + 'hpg_store_info.csv')
air_visit = pd.read_csv(data_path + 'air_visit_data.csv')
store_id = pd.read_csv(data_path + 'store_id_relation.csv')
date_info = pd.read_csv(data_path + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'})
submission = pd.read_csv(data_path + 'sample_submission.csv')

submission['visit_date'] = submission['id'].str[-10:]
submission['air_store_id'] = submission['id'].str[:-11]

print('submission的时间范围：{}  -----  {}'.format(submission['visit_date'].min(),submission['visit_date'].max()))
print('air_visit的时间范围：{}  -----  {}'.format(air_visit['visit_date'].min(),air_visit['visit_date'].max()))
print('air_reserve的时间范围：{}  -----  {}'.format(air_reserve['visit_datetime'].min(),air_reserve['visit_datetime'].max()))
print('hpg_reserve的时间范围：{}  -----  {}'.format(hpg_reserve['visit_datetime'].min(),hpg_reserve['visit_datetime'].max()))

print('submission的餐厅数量：{}'.format(submission['air_store_id'].nunique()))
print('submission的记录数量：{}'.format(submission.shape[0]))
print('air_visit的餐厅数量：{}'.format(air_visit['air_store_id'].nunique()))
print('air_visit的记录数量：{}'.format(air_visit.shape[0]))
print('air_reserve的餐厅数量：{}'.format(air_reserve['air_store_id'].nunique()))
print('air_reserve的记录数量：{}'.format(air_reserve.shape[0]))
print('hpg_reserve的餐厅数量：{}'.format(hpg_reserve['hpg_store_id'].nunique()))
print('hpg_reserve的记录数量：{}'.format(hpg_reserve.shape[0]))
print('air和hpg重叠的商店数量：{}'.format(store_id.shape[0]))

print('日本市数量：\n {}'.format(air_store['air_area_name'].apply(lambda x: x.split(' ')[0]).value_counts()))


hpg_reserve = pd.merge(hpg_reserve, holiday, how='inner', on=['hpg_store_id'])





















