import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from tool.tool import *
from zillow.make_feat import *
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

data_path = 'C:/Users/csw/Desktop/python/zillow/data/'
prop_2016_path = data_path + 'properties_2016.csv'
prop_2017_path = data_path + 'properties_2017.csv'
sample_path = data_path + 'sample_submission.csv'
train_2016_path = data_path + 'train_2016_v2.csv'
train_2017_path = data_path + 'train_2017.csv'

print('读取数据...')

prop_2016 = pd.read_csv(prop_2016_path)
prop_2017 = pd.read_csv(prop_2016_path)
train_2016 = pd.read_csv(train_2016_path,parse_dates=["transactiondate"])
train_2017 = pd.read_csv(train_2017_path,parse_dates=["transactiondate"])
submission = pd.read_csv(sample_path)

prop_2016['year'] = 2016
prop_2017['year'] = 2017
train_2016['year'] = train_2016['transactiondate'].dt.year
train_2017['year'] = train_2017['transactiondate'].dt.year
prop = pd.concat([prop_2016,prop_2017])
train = pd.concat([train_2016,train_2017])

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)
for c in prop.columns:
    prop[c] = prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))


#Indicator whether it has AC or not
prop['N-ACInd'] = (prop['airconditioningtypeid']!=5)*1
#Indicator whether it has Heating or not
prop['N-HeatInd'] = (prop['heatingorsystemtypeid']!=13)*1
prop['assessmentyear'] = prop['year'] - prop['assessmentyear']
prop['yearbuilt'] = prop['year'] - prop['yearbuilt']



# 附近地点logerror
data_feat = prop.merge(train[['parcelid','year','logerror','transactiondate']],on=['parcelid','year'],how='left')
data_feat[['loc_mean_error','saled_rate']] = get_near_loc_error(data_feat,7)
data_feat[['loc_mean_error2','saled_rate2']] = get_near_loc_error(data_feat,6)
# train_df[['loc_mean_error3','saled_rate3']] = get_ex_near_loc_error(train_df)
data_feat['month'] = data_feat['transactiondate'].dt.month
data_feat.fillna(-1,inplace=True)

train_df = train[['parcelid','year']].merge(data_feat,on=['parcelid','year'],how='left')
# train_df = data_feat[data_feat['parcelid'].isin(train['parcelid']).values].copy()
train_df.index = list(range(len(train_df)))
predictors = train_df.columns.drop(['logerror','transactiondate',
       'poolsizesum', 'pooltypeid10', 'fips', 'typeconstructiontypeid',
       'assessmentyear', 'architecturalstyletypeid', 'basementsqft', 'N-ACInd',
       'buildingclasstypeid', 'decktypeid', 'fireplaceflag', 'storytypeid',
       'yardbuildingsqft26', 'finishedsquarefeet13', 'finishedsquarefeet6',
       'N-HeatInd'])
train_X = train_df[predictors]
train_y = train_df['logerror']
weights = 1/(abs(train_y)*16+1)

sample2 = make_sample(list(range(len(train_X))),5,seed=66)[0]
sample1 = list(set(range(len(train_X))) - set(sample2))

weights = 1/(abs(train_y)*16+1)
import lightgbm as lgb
lgb_train = lgb.Dataset(train_X.iloc[sample1,], train_y[sample1], weight=weights[sample1])
lgb_test = lgb.Dataset(train_X.iloc[sample2,], train_y[sample2])


params = {
'learning_rate': 0.008,
'boosting_type': 'gbdt',
'objective': 'regression',
'metric': 'mae',
'sub_feature': 0.5,
'num_leaves': 60,
'min_data':500,
'min_hessian': 1,
'verbose': 0,
}

watchlist = [lgb_test]
gbm = lgb.train(params, lgb_train, 10000, watchlist,early_stopping_rounds=100, verbose_eval=1000)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')



