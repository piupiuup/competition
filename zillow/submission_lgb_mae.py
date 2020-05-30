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
# data_feat[['loc_mean_error','saled_rate']] = get_near_loc_error(data_feat,7)
# data_feat[['loc_mean_error2','saled_rate2']] = get_near_loc_error(data_feat,6)
# train_df[['loc_mean_error3','saled_rate3']] = get_ex_near_loc_error(train_df)
data_feat['month'] = data_feat['transactiondate'].dt.month
data_feat.fillna(-1,inplace=True)


train_df = train[['parcelid','year']].merge(data_feat,on=['parcelid','year'],how='left')
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


print('开始CV 5折训练...')
t0 = time.time()
mean_score = []
preds = np.zeros(len(train_df))
kf = KFold(len(train_df), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    model_path = cache_path + 'lgb_model_{}.model'.format(i)
    if not (os.path.exists(model_path) & 1):
        lgb_train = lgb.Dataset(train_X.iloc[train_index], train_y.iloc[train_index], weight=weights.iloc[train_index])
        # lgb_test = lgb.Dataset(train_X.iloc[test_index], train_y.iloc[test_index])

        params = {
            'learning_rate': 0.008,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'sub_feature': 0.5,
            'num_leaves': 60,
            'min_data': 500,
            'min_hessian': 1,
            'verbose': 0,
        }
        gbm = lgb.train(params, lgb_train, 4500)
        pickle.dump(gbm, open(model_path,'wb+'))
    gbm = pickle.load(open(model_path,'rb+'))
    preds_sub = gbm.predict(train_X.iloc[test_index])
    preds[test_index] += preds_sub
    score = mean_absolute_error(train_y.iloc[test_index].values, preds_sub)
    mean_score.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
train_pred = pd.DataFrame({'parcelid':train_df['parcelid'].values,'pred':preds})
train_df.to_csv(cache_path + 'train_pred_lgb_mae.csv', index=False)
print('mae平均得分: {}'.format(np.mean(mean_score)))
print('CV训练用时{}秒'.format(time.time() - t0))



def CV_predict(data_feat,predictors):
    preds = np.zeros(len(data_feat))
    for i in tqdm(range(5)):
        gbm = pickle.load(open(model_path, 'rb+'))
        preds_sub = gbm.predict(data_feat[predictors])
        preds += preds_sub
    preds = preds / 5.0
    return preds


submission = pd.read_csv(sample_path)
submission['year'] = 2016
submission.rename(columns={'ParcelId':'parcelid'},inplace=True)
data_feat.drop_duplicates(['parcelid','year'],inplace=True)
test_df = submission[['parcelid','year']].merge(data_feat,on=['parcelid','year'],how='left')
print('开始预测2016年...')
for i,month in enumerate([10,11,12]):
    print('预测{}月份的...'.format(month))
    test_df['month'] = month
    submission.iloc[:,(i+1)] = CV_predict(test_df,predictors)
submission['year'] = 2017
test_df = submission[['parcelid','year']].merge(data_feat,on=['parcelid','year'],how='left')
print('开始预测2017年...')
for i,month in enumerate([10,11,12]):
    print('预测{}月份的...'.format(month))
    test_df['month'] = month
    submission.iloc[:,(i+4)] = CV_predict(test_df,predictors)
submission.rename(columns={'parcelid':'ParcelId'},inplace=True)
submission.to_csv('C:/Users/csw/Desktop/python/zillow/submission/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')





