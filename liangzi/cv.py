import xgboost as xgb
from liangzi.feat5 import *

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')

print('构造特征...')
data_feat = make_set(data_path)
train_feat = train[['id']].merge(data_feat,on='id',how='left')

def evalerror(pred, df):
    auc = roc_auc_score(df.label,pred)
    return ('auc', auc, False)

predictors = [f for f in train_feat.columns if f not in ['id','label','enddate']]

print('开始训练...')
lgb_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 200,
    'learning_rate': 0.005,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 100,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_cv = lgb.cv(lgb_params, lgb_train, num_boost_round=2000, nfold=5, metrics='auc', verbose_eval=200,early_stopping_rounds=50)


test_feat = train_feat[train_feat['enum']!=-1].sample(frac=0.2, random_state=70, axis=0)
train_feat = train_feat[~train_feat['id'].isin(test_feat['id'].values)]

lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label,reference=lgb_train)

# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     # 'max_depth': 20,
#     'num_leaves': 200,
#     'learning_rate': 0.005,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.95,
#     'bagging_freq': 5,
#     'verbose': -1,
#     'seed': 100,
# }
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10000,
#                 valid_sets=lgb_test,
#                 verbose_eval = 50,
#                 early_stopping_rounds=66)



















