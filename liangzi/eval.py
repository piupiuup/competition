import xgboost as xgb
from liangzi.feat4 import *

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/eval/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat = make_set(train,train,data_path)
test_feat = make_set(train,test,data_path)
train_feat['id'] = train_feat['id'].apply(lambda x: 0-int(x[1:]) if 'p' in x else int(x[1:]))
test_feat['id'] = test_feat['id'].apply(lambda x: 0-int(x[1:]) if 'p' in x else int(x[1:]))
# data_feat = pd.concat([train_feat,test_feat])

# train_feat = train_feat[['id']].merge(data_feat,on='id',how='left')
# test_feat = test_feat[['id']].merge(data_feat,on='id',how='left')
# train_feat = grp_normalize(train_feat,'hy',['zczb2','knn2'],start=0)
# test_feat = grp_normalize(test_feat,'hy',['knn','knn2'],start=0)
# train_feat = grp_standard(train_feat,'hy',['zczb2'])
# test_feat = grp_standard(test_feat,'hy',['zczb2'])

predictors = train_feat.columns.drop(['label','enddate','hy_0.0', 'hy_12.0', 'hy_16.0', 'hy_57.0',
                                      'hy_76.0', 'hy_90.0','hy_91.0', 'hy_93.0', 'hy_94.0', 'hy_95.0',
                                      'hy_96.0'])

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    # 'max_depth': 20,
    'num_leaves': 150,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 100,
}

lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 50,
                early_stopping_rounds=66)


# xgb_train = xgb.DMatrix(train_feat[predictors], train_feat.label)
# xgb_eval = xgb.DMatrix(test_feat[predictors], test_feat.label)
#
# xgb_params = {'objective': 'binary:logistic',
#               'eval_metric': ['auc'],
#               'eta': 0.05,
#               'max_depth': 6,
#               'subsample': 0.7,
#               'colsample_bytree': 0.7,
#               'lambda': 500,
#               'alpha': 5,
#               'silent': 1,
#               'verbose_eval': True}
# watch_list = [(xgb_train, 'train'), (xgb_eval, 'test')]
# gbm = xgb.train(xgb_params,
#                 xgb_train,
#                 num_boost_round=5000,
#                 verbose_eval=True,
#                 evals=watch_list,
#                 early_stopping_rounds=50)

feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
print(feat_imp)

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
print('线下auc得分为： {}'.format(roc_auc_score(test_feat.label,preds)))
preds_scatter = get_threshold(preds)
print('线下F1得分为： {}'.format(f1_score(test_feat.label,preds_scatter)))




















