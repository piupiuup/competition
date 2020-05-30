from renji.feat2 import *


print('读取train数据...')
train = pd.read_hdf(train_hdf_path,'w')
print('切分数据...')
select_index = list(train.index[::4])
test = train.iloc[select_index, :]
train = train.iloc[~train.index.isin(select_index), :]

print('构造特征...')
train_feat = make_set(train)
test_feat = make_set(test)
predictors = [c for c in train_feat.columns if c not in ['id','label']]


print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
print('线下auc得分为： {}'.format(roc_auc_score(test_feat.label,preds)))
preds_scatter = get_threshold(preds,silent=0)
print('线下F1得分为： {}'.format(f1_score(test_feat.label,preds_scatter)))














