from mayi.feat2 import *
import lightgbm as lgb

# cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
# test = test.sample(frac=0.1,random_state=66, axis=0)

data_feat = make_feats(train,test).fillna(0)

train_feat = data_feat[data_feat['time_stamp']<'2017-08-29'].copy()
test_feat = data_feat[data_feat['time_stamp']>='2017-08-29'].copy()
train_feat = grp_normalize(train_feat,'row_id',['knn','knn2'],start=0)
test_feat = grp_normalize(test_feat,'row_id',['knn','knn2'],start=0)
train_feat = grp_rank(train_feat,'row_id',['cos'],ascending=False)
test_feat = grp_rank(test_feat,'row_id',['cos'],ascending=False)
train_feat = grp_standard(train_feat,'row_id',['shop_count','loc_knn2'])
test_feat = grp_standard(test_feat,'row_id',['shop_count','loc_knn2'])

predictors = data_feat.columns.drop(['row_id','time_stamp', 'user_id','row_tfidf',
                                     'shop_id', 'wifi_infos', 'label'])

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
    'verbose': -1,
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
preds = gbm.predict(test_feat[predictors])
test_feat['pred'] = preds
test_feat.sort_values('pred',inplace=True)
result = test_feat.drop_duplicates('row_id',keep='last')
test_test = test[test['time_stamp']>='2017-08-29']
test_test = test_test.merge(result,on='row_id',how='left')
print('准确率为：{}'.format(acc(test_test)))









