from mayi.feat2 import *
import datetime
import lightgbm as lgb

cache_path = 'F:/mayi_cache2/'
shop_path = data_path + 'ccf_first_round_shop_info.csv'

data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'
shop = pd.read_csv(shop_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
train_feat = make_feats(train,test).fillna(0)

data_path = 'C:/Users/csw/Desktop/python/mayi/data/'
test_path = data_path + 'evaluation_public.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'
test = pd.read_csv(test_path)
# test = test[test['time_stamp']<'2017-09-08']
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
test_feat = make_feats(train,test).fillna(0)

train_feat = grp_normalize(train_feat,'row_id',['knn','knn2'],start=0)
test_feat = grp_normalize(test_feat,'row_id',['knn','knn2'],start=0)
train_feat = grp_rank(train_feat,'row_id',['cos'],ascending=False)
test_feat = grp_rank(test_feat,'row_id',['cos'],ascending=False)
train_feat = grp_standard(train_feat,'row_id',['shop_count','loc_knn2'])
test_feat = grp_standard(test_feat,'row_id',['shop_count','loc_knn2'])

predictors = train_feat.columns.drop(['row_id','time_stamp', 'user_id','shop_id',
                                     'mall_id', 'wifi_infos', 'label'])

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
gbm = lgb.train(params,lgb_train, num_boost_round=700)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
test_feat['pred'] = preds
test_feat.sort_values('pred',inplace=True)
submission = test_feat.drop_duplicates('row_id',keep='last')
test = pd.read_csv(test_path)
submission = test[['row_id']].merge(submission[['row_id','shop_id']],on='row_id',how='left')
submission.to_csv(r'C:\Users\csw\Desktop\python\mayi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)



















