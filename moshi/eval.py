from moshi.feat1 import *
import lightgbm as lgb

data_path = 'C:/Users/csw/Desktop/python/JD/moshi/data/eval/'
train_path = data_path + 't_login.csv'
test_path = data_path + 't_login_test.csv'
train_label_path = data_path + 't_trade.csv'
test_label_path = data_path + 't_trade_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_label = pd.read_csv(train_label_path)
test_label = pd.read_csv(test_label_path)

train.drop(['timestamp','is_scan','is_sec'],axis=1,inplace=True)
test.drop(['timestamp','is_scan','is_sec'],axis=1,inplace=True)
train.rename(columns={'time':'log_time'},inplace=True)
test.rename(columns={'time':'log_time'},inplace=True)
train_label.rename(columns={'is_risk':'label'},inplace=True)
test_label.rename(columns={'is_risk':'label'},inplace=True)
data = pd.concat([train,test])


train_feat = make_feats(data,train_label,train_label)
test_feat = make_feats(data,test_label,train_label)

predictors = train_feat.columns.drop(['rowkey','label', 'device','log_id', 'log_time','time'])

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 4,
    'num_leaves': 32,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label)

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
preds_scatter = get_threshold(preds)
print('f1得分为：{}'.format(f1(test_feat['label'].astype(int).values,preds_scatter)))









