from xindai.feat2 import *
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

train_feat = pd.DataFrame()
start_date = '2016-10-02'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-1)),30).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)
train_feat['label'] = train_feat['loan_sum'].apply(lambda x:1 if x>0 else 0)
test_feat['label'] = test_feat['loan_sum'].apply(lambda x:1 if x>0 else 0)

train_feat['date_rate'] = train_feat['date_rate']/train_feat['date_rate'].mean()*1.3
test_feat['date_rate'] = test_feat['date_rate']/test_feat['date_rate'].mean()
predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum','label']+delect_id)]

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.loan_sum)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.loan_sum,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 100,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
preds = gbm.predict(test_feat[predictors])










