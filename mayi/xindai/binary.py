from xindai.feat1 import *
import datetime
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2016-10-02'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7))).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30)).fillna(-1)
train_feat['date_rate'] = train_feat['date_rate']/train_feat['date_rate'].mean()*1.3
test_feat['date_rate'] = test_feat['date_rate']/test_feat['date_rate'].mean()

predictors = train_feat.columns.drop(['uid','loan_sum'])
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    # 'max_depth': 8,
    'num_leaves': 60,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 66,
}

lgb_train = lgb.Dataset(train_feat[predictors], train_feat['loan_sum'])
# lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
gbm = lgb.train(params, lgb_train, 6000)
test_preds = gbm.predict(test_feat[predictors])


test_binary_pred = pd.DataFrame({'uid':test_feat.uid.values,'date':'2016-12-01','binary_pred':test_preds})
test_binary_pred.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\data\data_binary_pred.csv',index=False)












