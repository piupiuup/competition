from corporacion.feat3 import *

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

train_feat = pd.DataFrame()
end_date = '2017-07-12'
n = 0
for i in range(1):
    train_feat_sub = make_feats(date_add_days(end_date, i*(-7))).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(end_date, 14)).fillna(-1)
# test_feat['onpromotion'] = test_feat['onpromotion'].replace(-1,0)

predictors = [f for f in test_feat.columns if f not in (list(range(16)))]

def evalerror(pred, df):
    label = df.get_label().values.copy()
    # a = df.perishable
    rmse = mean_squared_error(label,pred)**0.5
    return ('rmse',rmse,False)

print('开始训练...')
params = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 30,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('第1天...')
lgb_train = lgb.Dataset(train_feat[predictors], train_feat[0], weight=train_feat["perishable"])
lgb_test = lgb.Dataset(test_feat[predictors], test_feat[0], weight=test_feat["perishable"])

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_test,
                verbose_eval = 100,
                feval = evalerror,
                early_stopping_rounds=100)
pred = gbm.predict(test_feat[predictors])
print('第{}天得分：    {}'.format(i,(sum((pred-test_feat[0].values)**2*test_feat['perishable'].values)/
                                sum(test_feat['perishable']))**0.5))
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')


