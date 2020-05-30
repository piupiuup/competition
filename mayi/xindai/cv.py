from xindai.feat5 import *
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

test_feat = make_feats('2016-11-01',30).fillna(-1)
predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum']+delect_id)]

def evalerror(pred, df):
    label = df.get_label()
    rmse = mean_squared_error(label, pred) ** 0.5
    return ('RMSE', rmse, False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.loan_sum)

gbm = lgb.cv(params, lgb_test,
             num_boost_round=10000,
             verbose_eval=100,
             feval=evalerror,
             early_stopping_rounds=100)








