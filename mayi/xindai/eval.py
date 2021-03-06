from xindai.feat5 import *
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

train_feat = pd.DataFrame()
start_date = '2016-10-02'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-1)),30).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)

predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum']+delect_id)]
label_mean = test_feat.loan_sum.mean()

def evalerror(pred, df):
    label = df.get_label().values.copy()
    pred_temp = np.array(pred.copy())
    pred_mean = np.mean(pred_temp)
    pred_temp = pred_temp/pred_mean*label_mean
    rmse = mean_squared_error(label,pred_temp)**0.5
    return ('rmse',rmse,False)

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
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.loan_sum)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.loan_sum,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_test,
                verbose_eval = 100,
                feval = evalerror,
                early_stopping_rounds=100)
pred = gbm.predict(test_feat[predictors])
print('普通lgb得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],pred)**0.5))
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')







