from recruit.feat2 import *
import datetime
import catboost as cb
import xgboost as xgb
from tqdm import tqdm
import lightgbm as lgb
from catboost import Pool
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2017-01-29'
period = [14,28,56,1000]
for i in range(20):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),39,period)
    train_feat = pd.concat([train_feat,train_feat_sub])
for i in range(1,6):
    train_feat_sub = make_feats(date_add_days(start_date,i*(7)),42-(i*7),period)
    train_feat = pd.concat([train_feat,train_feat_sub])

eval_feat = make_feats(date_add_days(start_date, 42),39,period)
test_feat = make_feats(date_add_days(start_date, 84),39,period)

predictors = [f for f in test_feat.columns if f not in (['id','store_id','visit_date','end_date','air_area_name','visitors','month'])]

params = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

t0 = time.time()
date_dict = {date:diff_of_days(train_feat['end_date'].max(),date) for date in train_feat['end_date'].unique()}
weight = 1000-train_feat['end_date'].map(date_dict)
lgb_train = lgb.Dataset(train_feat[predictors], train_feat['visitors'],weight=weight)
lgb_eval = lgb.Dataset(eval_feat[predictors], eval_feat['visitors'])
lgb_test = lgb.Dataset(test_feat[predictors], test_feat['visitors'])

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                verbose_eval = 100,
                early_stopping_rounds = 100)

lgb_eval_pred = gbm.predict(eval_feat[predictors])
print('lgb线下的得分：{}'.format(mean_squared_error(eval_feat['visitors'],lgb_eval_pred)**0.5))
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
print('lgb训练用时{}秒'.format(time.time() - t0))


print('开始xgb训练...')
t0 = time.time()
params = {'booster': 'gbtree',
          'eval_metric': 'rmse',
          'gamma': 1,
          'min_child_weight': 1.5,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.03,
          'tree_method': 'exact',
          'seed': 2017,
          # 'nthread': 12
          }
xgb_train = xgb.DMatrix(train_feat[predictors], train_feat['visitors'])
xgb_eval = xgb.DMatrix(eval_feat[predictors], eval_feat['visitors'])
watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]
xgb_model = xgb.train(params,
                  xgb_train,
                  2000,
                  evals=watchlist,
                  verbose_eval=20,
                  early_stopping_rounds=20)
xgb_eval_pred = xgb_model.predict(xgb_eval)
print('xgb线下的得分：{}'.format(mean_squared_error(eval_feat['visitors'],xgb_eval_pred)**0.5))
print('xgb训练用时{}秒'.format(time.time() - t0))



print('开始cb训练...')
t0 = time.time()
train_pool = Pool(train_feat[predictors], train_feat['visitors'])
eval_pool = Pool(eval_feat[predictors], eval_feat['visitors'])
cb_model = cb.CatBoostRegressor(iterations=2000, depth=7, learning_rate=0.06, eval_metric='RMSE',
                             od_type='Iter', od_wait=20, random_seed=42,
                             bagging_temperature=0.85, rsm=0.85, verbose=False)
cb_model.fit(train_pool, use_best_model=True, eval_set=eval_pool, verbose=True)
cb_eval_pred = cb_model.predict(eval_pool)
print('xgb线下的得分：{}'.format(mean_squared_error(eval_feat['visitors'],cb_eval_pred)**0.5))
print('xgb训练用时{}秒'.format(time.time() - t0))



model_preds = pd.DataFrame({'lgb_pred':lgb_eval_pred,
                            'xgb_pred':xgb_eval_pred,
                            'cb_pred':cb_eval_pred})
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(model_preds, eval_feat.visitors)
final_pred = lr.predict(model_preds)
print('融合后修正前得分为：{}'.format(mean_squared_error(eval_feat['visitors'],final_pred)**0.5))



