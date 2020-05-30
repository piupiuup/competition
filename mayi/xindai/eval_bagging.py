from xindai.feat5 import *
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

train_feat = pd.DataFrame()
start_date = '2016-10-02'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-1)),30).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)
# train_feat = make_feats('2016-08-15').fillna(-1)
# test_feat = make_feats('2016-11-03').fillna(-1)

# train_feat = test_feat[:60000]
# test_feat = test_feat[60000:]

# train_feat['date_rate'] = train_feat['date_rate']/train_feat['date_rate'].mean()*1.3
# test_feat['date_rate'] = test_feat['date_rate']/test_feat['date_rate'].mean()
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
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 100,
                feval = evalerror,
                early_stopping_rounds=100)
lgb_pred = gbm.predict(test_feat[predictors])
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
print('lgb修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],lgb_pred)**0.5))
lgb_pred = lgb_pred/np.mean(lgb_pred)*1.1946
print('lgb得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],lgb_pred)**0.5))



##############################使用随机森林预测##################################
model = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=12,
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, max_features=50,
                              max_leaf_nodes=None, bootstrap=True,
                              oob_score=False, n_jobs=-1, random_state=66,
                              verbose=0, warm_start=False)
model = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=10,
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, max_features=50,
                              max_leaf_nodes=None, bootstrap=True,
                              oob_score=False, n_jobs=-1, random_state=66,
                              verbose=0, warm_start=False)
rf_model = model.fit(train_feat[predictors], train_feat['loan_sum'])
rf_pred = rf_model.predict(test_feat[predictors])
print('rf修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],rf_pred)**0.5))
# rf_pred = rf_pred/np.mean(rf_pred)*1.1946
# print('randomforest得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],rf_pred)**0.5))




##############################使用GBDT预测##################################
model = GradientBoostingRegressor(n_estimators=1500,
          max_depth= 8,
          max_leaf_nodes= 20,
          min_samples_leaf= 30,
          learning_rate= 0.01,
          loss= 'ls',
          subsample= 0.6,
          max_features='sqrt',
        random_state=66)
et_model = model.fit(train_feat[predictors], train_feat['loan_sum'])
gbrt_pred = et_model.predict(test_feat[predictors])
print('gbrt修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],gbrt_pred)**0.5))
# gbrt_pred = gbrt_pred/np.mean(gbrt_pred)*1.1946
# print('gbdt得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],gbrt_pred)**0.5))

##############################使用ET预测##################################
model = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, min_samples_split=2,
                             min_samples_leaf=1, max_depth=21, max_features=23,
                            criterion='mse',random_state=66)
et_model = model.fit(train_feat[predictors], train_feat['loan_sum'])
et_pred = et_model.predict(test_feat[predictors])
print('et修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],et_pred)**0.5))
# et_pred = et_pred/np.mean(et_pred)*1.1946
# print('et得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],et_pred)**0.5))


#############################使用xgb预测####################################
xgtrain_x = xgb.DMatrix(train_feat[predictors], train_feat['loan_sum'])
xgtrain_y = xgb.DMatrix(test_feat[predictors], test_feat['loan_sum'])

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 8,
          'lambda':5,
          'subsample': 0.8,
          'colsample_bytree': 0.7,
          'min_child_weight': 10,  # 8~10
          'eta': 0.01,
          'seed':66,
          # 'nthread':12
          }
params['silent'] = 1
watchlist = [(xgtrain_x, 'train'), (xgtrain_y, 'eval')]
model = xgb.train(params, xgtrain_x, 5000, watchlist, early_stopping_rounds=20,verbose_eval = 30)
xgb_pred = model.predict(xgtrain_y)
print('xgb修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],xgb_pred)**0.5))
xgb_pred = xgb_pred/np.mean(xgb_pred)*1.1946
print('xgb修正后得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],xgb_pred)**0.5))


# pred = np.average([lgb_pred,rf_pred,gbrt_pred,et_pred,xgb_pred],axis=0,
#                   weights=[5,0.1,2.2,0.1,0.1])
# print('融合得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],pred)**0.5))





##############################使用catboost预测##################################
train_pool = Pool(train_feat[predictors],train_feat['loan_sum'])
test_pool = Pool(test_feat[predictors],test_feat['loan_sum'])

cb_model = cb.CatBoostRegressor(iterations=2000, depth=7, learning_rate=0.06, eval_metric='RMSE',
                                 od_type='Iter', od_wait=20, random_seed=42, thread_count=7,
                                 bagging_temperature=0.85, rsm=0.85, verbose=False)

print("define model done")
cb_model.fit(train_pool, use_best_model=True, eval_set=test_pool, verbose=True)
cb_pred = cb_model.predict(test_pool)
print('catboost修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],cb_pred)**0.5))
cb_pred = cb_pred/np.mean(cb_pred)*1.1946
print('catboost修正后得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],cb_pred)**0.5))

model_preds = pd.DataFrame({'lgb_pred':lgb_pred,
                            'rf_pred':rf_pred,
                            'et_pred':et_pred,
                            'gbrt_pred':gbrt_pred,
                            'xgb_pred':xgb_pred,
                            'cb_pred':cb_pred})
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(model_preds, test_feat.loan_sum)
import pickle
pickle.dump(lr,open('lr.model','wb+'))
final_pred = lr.predict(model_preds)
print('融合后修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],xgb_pred)**0.5))
final_pred = final_pred/np.mean(final_pred)*1.1946
print('融合后修正后得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],final_pred)**0.5))

#
# print('开始CV 5折训练...')
# scores = []
# t0 = time.time()
# test_preds = np.zeros(len(test_feat))
# kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
# for idx, (train_index, test_index) in enumerate(kf):
#     X_train, X_test = train_feat.iloc[train_index]['loan_sum'], train_feat.iloc[test_index]['loan_sum']
#     y_train, y_test = train_feat.iloc[test_index]['loan_sum'], train_feat.iloc[test_index]['sum_loan']
#
#     train_pool = Pool(train_feat.iloc[train_index]['loan_sum'], train_feat.iloc[test_index]['loan_sum'])
#     val_pool = Pool(test_feat.iloc[test_index]['loan_sum'], test_feat.iloc[test_index]['sum_loan'])
#
#     model = cb.CatBoostRegressor(iterations=2000, depth=6, learning_rate=0.06, eval_metric='RMSE',
#                                  od_type='Iter', od_wait=20, random_seed=42, thread_count=32, \
#                                  bagging_temperature=0.85, rsm=0.85, verbose=True)
#     print("define model done")
#     model.fit(train_pool, use_best_model=True, eval_set=val_pool, verbose=True, )
#     del train_pool, val_pool;
#
#     test_pool = Pool(test_feat)
#     test_submit['label'] += model.predict(test_pool)
#
#
#
#
#
#







for i in np.arange(4,10):
    cb_model = cb.CatBoostRegressor(iterations=2000, depth=5, learning_rate=0.06, eval_metric='RMSE',
                                    od_type='Iter', od_wait=20, random_seed=42, thread_count=7,
                                    bagging_temperature=0.85, rsm=0.85, verbose=False)

    print("define model done")
    cb_model.fit(train_pool, use_best_model=True, eval_set=test_pool, verbose=False)
    cb_pred = cb_model.predict(test_pool)
    print('catboost修正前得分为：{}'.format(mean_squared_error(test_feat[0], cb_pred) ** 0.5))






