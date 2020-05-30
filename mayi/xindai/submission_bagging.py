from xindai.feat5 import *
import datetime
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
start_date = '2016-11-01'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),30).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)
# train_feat = make_feats('2016-08-15').fillna(-1)
# test_feat = make_feats('2016-11-03').fillna(-1)

# train_feat['date_rate'] = train_feat['date_rate']/train_feat['date_rate'].mean()*1.3
# test_feat['date_rate'] = test_feat['date_rate']/test_feat['date_rate'].mean()
predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum']+delect_id)]
predictors = list(reversed(predictors))

def evalerror(pred, df):
    label = df.get_label().values.copy()
    label_mean = np.mean(label)
    label = label - label_mean
    pred_temp = np.array(pred.copy())
    pred_mean = np.mean(pred_temp)
    pred_temp = pred_temp - pred_mean
    rmse = mean_squared_error(label,pred_temp)**0.5
    return ('rmse',rmse,False)

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

print('开始CV 5折训练...')
scores = []
t0 = time.time()
lgb_pred = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    gbm = lgb.train(params, lgb_train, 750)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    test_preds_sub = gbm.predict(test_feat[predictors])
    lgb_pred += test_preds_sub
lgb_pred = lgb_pred/5
print('CV训练用时{}秒'.format(time.time() - t0))
# lgb_pred = lgb_pred/np.mean(lgb_pred)*1.2575


##############################使用随机森林预测##################################
model = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=10,
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, max_features=50,
                              max_leaf_nodes=None, bootstrap=True,
                              oob_score=False, n_jobs=-1, random_state=66,
                              verbose=0, warm_start=False)
model = model.fit(train_feat[predictors], train_feat['loan_sum'])
rf_pred = model.predict(test_feat[predictors])
# rf_pred = rf_pred/np.mean(rf_pred)*1.2575




##############################使用GBDT预测##################################
model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.01,loss='ls',
                                  max_depth=7,criterion='friedman_mse',
                                  min_samples_split=2, min_samples_leaf=7,
                                  min_weight_fraction_leaf=0.0, subsample=0.7, max_features=9,
                                  max_leaf_nodes=None, random_state=66)
model = model.fit(train_feat[predictors], train_feat['loan_sum'])
gbrt_pred = model.predict(test_feat[predictors])
# gbrt_pred = gbrt_pred/np.mean(gbrt_pred)*1.2575

##############################使用ET预测##################################
model = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, min_samples_split=2,
                             min_samples_leaf=1, max_depth=21, max_features=23,
                            criterion='mse',random_state=66)
model = model.fit(train_feat[predictors], train_feat['loan_sum'])
et_pred = model.predict(test_feat[predictors])
# lgb_pred = lgb_pred/np.mean(lgb_pred)*1.1946
# et_pred = et_pred/np.mean(et_pred)*1.2575


#############################使用xgb预测####################################
xgtrain_x = xgb.DMatrix(train_feat[predictors], train_feat['loan_sum'])
xgtrain_y = xgb.DMatrix(test_feat[predictors])

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 7,
          # 'lambda':100,
          'subsample': 0.8,
          'colsample_bytree': 0.7,
          'min_child_weight': 10,  # 8~10
          'eta': 0.01,
          'seed':66,
          # 'nthread':12
          }
params['silent'] = 1
model = xgb.train(params, xgtrain_x, 260)
xgb_pred = model.predict(xgtrain_y)
# xgb_pred = xgb_pred/np.mean(xgb_pred)*1.2575


##############################使用catboost预测##################################
train_pool = Pool(train_feat[predictors],train_feat['loan_sum'])
test_pool = Pool(test_feat[predictors],test_feat['loan_sum'])

cb_model = cb.CatBoostRegressor(iterations=550, depth=6, learning_rate=0.03, eval_metric='RMSE',
                                 od_type='Iter', od_wait=20, random_seed=42, thread_count=7,
                                 bagging_temperature=0.85, rsm=0.85, verbose=True)

print("define model done")
cb_model.fit(train_pool)
cb_pred = cb_model.predict(test_pool)


model_preds = pd.DataFrame({'lgb_pred':lgb_pred,
                            'rf_pred':rf_pred,
                            'et_pred':et_pred,
                            'gbrt_pred':gbrt_pred,
                            'xgb_pred':xgb_pred,
                            'cb_pred':cb_pred})
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = pickle.load(open('lr.model','rb+'))
final_pred = lr.predict(model_preds)
final_pred = final_pred/np.mean(final_pred)*1.257


submission = pd.DataFrame({'uid':test_feat.uid.values,'pred':final_pred})[['uid','pred']]
submission['pred'] = submission['pred'].apply(lambda x: x if x>0.2 else 0.2)
submission.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, header=None, float_format='%.4f')

























