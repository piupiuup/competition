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


predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum']+delect_id)]
label_mean = test_feat.loan_sum.mean()

def evalerror(pred, df):
    label = df.get_label().values.copy()
    pred_temp = np.array(pred.copy())
    pred_mean = np.mean(pred_temp)
    pred_temp = pred_temp/pred_mean*label_mean
    rmse = mean_squared_error(label,pred_temp)**0.5
    return ('rmse',rmse,False)


print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_model_pred = pd.DataFrame(np.zeros((len(test_feat),6)),columns=[
    'lgb_pred','xgb_pred','gbrt_pred','et_pred','rf_pred','cb_pred'])
train_model_pred = pd.DataFrame(np.zeros((len(train_feat),6)),columns=[
    'lgb_pred','xgb_pred','gbrt_pred','et_pred','rf_pred','cb_pred'])
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    print('开始lgb训练...')
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
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    lgb_model = lgb.train(params, lgb_train, 800)
    train_model_pred['lgb_pred'].iloc[test_index] += lgb_model.predict(train_feat[predictors].iloc[test_index])
    test_model_pred['lgb_pred'] += lgb_model.predict(test_feat[predictors])


    print('开始rf训练...')
    model = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=10,
                                  min_samples_split=2, min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_features=50,
                                  max_leaf_nodes=None, bootstrap=True,
                                  oob_score=False, n_jobs=-1, random_state=66,
                                  verbose=0, warm_start=False)
    rf_model = model.fit(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    train_model_pred['rf_pred'].iloc[test_index] += rf_model.predict(train_feat[predictors].iloc[test_index])
    test_model_pred['rf_pred'] += rf_model.predict(test_feat[predictors])


    print('开始gbrt训练...')
    model = GradientBoostingRegressor(n_estimators=1500,
          max_depth= 8,
          max_leaf_nodes= 20,
          min_samples_leaf= 30,
          learning_rate= 0.01,
          loss= 'ls',
          subsample= 0.6,
          max_features='sqrt',
        random_state=66)
    gbrt_model = model.fit(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    train_model_pred['gbrt_pred'].iloc[test_index] += gbrt_model.predict(train_feat[predictors].iloc[test_index])
    test_model_pred['gbrt_pred'] += gbrt_model.predict(test_feat[predictors])


    print('开始et训练...')
    model = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, min_samples_split=2,
                                 min_samples_leaf=1, max_depth=21, max_features=23,
                                criterion='mse',random_state=66)
    et_model = model.fit(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    train_model_pred['et_pred'].iloc[test_index] += et_model.predict(train_feat[predictors].iloc[test_index])
    test_model_pred['et_pred'] += et_model.predict(test_feat[predictors])


    print('开始xgb训练...')
    xgb_train = xgb.DMatrix(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    xgb_eval = xgb.DMatrix(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    xgb_test = xgb.DMatrix(test_feat[predictors])

    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 8,
              'lambda': 5,
              'subsample': 0.8,
              'colsample_bytree': 0.7,
              'min_child_weight': 10,  # 8~10
              'eta': 0.01,
              'seed': 66,
              # 'nthread':12
              }
    params['silent'] = 1
    watchlist = [(xgb_train, 'train'), (xgb_eval, 'eval')]
    xgb_model = xgb.train(params, xgb_train, 5000, watchlist, early_stopping_rounds=40,verbose_eval = 40)
    train_model_pred['xgb_pred'].iloc[test_index] += xgb_model.predict(xgb_eval)
    test_model_pred['xgb_pred'] += xgb_model.predict(xgb_test)


    print('开始cb训练...')
    train_pool = Pool(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    eval_pool = Pool(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    test_pool = Pool(test_feat[predictors])
    cb_model = cb.CatBoostRegressor(iterations=400, depth=7, learning_rate=0.06, eval_metric='RMSE',
                                 od_type='Iter', od_wait=20, random_seed=42, thread_count=7,
                                 bagging_temperature=0.85, rsm=0.85, verbose=False)
    cb_model.fit(train_pool)
    train_model_pred['cb_pred'].iloc[test_index] += cb_model.predict(eval_pool)
    test_model_pred['cb_pred'] += cb_model.predict(test_pool)

test_model_pred = test_model_pred/5

#
# print('开始CV 5折训练...')
# test_final_pred = np.zeros(len(test_feat))
# train_final_pred = np.zeros(len(train_feat))
# kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
# for i, (train_index, test_index) in enumerate(kf):
#     print('第{}次训练...'.format(i))
#     params = {
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         # 'metric': 'mse',
#         'sub_feature': 0.9,
#         'num_leaves': 10,
#         'colsample_bytree': 0.7,
#         'feature_fraction': 0.9,
#         'min_data': 100,
#         'min_hessian': 1,
#         'verbose': -1,
#     }
#     lgb_train = lgb.Dataset(train_model_pred.iloc[train_index], train_feat['loan_sum'].iloc[train_index])
#     lgb_eval = lgb.Dataset(train_model_pred.iloc[test_index], train_feat['loan_sum'].iloc[test_index])
#     gbm = lgb.train(params, lgb_train,
#              num_boost_round=10000,
#              verbose_eval=50,
#              valid_sets=lgb_eval,
#              feval=evalerror,
#              early_stopping_rounds=50)
#     train_final_pred[test_index] += gbm.predict(train_model_pred.iloc[test_index])
#     test_final_pred += gbm.predict(test_model_pred)
# test_final_pred = test_final_pred/5
# print('训练集得分为：{}'.format(mean_squared_error(train_feat['loan_sum'],train_final_pred)**0.5))
# print('测试集得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],test_final_pred)**0.5))






print('开始CV 5折训练...')
test_final_pred = np.zeros(len(test_feat))
train_final_pred = np.zeros(len(train_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(train_model_pred.iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    train_final_pred[test_index] += lr.predict(train_model_pred.iloc[test_index])
    test_final_pred += lr.predict(test_model_pred)
test_final_pred = test_final_pred/5
print('训练集得分为：{}'.format(mean_squared_error(train_feat['loan_sum'],train_final_pred)**0.5))
print('测试集得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],test_final_pred)**0.5))








from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_model_pred, train_feat.loan_sum)
import pickle
pickle.dump(lr,open('lr.model','wb+'))

train_final_pred = lr.predict(train_model_pred)
test_final_pred = lr.predict(test_model_pred)

print('融合后修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],test_final_pred)**0.5))
# test_final_pred = test_final_pred/np.mean(test_final_pred)*1.1946
# print('融合后修正后得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],test_final_pred)**0.5))

for i in ['lgb_pred','xgb_pred','gbrt_pred','et_pred','rf_pred','cb_pred']:
    print('{0} 模型训练集cv得分：{1}'.format(i,mean_squared_error(train_feat['loan_sum'],test_model_pred[i])**0.5))
print()
print('训练集stacking cv得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],train_final_pred)**0.5))

for i in ['lgb_pred','xgb_pred','gbrt_pred','et_pred','rf_pred','cb_pred']:
    print('{0} 模型测试集得分：{1}'.format(i,mean_squared_error(test_feat['loan_sum'],test_model_pred[i])**0.5))
print()
print('测试集stacking得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],test_final_pred)**0.5))

test_model_pred2 = test_model_pred/test_model_pred.mean()*test_feat['loan_sum'].mean()
# model_preds = pd.DataFrame({'lgb_pred':lgb_pred,
#                             'rf_pred':rf_pred,
#                             'et_pred':et_pred,
#                             'gbrt_pred':gbrt_pred,
#                             'xgb_pred':xgb_pred,
#                             'cb_pred':cb_pred})
# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(model_preds, test_feat.loan_sum)
# import pickle
# pickle.dump(lr,open('lr.model','wb+'))
# final_pred = lr.predict(model_preds)
# print('融合后修正前得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],xgb_pred)**0.5))
# final_pred = final_pred/np.mean(final_pred)*1.1946
# print('融合后修正后得分为：{}'.format(mean_squared_error(test_feat['loan_sum'],final_pred)**0.5))

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














