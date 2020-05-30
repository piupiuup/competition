import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from mayi.xindai.feat6 import *

train_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\lida_11_feat.csv').fillna(-1)
test_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\lida_12_feat.csv').fillna(-1)


predictors = [f for f in test_feat.columns if f not in (['uid','loan_sum']+delect_id)]

def evalerror(pred, df):
    label = df.get_label()
    rmse = mean_squared_error(label, pred) ** 0.5
    return ('RMSE', rmse, False)


print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_model_pred = pd.DataFrame(np.zeros((len(test_feat),4)),columns=[
    'lgb_pred','xgb_pred','gbrt_pred','cb_pred'])
train_model_pred = pd.DataFrame(np.zeros((len(train_feat),4)),columns=[
    'lgb_pred','xgb_pred','gbrt_pred','cb_pred'])
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
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    lgb_model = lgb.train(params, lgb_train,
             valid_sets=lgb_test,
             num_boost_round=10000,
             verbose_eval=100,
             feval=evalerror,
             early_stopping_rounds=100)
    train_model_pred['lgb_pred'].iloc[test_index] += lgb_model.predict(train_feat[predictors].iloc[test_index])
    test_model_pred['lgb_pred'] += lgb_model.predict(test_feat[predictors])


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



from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_model_pred, train_feat.loan_sum)
# import pickle
# pickle.dump(lr,open('lr.model','wb+'))

train_final_pred = lr.predict(train_model_pred)
test_final_pred = lr.predict(test_model_pred)


for i in ['lgb_pred','xgb_pred','gbrt_pred','cb_pred']:
    print('{0} 模型训练集cv得分：{1}'.format(i,mean_squared_error(train_feat['loan_sum'],train_model_pred[i])**0.5))
print()
print('训练集stacking cv得分为：{}'.format(mean_squared_error(train_feat['loan_sum'],train_final_pred)**0.5))

print('模型权重：\n{}'.format(dict(zip(test_model_pred.columns,list(lr.coef_)))))
print('模型均值：\n{}'.format(dict(zip(test_model_pred.columns,test_model_pred.mean().values))))
print('最终均值：{}'.format(np.mean(test_model_pred)))
# submission = pd.DataFrame({'uid':test_feat.uid.values,'pred':test_final_pred/np.mean(test_final_pred)*1.2575})[['uid','pred']]
# submission['pred'] = submission['pred'].apply(lambda x: x if x>0.1 else 0.1)
# submission.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                   index=False, header=None, float_format='%.4f')