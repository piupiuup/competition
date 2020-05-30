from corporacion.feat4 import *

import gc
import datetime
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

train_feat = pd.DataFrame()
end_date = '2017-07-12'
for j in range(6):
    train_feat_sub = make_feats(date_add_days(end_date, j * (-7))).fillna(-1)
    train_feat = pd.concat([train_feat, train_feat_sub])
eval_feat = make_feats(date_add_days(end_date, 14)).fillna(-1)
test_feat = make_feats(date_add_days(end_date, 35)).fillna(-1)
predictors = [f for f in test_feat.columns if f not in (list(range(16)))]

def evalerror(pred, df):
    label = df.get_label().values.copy()
    rmse = mean_squared_error(label,pred)**0.5
    return ('rmse',rmse,False)

print('开始训练...')
params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 7,
              'lambda': 5,
              'subsample': 0.8,
              'colsample_bytree': 0.7,
              'min_child_weight': 10,  # 8~10
              'eta': 0.1,
              'seed': 66,
              # 'nthread':12
              }

preds = []
trues = []
scores = {}
submission = pd.DataFrame()
for i in range(16):

    print('第{}轮循环'.format(i))
    date = date_add_days('2017-08-16',i)

    train_pool = Pool(train_feat[predictors], train_feat[i])
    eval_pool = Pool(eval_feat[predictors], eval_feat[i])
    test_pool = Pool(test_feat[predictors])
    cb_model = cb.CatBoostRegressor(iterations=1000, depth=5, learning_rate=0.1, eval_metric='RMSE',
                                    od_type='Iter', od_wait=20, random_seed=42, thread_count=7,
                                    bagging_temperature=0.5, rsm=0.85, verbose=False)
    cb_model.fit(train_pool, use_best_model=True, eval_set=eval_pool, verbose=True)
    eval_pred = cb_model.predict(eval_pool)
    test_pred = cb_model.predict(test_pool)
    score = mean_squared_error(eval_feat[i], eval_pred) ** 0.5
    preds.append(eval_pred)
    trues.append(eval_feat[i].values)
    scores[i + 1] = score
    submission = pd.concat([submission,pd.DataFrame({'store_nbr':test_feat['store_nbr'].values,
                                                     'item_nbr':test_feat['item_nbr'].values,
                                                     'date':date,
                                                     'unit_sales':np.exp(test_pred)-1})])
    gc.collect()
# 线下
preds = np.concatenate(preds)
trues = np.concatenate(trues)
print('16天普通lgb得分为：{}'.format(mean_squared_error(trues,preds)**0.5))
# 线上
test = pd.read_csv(data_path + 'test.csv')
submission = test.merge(submission,on=['store_nbr','item_nbr','date'],how='left')
submission['unit_sales'] = submission['unit_sales'].apply(lambda x: x if x>0 else 0)
submission[['id','unit_sales']].to_csv(r'C:\Users\csw\Desktop\python\Corporacion\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, float_format='%.4f')










