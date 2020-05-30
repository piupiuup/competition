from corporacion.feat4 import *

import gc
import datetime
import xgboost as xgb
import lightgbm as lgb
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
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    # 'sub_feature': 0.7,
    'num_leaves': 30,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.3,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

preds = []
trues = []
scores = {}
submission = pd.DataFrame()
for i in range(16):

    print('第{}轮循环'.format(i))
    date = date_add_days('2017-08-16',i)
    lgb_train = lgb.Dataset(train_feat[predictors], train_feat[i], weight=train_feat["perishable"])
    lgb_eval = lgb.Dataset(eval_feat[predictors], eval_feat[i], weight=eval_feat["perishable"])
    lgb_test = lgb.Dataset(test_feat[predictors], test_feat[i], weight=test_feat["perishable"])

    # gbm = lgb.train(params,lgb_train,1800)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=4000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=200,
                    feval=evalerror,
                    verbose_eval=200,
    )
    test_pred = gbm.predict(test_feat[predictors])
    eval_pred = gbm.predict(eval_feat[predictors])
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










