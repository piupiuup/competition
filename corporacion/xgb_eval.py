from corporacion.feat4 import *

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
params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 8,
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
for i in range(16):
    print('第{}轮循环'.format(i))
    xgb_train = xgb.DMatrix(train_feat[predictors], train_feat[i])
    xgb_eval = xgb.DMatrix(test_feat[predictors], test_feat[i])

    watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]
    model = xgb.train(params,
                      xgb_train,
                      1000,
                      evals=watchlist,
                      verbose_eval=50,
                      early_stopping_rounds=50)
    pred = model.predict(xgb_eval)
    score = mean_squared_error(test_feat[i],pred)**0.5
    print('第{}天得分为：{}'.format(i+1,score))
    preds.append(pred)
    trues.append(test_feat[i].values)
    scores[i+1] = score
    gc.collect()
preds = np.concatenate(preds)
trues = np.concatenate(trues)
print('16天普通lgb得分为：{}'.format(mean_squared_error(trues,preds)**0.5))



