from recruit.feat2 import *
import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2017-01-29'
for i in range(20):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),39)
    train_feat = pd.concat([train_feat,train_feat_sub])
for i in range(1,6):
    train_feat_sub = make_feats(date_add_days(start_date,i*(7)),42-(i*7))
    train_feat = pd.concat([train_feat,train_feat_sub])
eval_feat = make_feats(date_add_days(start_date, 42),39)

# lbl = LabelEncoder()
# lbl.fit(list(train_feat['store_id'].values) + list(eval_feat['store_id'].values) + list(test_feat['store_id'].values))
# train_feat['store_id'] = lbl.transform(train_feat['store_id'].values)
# eval_feat['store_id'] = lbl.transform(eval_feat['store_id'].values)
# test_feat['store_id'] = lbl.transform(test_feat['store_id'].values)
# lbl.fit(list(train_feat['air_area_name'].values) + list(eval_feat['air_area_name'].values) + list(test_feat['air_area_name'].values))
# train_feat['air_area_name'] = lbl.transform(train_feat['air_area_name'].values)
# eval_feat['air_area_name'] = lbl.transform(eval_feat['air_area_name'].values)
# test_feat['air_area_name'] = lbl.transform(test_feat['air_area_name'].values)

predictors = [f for f in eval_feat.columns if f not in (['id','store_id','visit_date','end_date','air_area_name','visitors'])]

params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 5,
              'lambda': 5,
              'subsample': 0.8,
              'colsample_bytree': 0.7,
              'min_child_weight': 10,  # 8~10
              'eta': 0.05,
              'seed': 66,
              # 'nthread':12
              }

t0 = time.time()
xgb_train = xgb.DMatrix(train_feat[predictors], train_feat['visitors'])
xgb_eval = xgb.DMatrix(eval_feat[predictors], eval_feat['visitors'])

watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]
model = xgb.train(params,
                  xgb_train,
                  1000,
                  evals=watchlist,
                  verbose_eval=20,
                  early_stopping_rounds=20)
xgb_eval_pred = model.predict(xgb_eval)
print('线下的得分：{}'.format(mean_squared_error(eval_feat['visitors'],xgb_eval_pred)**0.5))


















