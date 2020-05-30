from recruit.feat2 import *
import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
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
test_feat = make_feats(date_add_days(start_date, 84),39)

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
test_feat['pred'] = 0
for i in range(7):
    print('week{}...'.format(i+1))
    train_feat_w = pd.concat(([train_feat]+[train_feat[train_feat['dow']==i]]*2))
    date_dict = {date: diff_of_days(train_feat_w['end_date'].max(), date) for date in train_feat_w['end_date'].unique()}
    weight = 1000 - train_feat_w['end_date'].map(date_dict)
    lgb_train = lgb.Dataset(train_feat_w[predictors], train_feat_w['visitors'], weight=weight)
    lgb_eval = lgb.Dataset(eval_feat[eval_feat['dow']==i][predictors], eval_feat[eval_feat['dow']==i]['visitors'])
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    verbose_eval=100,
                    early_stopping_rounds=100)
    pred_w = gbm.predict(eval_feat[eval_feat['dow']==i][predictors])
    eval_feat.loc[eval_feat['dow']==i,'pred'] = pred_w

print('线下的得分：{}'.format(mean_squared_error(eval_feat['visitors'],eval_feat['pred'])**0.5))














