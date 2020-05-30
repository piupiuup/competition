from xindai.feat2 import *
import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat_temp = pd.DataFrame()
start_date = '2016-10-02'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),30).fillna(-1)
    train_feat_temp = pd.concat([train_feat_temp,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)
# train_feat = make_feats('2016-08-15').fillna(-1)
# test_feat = make_feats('2016-11-03').fillna(-1)

train_feat_temp['date_rate'] = train_feat_temp['date_rate']/train_feat_temp['date_rate'].mean()*1.3
test_feat['date_rate'] = test_feat['date_rate']/test_feat['date_rate'].mean()
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
    'learning_rate': 0.005,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.6,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['sum_loan_120']>0])
print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_preds1 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    gbm = lgb.train(params, lgb_train, 1600)
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds1 += test_preds_sub
test_preds1 = test_preds1/5
print('CV训练用时{}秒'.format(time.time() - t0))


train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['sum_loan_120']<=0])
print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_preds2 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    gbm = lgb.train(params, lgb_train, 750)
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds2 += test_preds_sub
test_preds2 = test_preds2/5
print('CV训练用时{}秒'.format(time.time() - t0))

test_feat['pred1'] = test_preds1
test_feat['pred2'] = test_preds2
test_feat['pred'] = test_feat.apply(lambda x: x.pred1 if x.sum_loan_120>0 else x.pred2, axis=1)


loan_sum_mean = test_feat['loan_sum'].mean()
print('得分为： {}'.format(mean_squared_error(test_feat['pred']/np.mean(test_feat['pred'])*loan_sum_mean,test_feat['loan_sum'])**0.5))













