from xindai.feat3 import *
import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2016-11-01'
for i in range(1):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),30).fillna(-1)
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 30),30).fillna(-1)
# train_feat = make_feats('2016-08-15').fillna(-1)
# test_feat = make_feats('2016-11-03').fillna(-1)

train_feat['date_rate'] = train_feat['date_rate']/train_feat['date_rate'].mean()*1.3
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
    'learning_rate': 0.003,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.8,
    'num_leaves': 60,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


lgb_train = lgb.Dataset(train_feat[predictors], train_feat['loan_sum'])
# lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
gbm = lgb.train(params, lgb_train, 2200)
test_preds = gbm.predict(test_feat[predictors])

pred_mean = np.mean(test_preds)
print(pred_mean)
submission = pd.DataFrame({'uid':test_feat.uid.values,'pred':test_preds/pred_mean*1.2575})[['uid','pred']]
submission['pred'] = submission['pred'].apply(lambda x: x if x>0.1 else 0.1)
submission.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, header=None, float_format='%.4f')














