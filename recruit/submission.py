from recruit.feat2 import *
import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train_feat = pd.DataFrame()
start_date = '2017-03-12'
for i in range(58):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),39)
    train_feat = pd.concat([train_feat,train_feat_sub])
for i in range(1,6):
    train_feat_sub = make_feats(date_add_days(start_date,i*(7)),42-(i*7))
    train_feat = pd.concat([train_feat,train_feat_sub])
test_feat = make_feats(date_add_days(start_date, 42),39)


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
date_dict = {date:diff_of_days(train_feat['end_date'].max(),date) for date in train_feat['end_date'].unique()}
weight = 1000-train_feat['end_date'].map(date_dict)
lgb_train = lgb.Dataset(train_feat[predictors], train_feat['visitors'],weight=weight)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat['visitors'])

gbm = lgb.train(params,lgb_train,2300)

pred = gbm.predict(test_feat[predictors])
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

print('训练用时{}秒'.format(time.time() - t0))
subm = pd.DataFrame({'id':test_feat.store_id + '_' + test_feat.visit_date,'visitors':np.expm1(pred)})
subm = submission[['id']].merge(subm,on='id',how='left').fillna(0)
subm[['id','visitors']].to_csv(r'C:\Users\csw\Desktop\python\recruit\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False,  float_format='%.4f')







