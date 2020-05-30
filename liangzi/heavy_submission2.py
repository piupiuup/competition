from liangzi.feat2 import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/concat_data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat_temp = make_set(train,train,data_path)
test_feat = make_set(train,test,data_path)
sumbmission = test_feat[['id']].copy()

predictors = train_feat_temp.columns.drop(['id','label','enddate','hy_16.0', 'hy_91.0', 'hy_94.0'])
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    # 'max_depth': 20,
    'num_leaves': 150,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 100,
}

print('开始训练1...')
train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['prov']==11])
lgb_train = lgb.Dataset(train_feat[predictors], train_feat['label'])
gbm = lgb.train(params, lgb_train, 1200)
test_preds11 = gbm.predict(test_feat[predictors])

print('开始训练2...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12) & (train_feat_temp['enum'] == -1)])
lgb_train = lgb.Dataset(train_feat[predictors], train_feat['label'])
gbm = lgb.train(params, lgb_train, 1200)
test_preds12 = gbm.predict(test_feat[predictors])

test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':test_feat['pred'].values})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')










