from tool.tool import *
from dianxin.feat1 import *

data_path = 'C:/Users/cui/Desktop/python/dianxin/data/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
print('读取train数据...')
train = pd.read_csv(data_path + 'train.csv')
train['label'] = train['current_service'].map(d).astype(int)
test = pd.read_csv(data_path + 'test.csv')
data = train.append(test)
# test['label'] = np.nan


print('构造特征...')
data_feat = make_feat(data,'online')

print('切分数据...')
test_feat = data_feat[data_feat['user_id'].isin(test['user_id'])].copy()
train_feat = data_feat[data_feat['user_id'].isin(train['user_id'])].copy()

predictors = train_feat.columns.drop(['user_id', 'current_service', 'label'])


print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':11,
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66,
}

lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label, reference=lgb_train)

gbm = lgb.train(params, lgb_train,1000)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

print('开始预测...')
preds = gbm.predict(test_feat[predictors])
int_preds = pd.Series(preds.argmax(axis=1))
test_feat['current_service'] = int_preds.map(rd).values
print('预估得分：    {}'.format(exp_multi_f1(preds,int_preds)**2))
test_feat[['user_id','current_service']].to_csv(r'C:\Users\cui\Desktop\python\dianxin\submission\xindai_sumbmission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),index=False)

