from tool.tool import *
from dianxin.feat1 import *

data_path = 'C:/Users/cui/Desktop/python/dianxin/data/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
print('读取train数据...')
wd = {89950166: 0.3915909056178194, 89950167: 1.053882399221542, 89950168: 0.4452184327151803,
 90063345: 1.5021957572587552, 90109916: 0.9646354439100757, 90155946: 1.1615294041000743,
 99999825: 0.6014659236944603, 99999826: 0.6328728469602657, 99999827: 0.5172497567920877,
 99999828: 0.6474052561792734, 99999830: 1.1096814149919525}

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

predictors = [c for c in train_feat.columns if (c not in ['user_id', 'current_service', 'label'])]

print('开始训练...')
param = {'objective': 'multi:softprob',
        'eta': 0.1,
        'max_depth': 6,
        'silent': 1,
        'num_class': 11,
        'eval_metric': "mlogloss",
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 66
}

xgb_train = xgb.DMatrix(train_feat[predictors], train_feat['label'])
xgb_test = xgb.DMatrix(test_feat[predictors])

clf = xgb.train(param, xgb_train,1800)
feat_imp = pd.Series(clf.get_fscore(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'xgb_feat_imp.csv')

print('开始预测...')
preds = clf.predict(xgb_test)
int_preds = pd.Series(preds.argmax(axis=1))
test_feat['current_service'] = int_preds.map(rd).values
print('预估得分：    {}'.format(exp_multi_f1(preds,int_preds)**2))
test_feat[['user_id','current_service']].to_csv(r'C:\Users\cui\Desktop\python\dianxin\submission\xindai_sumbmission_xgb_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),index=False)



