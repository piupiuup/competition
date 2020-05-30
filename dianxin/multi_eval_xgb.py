from tool.tool import *
from dianxin.feat1 import *

data_path = 'C:/Users/cui/Desktop/python/dianxin/data/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
wd = {89950166: 0.3915909056178194, 89950167: 1.053882399221542, 89950168: 0.4452184327151803,
 90063345: 1.5021957572587552, 90109916: 0.9646354439100757, 90155946: 1.1615294041000743,
 99999825: 0.6014659236944603, 99999826: 0.6328728469602657, 99999827: 0.5172497567920877,
 99999828: 0.6474052561792734, 99999830: 1.1096814149919525}
print('读取train数据...')
train = pd.read_csv(data_path + 'train.csv')
# test = pd.read_csv(data_path + 'test.csv')
# test['label'] = np.nan


print('构造特征...')
train_feat = make_feat(train,'offline')
train['label'] = train['current_service'].map(d).astype(int)

print('切分数据...')
select_index = list(train_feat.index[::2])
test_feat = train_feat.iloc[select_index, :]
train_feat = train_feat.iloc[~train.index.isin(select_index), :]

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


xgb_train = xgb.DMatrix(train_feat[predictors], train_feat['label'], weight=train['current_service'].map(wd))
xgb_eval = xgb.DMatrix(test_feat[predictors], test_feat['label'])
watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]

clf = xgb.train(param,
                xgb_train,
                num_boost_round = 1000,
                evals=watchlist,
                verbose_eval=50,
                early_stopping_rounds=50)



print('开始预测...')
preds = clf.predict(xgb_eval)
int_preds = pd.Series(preds.argmax(axis=1))
print('预估得分：    {}'.format(exp_multi_f1(preds,int_preds)**2))
print('线下的得分为：  {}'.format(multi_f1(test_feat['label'],int_preds)**2))






