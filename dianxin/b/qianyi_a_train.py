from tool.tool import *
from dianxin.feat1 import *

data_path = 'C:/Users/cui/Desktop/python/dianxin/data/b/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
print('读取train数据...')

cc = ['service_type', 'is_mix_service','is_promise_low_consume',
'net_service', 'gender', 'age', 'online_time','contract_type',
'1_total_fee','2_total_fee', '3_total_fee', '4_total_fee',
'month_traffic','last_month_traffic', 'local_trafffic_month',
'local_caller_time', 'service1_caller_time','service2_caller_time',
'many_over_bill',  'contract_time','pay_times', 'pay_num']
train_old = pd.read_csv('C:/Users/cui/Desktop/python/dianxin/data/' + 'train.csv')
train = pd.read_csv(data_path + 'train_new.csv')
test = pd.read_csv(data_path + 'test_new.csv')
data = train.append(test).append(train_old)
data = data.drop_duplicates(cc)
# test['label'] = np.nan


print('构造特征...')
data_feat = make_feat(data,'online')

print('切分数据...')
test_feat = data_feat[~data_feat['user_id'].isin(train_old['user_id'])].copy()
test_feat2 = data_feat[data_feat['user_id'].isin(train['user_id'])].copy()
train_feat = data_feat[data_feat['user_id'].isin(train_old['user_id'])].copy()

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
xgb_test2 = xgb.DMatrix(test_feat2[predictors], test_feat2['label'])
watchlist = [(xgb_test2, 'val')]

clf = xgb.train(param,
                xgb_train,
                num_boost_round = 1000,
                evals=watchlist,
                verbose_eval=50,
                early_stopping_rounds=50)
feat_imp = pd.Series(clf.get_fscore(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'xgb_feat_imp.csv')

print('开始预测...')
preds = pd.DataFrame(clf.predict(xgb_test))
preds['user_id'] = test_feat['user_id'].values
preds.to_csv(r'C:\Users\cui\Desktop\python\dianxin\submission\xindai_qianyi_pred.csv',index=False)



