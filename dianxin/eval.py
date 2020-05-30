from tool.tool import *
from dianxin.feat1 import *

# F1最优化
# stacking


d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
cc = ['service_type', 'is_mix_service', 'online_time', '1_total_fee',
       '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
       'many_over_bill', 'contract_type', 'contract_time',
       'net_service', 'pay_times', 'pay_num',
       'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
       'service1_caller_time', 'service2_caller_time', 'complaint_level']
print('读取train数据...')
train = pd.read_csv(data_path + 'train.csv')
train = train.drop_duplicates(cc)
# test = pd.read_csv(data_path + 'test.csv')
# test['label'] = np.nan


print('构造特征...')
train_feat = make_feat(train,'offline')
train['label'] = (train['current_service']==89950166).astype(int)

print('切分数据...')
select_index = list(train_feat.index[::2])
test_feat = train_feat.iloc[select_index, :]
train_feat = train_feat.iloc[~train.index.isin(select_index), :]

predictors = [c for c in train_feat.columns if (c not in ['user_id', 'current_service', 'label'])]

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
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

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval=50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')


print('开始预测...')
preds = gbm.predict(test_feat[predictors])
int_preds = pd.Series(preds.argmax(axis=1))
print('预估得分：    {}'.format(exp_multi_f1(preds,int_preds)**2))
print('线下的得分为：  {}'.format(multi_f1(test_feat['label'],int_preds)**2))


