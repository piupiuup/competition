from tool.tool import *
from dianxin.feat1 import *


data_path = 'C:/Users/cui/Desktop/python/dianxin/data/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
cc = ['service_type', 'is_mix_service','is_promise_low_consume',
'net_service', 'gender', 'age', 'online_time','contract_type',
'1_total_fee','2_total_fee', '3_total_fee', '4_total_fee',
'month_traffic','last_month_traffic', 'local_trafffic_month',
'local_caller_time', 'service1_caller_time','service2_caller_time',
'many_over_bill',  'contract_time','pay_times', 'pay_num']
print('读取train数据...')
train = pd.read_csv(data_path + 'train.csv')
train['label'] = train['current_service'].map(d).astype(int)
train = train.drop_duplicates(cc)
test = pd.read_csv(data_path + 'test.csv')
data = train.append(test)
# test['label'] = np.nan


print('构造特征...')
data_feat = make_feat(data,'online')

print('切分数据...')
test_feat = data_feat[data_feat['user_id'].isin(test['user_id'])].copy()
train_feat = data_feat[data_feat['user_id'].isin(train['user_id'])].copy()

predictors = train_feat.columns.drop(['user_id', 'current_service', 'label'])



print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros((len(train_feat),11))
test_preds = np.zeros((len(test_feat),11))
xgb_test = xgb.DMatrix(test_feat[predictors])
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgb.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgb.DMatrix(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

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
    watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]

    clf = xgb.train(param,
                    xgb_train,
                    num_boost_round=3000,
                    evals=watchlist,
                    verbose_eval=50,
                    early_stopping_rounds=50)

    train_preds[test_index] += clf.predict(xgb_eval)
    test_preds += clf.predict(xgb_test)

preds = test_preds.copy()/5
train_int_preds = pd.Series(train_preds.argmax(axis=1))
print('线下的得分为：  {}'.format(multi_f1(train_feat['label'],train_int_preds)**2))
train_preds = pd.DataFrame(train_preds,columns=[str(i)+'_xgb1' for i in range(11)])
train_preds['user_id'] = train_feat['user_id'].values
test_preds = pd.DataFrame(test_preds/5,columns=[str(i)+'_xgb1' for i in range(11)])
test_preds['user_id'] = test_feat['user_id'].values
data_preds = train_preds.append(test_preds)
data_preds.to_csv( r'C:\Users\cui\Desktop\python\dianxin\submission\data_preds_xgb1.csv', index=False)

int_preds = pd.Series(preds.argmax(axis=1))
test_feat['current_service'] = int_preds.map(rd).values
print('预估线上得分：    {}'.format(exp_multi_f1(preds,int_preds)**2))
test_feat[['user_id','current_service']].to_csv(r'C:\Users\cui\Desktop\python\dianxin\submission\xindai_sumbmission_xgb1_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),index=False)
























