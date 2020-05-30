from tool.tool import *
from tool.tool_model import *
from dianxin.feat1 import *
from dianxin.b.white_user_id import *


data_path = 'C:/Users/cui/Desktop/python/dianxin/data/b/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
d1 = {0:0,4:1,8:2}
rd1 = {0:0,1:4,2:8}
d4 = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7}
rd4 = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 9, 7: 10}

cc = ['service_type', 'is_mix_service','is_promise_low_consume',
'net_service', 'gender', 'age', 'online_time','contract_type',
'1_total_fee','2_total_fee', '3_total_fee', '4_total_fee',
'month_traffic','last_month_traffic', 'local_trafffic_month',
'local_caller_time', 'service1_caller_time','service2_caller_time',
'many_over_bill',  'contract_time','pay_times', 'pay_num']
print('读取train数据...')
train = pd.read_csv(data_path + 'train_new.csv')
train_old = pd.read_csv('C:/Users/cui/Desktop/python/dianxin/data/' + 'train.csv')
train = train.append(train_old)
train['label'] = train['current_service'].map(d).astype(int)
train = train.drop_duplicates(cc)
test = pd.read_csv(data_path + 'test_new.csv')
data = train.append(test)
# test['label'] = np.nan

print('构造特征...')
data_feat = make_feat(data,'online')
test_feat = data_feat[data_feat['user_id'].isin(test['user_id'])].copy()
train_feat = data_feat[data_feat['user_id'].isin(train['user_id'])].copy()


train_feat1 = train_feat[train_feat['service_type']==1].copy()
test_feat1 = test_feat[test_feat['service_type']==1].copy()
train_feat1['label'] = train_feat1['label'].map(d1)
predictors1 = [c for c in train_feat.columns if (c not in ['user_id', 'current_service', 'label']) and
               ('contract_type' not in c) and ('service_type' not in c)]
params = {'objective': 'multi:softprob',
         'eta': 0.1,
         'max_depth': 6,
         'silent': 1,
         'num_class': 3,
         'eval_metric': "mlogloss",
         'min_child_weight': 3,
         'subsample': 0.7,
         'colsample_bytree': 0.7,
         'seed': 66
         }
train_preds1,test_preds1 = xgb_cv(params,train_feat1,test_feat1,predictors1)
int_train_preds1 = train_preds1.argmax(axis=1)
int_test_preds1 = test_preds1.argmax(axis=1)
print('线下第一类的得分为：  {}'.format(multi_f1(train_feat1['label'],int_train_preds1)**2))
train_preds1 = pd.DataFrame(train_preds1)
train_preds1['user_id'] = train_feat1['user_id'].values
test_preds1 = pd.DataFrame(test_preds1)
test_preds1['user_id'] = test_feat1['user_id'].values
data_pred1 = train_preds1.append(test_preds1)
data_pred1.columns = [rd1[i] if i in rd1 else i for i in data_pred1.columns]


train_feat4 = train_feat[train_feat['service_type']!=1].copy()
test_feat4 = test_feat[test_feat['service_type']!=1].copy()
train_feat4['label'] = train_feat4['label'].map(d4)
predictors4 = [c for c in train_feat.columns if (c not in ['user_id', 'current_service', 'label'])]
params = {'objective': 'multi:softprob',
                 'eta': 0.1,
                 'max_depth': 6,
                 'silent': 1,
                 'num_class': 8,
                 'eval_metric': "mlogloss",
                 'min_child_weight': 3,
                 'subsample': 0.7,
                 'colsample_bytree': 0.7,
                 'seed': 66
                 }
train_preds4,test_preds4 = xgb_cv(params,train_feat4,test_feat4,predictors4)
int_train_preds4 = train_preds4.argmax(axis=1)
int_test_preds4 = test_preds4.argmax(axis=1)
print('线下第四类的得分为：  {}'.format(multi_f1(train_feat4['label'],int_train_preds4)**2))
train_preds4 = pd.DataFrame(train_preds4)
train_preds4['user_id'] = train_feat4['user_id'].values
test_preds4 = pd.DataFrame(test_preds4)
test_preds4['user_id'] = test_feat4['user_id'].values
data_pred4 = train_preds4.append(test_preds4)
data_pred4.columns = [rd4[i] if i in rd4 else i for i in data_pred4.columns]

# 输出预测概率，做stacking使用
data_pred = data_pred1.append(data_pred4).fillna(0)
data_pred.to_csv( r'C:\Users\cui\Desktop\python\dianxin\submission\data_preds_xgb1_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

# 计算cv得分
int_train_preds1 = pd.DataFrame({'user_id':train_feat1['user_id'].values,'pred':[rd1[i] for i in int_train_preds1 ]})
int_train_preds4 = pd.DataFrame({'user_id':train_feat4['user_id'].values,'pred':[rd4[i] for i in int_train_preds4 ]})
int_train_preds = int_train_preds1.append(int_train_preds4)
train_feat = train_feat.merge(int_train_preds,on='user_id',how='left')
train_feat['label'] = train_feat['current_service'].map(d)
print('线下F1得分为：  {}'.format(multi_f1(train_feat['label'],train_feat['pred'])**2))

int_test_preds1 = pd.DataFrame({'user_id':test_feat1['user_id'].values,'current_service':[rd1[i] for i in int_test_preds1 ]})
int_test_preds4 = pd.DataFrame({'user_id':test_feat4['user_id'].values,'current_service':[rd4[i] for i in int_test_preds4 ]})
test_preds = int_test_preds1.append(int_test_preds4)
test_preds['current_service'] = test_preds['current_service'].map(rd)
submission = test_feat[['user_id']].merge(test_preds,on='user_id',how='left')
submission = replace_white_user_id(submission)
submission[['user_id','current_service']].to_csv(r'C:\Users\cui\Desktop\python\dianxin\submission\xindai_sumbmission_xgb1_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),index=False)
























