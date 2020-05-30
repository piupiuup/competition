from mayi.feat2 import *
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
# test = test.sample(frac=0.1,random_state=66, axis=0)

data_feat = make_feats(train,test).fillna(0)

train_feat = data_feat[data_feat['time_stamp']<'2017-08-29'].copy()
test_feat = data_feat[data_feat['time_stamp']>='2017-08-29'].copy()
train_feat = grp_normalize(train_feat,'row_id',['knn','knn2'],start=0)
test_feat = grp_normalize(test_feat,'row_id',['knn','knn2'],start=0)
train_feat = grp_rank(train_feat,'row_id',['cos'],ascending=False)
test_feat = grp_rank(test_feat,'row_id',['cos'],ascending=False)
train_feat = grp_standard(train_feat,'row_id',['shop_count','loc_knn2'])
test_feat = grp_standard(test_feat,'row_id',['shop_count','loc_knn2'])

predictors = data_feat.columns.drop(['row_id','time_stamp', 'user_id',
                                     'shop_id', 'wifi_infos', 'label'])


print('开始训练...')
lr_model = LogisticRegression(C = 1.0, penalty = 'l1')
lr_model.fit(train_feat[predictors], train_feat['label'])

print('开始预测...')
preds = lr_model.predict_proba(test_feat[predictors])

test_feat['pred'] = preds[:,1]
test_feat.sort_values('pred',inplace=True)
result = test_feat.drop_duplicates('row_id',keep='last')
test_test = test[test['time_stamp']>='2017-08-29']
test_test = test_test.merge(result,on='row_id',how='left')
print('准确率为：{}'.format(acc(test_test)))









