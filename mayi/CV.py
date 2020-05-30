from mayi.feat2 import *
import lightgbm as lgb
from sklearn.cross_validation import KFold

# cache_path = 'F:/mayi_cache2/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
# test = test.sample(frac=0.1,random_state=66, axis=0)

train_feat = make_feats(train,test).fillna(0)

train_feat = grp_normalize(train_feat,'row_id',['knn','knn2'],start=0)
train_feat = grp_rank(train_feat,'row_id',['cos'],ascending=False)
train_feat = grp_standard(train_feat,'row_id',['shop_count','loc_knn2'])

predictors = train_feat.columns.drop(['row_id','time_stamp', 'user_id',
                                     'shop_id', 'wifi_infos', 'label'])

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 66,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i+1))
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])


    gbm = lgb.train(params, lgb_train, 2300)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))
train_feat[['row_id','shop_id','pred']].to_csv(r'C:\Users\csw\Desktop\python\mayi\data\test_pred-25-31.csv',index=False)

train_feat['pred'] = train_preds
train_feat.sort_values('pred',inplace=True)
result = train_feat.drop_duplicates('row_id',keep='last')
test_test = test
test_test = test_test.merge(result,on='row_id',how='left')
print('准确率为：{}'.format(acc(test_test)))









