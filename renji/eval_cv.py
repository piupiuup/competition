from mayi.feat2 import *
import datetime
import lightgbm as lgb
from sklearn.cross_validation import KFold

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

train_feat = make_feats(train,test).fillna(0)

train_feat = grp_normalize(train_feat,'row_id',['knn','knn2'],start=0)
train_feat = grp_rank(train_feat,'row_id',['cos'],ascending=False)
train_feat = grp_standard(train_feat,'row_id',['shop_count','loc_knn2'])

predictors = train_feat.columns.drop(['row_id','time_stamp', 'user_id',
                                     'shop_id', 'wifi_infos', 'label'])


print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

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
        'verbose': -1,
        'seed': 66,
    }
    gbm = lgb.train(params, lgb_train, 700)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub

train_feat['pred'] = train_preds
print('auc平均得分: {}'.format(np.mean(acc(train_feat))))
print('CV训练用时{}秒'.format(time.time() - t0))

train_pred = train_feat[['row_id','shop_id','pred']].copy()
train_pred.to_csv(r'C:\Users\csw\Desktop\python\mayi\data\data_pred_25_31.csv',index=False)





















