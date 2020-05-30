from liangzi.feat4 import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/concat_data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat_temp = make_set(train,train,data_path)
test_feat = make_set(train,test,data_path)
sumbmission = test_feat[['id']].copy()

train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['prov']==11])
train_feat = train_feat.append(train_feat_temp[train_feat_temp['prov']==11])
train_feat['pred11'] = 0
predictors = [f for f in train_feat.columns if f not in ['id','label','pred11','pred12','enddate','hy_16.0', 'hy_91.0', 'hy_94.0']]

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat_temp), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].loc[train_index], train_feat['label'].loc[train_index])
    lgb_test = lgb.Dataset(train_feat[predictors].loc[test_index], train_feat['label'].loc[test_index])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        # 'max_depth': 20,
        'num_leaves': 150,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 100,
    }
    gbm = lgb.train(params, lgb_train, 900)
    train_preds_sub = gbm.predict(train_feat[predictors].loc[test_index])
    train_feat.loc[test_index,'pred11'] = train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds11 += test_preds_sub

test_preds11 = test_preds11/5
train_pred11 = train_feat[train_feat.prov==11][['id','pred11']].drop_duplicates()
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat = train_feat.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat['pred12'] = 0
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat_temp), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].loc[train_index], train_feat['label'].loc[train_index])
    lgb_test = lgb.Dataset(train_feat[predictors].loc[test_index], train_feat['label'].loc[test_index])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 20,
        'num_leaves': 150,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 100,
    }
    gbm = lgb.train(params, lgb_train, 900)
    train_preds_sub = gbm.predict(train_feat[predictors].loc[test_index])
    train_feat.loc[test_index, 'pred12'] = train_preds_sub
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds12 += test_preds_sub

test_preds12 = test_preds12/5
train_pred12 = train_feat[train_feat.prov==12][['id','pred12']].drop_duplicates()
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

train_pred = pd.concat([train_pred11,train_pred12])
test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':1-test_feat['pred'].values})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')











