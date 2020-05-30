from liangzi.feat4 import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat = make_set(train,train,data_path)
test_feat = make_set(train,test,data_path)
sumbmission = test_feat[['id']].copy()

predictors = train_feat.columns.drop(['id','label','enddate','hy_16.0', 'hy_91.0', 'hy_94.0'])

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds1 = np.zeros(len(train_feat))
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    weight = train_feat['id'].iloc[train_index].apply(lambda x: 1 if 's' in x else 2)
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index], weights=weight)
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

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
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds1[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds11 += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds11 = test_preds11/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds2 = np.zeros(len(train_feat))
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    weight = train_feat['id'].iloc[train_index].apply(lambda x: 1 if 's' not in x else 2)
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index], weights = weight)
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

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
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds2[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds12 += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds12 = test_preds12/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

train_feat['pred11'] = test_preds11
train_feat['pred12'] = test_preds12
train_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
print('auc得分:{}'.format(roc_auc_score(train_feat['label'].values,train_feat['pred'].values)))
# preds_scatter = get_threshold(test_feat['pred'].values)
# submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':test_feat['pred'].values})
# submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')






