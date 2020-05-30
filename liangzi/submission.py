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
train_feat = make_set(train,data_path)
test_feat = make_set(test,data_path)
sumbmission = test_feat[['id']].copy()

train_feat['id'] = train_feat['id'].apply(lambda x: 0-int(x[1:]) if 'p' in x else int(x[1:]))
test_feat['id'] = test_feat['id'].apply(lambda x: 0-int(x[1:]) if 'p' in x else int(x[1:]))

predictors = train_feat.columns.drop(['label','enddate','hy_16.0', 'hy_91.0', 'hy_94.0'])

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
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
    gbm = lgb.train(params, lgb_train, 650)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds = test_preds/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

preds_scatter = get_threshold(test_preds)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':1-test_preds})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')






