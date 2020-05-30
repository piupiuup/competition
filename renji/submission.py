from renji.feat2 import *
from sklearn.cross_validation import KFold

train_hdf_path = r'C:\Users\csw\Desktop\python\360\data\train.hdf'
test_hdf_path = r'C:\Users\csw\Desktop\python\360\data\evaluation_public.hdf'

print('读取数据...')
train = pd.read_hdf(train_hdf_path,'w')
test = pd.read_hdf(test_hdf_path,'w')

print('构造特征...')
train_feat = make_set(train)
test_feat = make_set(test)
predictors = train_feat.columns.drop(['id', 'label'])

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds = np.zeros(len(test_feat))
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
        'verbose': 0,
        'seed': 66,
    }
    gbm = lgb.train(params, lgb_train, 450)
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

train_pred = pd.DataFrame({'id':train_feat['id'],'pred':train_preds})
test_pred = pd.DataFrame({'id':test_feat['id'],'pred':test_preds})
data_pred = pd.concat([train_pred,test_pred])
data_pred.to_csv(r'C:\Users\csw\Desktop\python\360\data\data_pred.csv',index=False)

preds_scatter = get_threshold(test_preds)
submission = pd.DataFrame({'id':test['id'].values,'pred':preds_scatter})
submission['pred'] = submission['pred'].map({1:'POSITIVE',0:'NEGATIVE'})
submission.to_csv(r'C:\Users\csw\Desktop\python\360\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, header=None)














