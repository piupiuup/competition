import xgboost as xgb
from liangzi.feat3 import *
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
train_feat = make_set(train,train,data_path).fillna(-1)
test_feat = make_set(train,test,data_path).fillna(-1)

predictors = train_feat.columns.drop(['label'])
xgb_test = xgb.DMatrix(test_feat[predictors], test_feat.label)

print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgb.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgb.DMatrix(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

    xgb_params = {'objective': 'binary:logistic',
                  'eval_metric': ['auc'],
                  'eta': 0.05,
                  'max_depth': 6,
                  'subsample': 0.7,
                  'colsample_bytree': 0.7,
                  'lambda': 500,
                  'alpha': 5,
                  'silent': 1,
                  'verbose_eval': True}

    gbm = xgb.train(xgb_params,xgb_train,1000)
    train_preds_sub = gbm.predict(xgb_eval)
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(xgb_test)
    test_preds += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds = test_preds/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))


submission = pd.DataFrame({'EID':test_feat['id'],'FORTARGET':0,'PROB':test_preds})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')






