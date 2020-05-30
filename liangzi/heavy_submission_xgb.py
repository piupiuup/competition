from liangzi.feat4 import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
import xgboost

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
predictors = train_feat.columns.drop(['id','label','enddate','hy_16.0', 'hy_91.0', 'hy_94.0'])

print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgboost.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgboost.DMatrix(test_feat[predictors])

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "auc"
        , "eta": 0.01
        , "max_depth": 12
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    bst = xgboost.train(params=xgb_params,dtrain=xgb_train,num_boost_round=1200)
    test_preds_sub = bst.predict(xgb_eval)
    test_preds11 += test_preds_sub

test_preds11 = test_preds11/5
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat = train_feat.append(train_feat_temp[(train_feat_temp['prov']==12)])
t0 = time.time()
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgboost.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgboost.DMatrix(test_feat[predictors])

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "auc"
        , "eta": 0.01
        , "max_depth": 12
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    bst = xgboost.train(params=xgb_params, dtrain=xgb_train, num_boost_round=1200)
    test_preds_sub = bst.predict(xgb_eval)
    test_preds12 += test_preds_sub

test_preds12 = test_preds12/5
print('CV训练用时{}秒'.format(time.time() - t0))

test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':test_feat['pred'].values})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)






