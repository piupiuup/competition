import xgboost as xgb
from liangzi.feat4 import *
import pandas as pd
from sklearn.linear_model import LogisticRegression


print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/eval/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat = make_set(train,train,data_path).fillna(-1)
test_feat = make_set(train,test,data_path).fillna(-1)
train_feat['id'] = train_feat['id'].apply(lambda x: 0-int(x[1:]) if 'p' in x else int(x[1:]))
test_feat['id'] = test_feat['id'].apply(lambda x: 0-int(x[1:]) if 'p' in x else int(x[1:]))

predictors = train_feat.columns.drop(['label','enddate','hy_0.0', 'hy_12.0', 'hy_16.0', 'hy_57.0',
                                      'hy_76.0', 'hy_90.0','hy_91.0', 'hy_93.0', 'hy_94.0', 'hy_95.0',
                                      'hy_96.0'])


print('开始训练...')
lr_model = LogisticRegression(C = 1.0, penalty = 'l1')
lr_model.fit(train_feat[predictors], train_feat['label'])

print('开始预测...')
preds = lr_model.predict_proba(test_feat[predictors])
print('线下AUC得分为： {}'.format(roc_auc_score(test_feat.label,preds[:,1])))
# preds_scatter = get_threshold(preds)
# print('线下F1得分为： {}'.format(f1_score(test_feat.label,preds_scatter)))







