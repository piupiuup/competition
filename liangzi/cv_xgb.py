import xgboost as xgb
from liangzi.feat4 import *

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/concat_data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')

print('构造特征...')
train_feat = make_set(train,train,data_path)

def evalerror(pred, df):
    auc = roc_auc_score(df.label,pred)
    return ('auc', auc, False)

predictors = list(train_feat.columns.drop(['id','label','enddate']))

import xgboost
xgb_train = xgboost.DMatrix(train_feat[predictors],train.label)
# xgb_eval = xgboost.DMatrix(test[features],test.label)
print('开始训练...')
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "auc"
    ,"eta"              : 0.01
    ,"max_depth"        : 12
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}


bst = xgboost.cv(params=xgb_params,
                 dtrain=xgb_train,
                 num_boost_round=5000,
                 verbose_eval=50,
                 early_stopping_rounds=100)
























