import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

# 获取排名靠前的特征
def get_feat_imp(train,ID='id',target='price_doc'):

    predictors = [x for x in train.columns if x not in [ID,target]]
    model = XGBRegressor( max_depth=5, learning_rate=0.05, n_estimators=385,
                          silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                          max_delta_step=0, subsample=0.93, seed=27)
    model.fit(train[predictors],train[target])
    feat_imp = pd.Series(model.booster().get_fscore(),index=predictors).sort_values(ascending=False)
    return feat_imp

# 读取数据
train_path = r"C:\Users\csw\Desktop\python\house\data\train.csv"
test_path = r"C:\Users\csw\Desktop\python\house\data\test.csv"
macro_path = r"C:\Users\csw\Desktop\python\house\data\macro.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
id_test = test.id

mult = 0.969

train["price_doc"] = train["price_doc"] * mult + 10


for c in train.columns:
    if train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))

for c in test.columns:
    if test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

imp = get_feat_imp(train.drop(["timestamp"],axis=1))
features = list(imp.head(100).index)
features.append('id')
test = test[features]
features.append('price_doc')
train = train[features]


dtrain = xgb.DMatrix(train.drop(["id", "price_doc"], axis=1), train["price_doc"])
dtest = xgb.DMatrix(test.drop(["id"], axis=1))

num_boost_rounds = 385  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output2 = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
print('log平均分为：%f' % np.log(output2['price_doc'].values).mean())
'''
a榜最后得分  0.31191
'''