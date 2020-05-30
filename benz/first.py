import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from xgboost.sklearn import XGBRegressor

# 获取排名靠前的特征
def get_feat_imp(train,ID=['ID'],target='y'):

    predictors = [x for x in train.columns if x not in ['ID','y']]
    model = XGBRegressor( max_depth=4, learning_rate=0.0045, n_estimators=700,
                          silent=True, objective='reg:linear', nthread=-1, min_child_weight=1,
                          max_delta_step=0, subsample=0.93, seed=27)
    model.fit(train[predictors],train[target])
    feat_imp = pd.Series(model.booster().get_fscore(),index=predictors).sort_values(ascending=False)
    return feat_imp


# read datasets
train = pd.read_csv(r'C:\Users\csw\Desktop\python\benz\data\train.csv')
test = pd.read_csv(r'C:\Users\csw\Desktop\python\benz\data\test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

# 获取排名靠前的特征
imp = get_feat_imp(train)
features = list(imp.head(144).index)
features.append('ID')
test = test[features]
features.append('y')
train = train[features]



##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.decomposition import TruncatedSVD

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(['y'], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(['y'], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(['y'], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(['y'], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(['y'], axis=1))
srp_results_test = srp.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

y_train = train["y"]
y_mean = np.mean(y_train)

### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

'''
xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                  dtrain,
                  num_boost_round=1000, # increase to have better results (~700)
                  early_stopping_rounds=50,
                  verbose_eval=10,
                  show_stdv=False
                 )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))
'''

'''
model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=675, max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5, min_child_samples=10, subsample=0.995, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True)
'''
num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

print(r2_score(dtrain.get_label(), model.predict(dtrain)))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('baseLine.csv', index=False)