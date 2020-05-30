from xindai.feat5 import *
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

train_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\piupiu_11_feat.csv')
# test_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\piupiu_12_feat.csv')
# train_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\wenchao_11_feat.csv')
# test_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\wenchao_12_feat.csv')
# train_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\lida_11_feat.csv')
# test_feat = pd.read_csv(r'C:\Users\csw\Desktop\python\JD\xindai\features\lida_12_feat.csv')


predictors = [f for f in train_feat.columns if f not in ['uid','loan_sum']]
label_mean = train_feat.loan_sum.mean()

def evalerror(pred, df):
    label = df.get_label()
    rmse = mean_squared_error(label, pred) ** 0.5
    return ('RMSE', rmse, False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.loan_sum)

gbm = lgb.cv(params, lgb_train,
             num_boost_round=10000,
             verbose_eval=100,
             feval=evalerror,
             early_stopping_rounds=100)

print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_model_pred = pd.DataFrame(np.zeros((len(test_feat),6)),columns=[
    'lgb_pred','xgb_pred','gbrt_pred','et_pred','rf_pred','cb_pred'])
train_model_pred = pd.DataFrame(np.zeros((len(train_feat),6)),columns=[
    'lgb_pred','xgb_pred','gbrt_pred','et_pred','rf_pred','cb_pred'])
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    print('开始lgb训练...')
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # 'metric': 'mse',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.7,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['loan_sum'].iloc[train_index])
    # lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['loan_sum'].iloc[test_index])
    lgb_model = lgb.train(params, lgb_train, 800)
    train_model_pred['lgb_pred'].iloc[test_index] += lgb_model.predict(train_feat[predictors].iloc[test_index])
    test_model_pred['lgb_pred'] += lgb_model.predict(test_feat[predictors])

print('lgb cv得分：{1}'.format(i,mean_squared_error(train_feat['loan_sum'],test_model_pred[i])**0.5))
