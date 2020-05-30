from tool.tool import *
from credit.feat1 import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 尝试分组标准化
# 从时间序列的角度考虑  滑窗
# 增加样本


cache_path = 'E:/credit/'
data_path = 'C:/Users/cui/Desktop/python/credit/data/'

cate_feat = ['CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','ORGANIZATION_TYPE']


train = pd.read_csv(data_path + 'application_train.csv').rename(columns = {'TARGET':'label'})
test_id = train.sample(frac=0.5,random_state=66)['SK_ID_CURR'].values
test_y = train[train['SK_ID_CURR'].isin(test_id)]['label'].copy()
train.loc[train['SK_ID_CURR'].isin(test_id),'label'] = np.nan

data = make_feat(train,'offline')
# data = compress(data)

test_feat = data[data['SK_ID_CURR'].isin(test_id)].copy()
train_feat = data[~data['SK_ID_CURR'].isin(test_id)].copy()



predictors = [c for c in train_feat.columns if c not in ['label']]

t0 = time.time()
print('开始训练...')

params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 70,
        'max_depth': 9,
        'subsample': 1,
        'colsample_bytree': 0.1,
        'min_split_gain': 0.09,
        'min_child_weight': 9.5,
        'reg_alpha': 1,
        'reg_lambda': 50,
        'verbose': 1
    }
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_y,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
preds = gbm.predict(test_feat[predictors])
print('线下得分：{}'.format(roc_auc_score(test_y,preds)))
print('CV训练用时{}秒'.format(time.time() - t0))