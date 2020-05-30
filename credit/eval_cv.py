from tool.tool import *
from credit.feat1 import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 尝试分组标准化



cache_path = 'E:/credit/'
data_path = 'C:/Users/cui/Desktop/python/credit/data/'

cate_feat = ['CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','ORGANIZATION_TYPE']


train = pd.read_csv(data_path + 'application_train.csv').rename(columns = {'TARGET':'label'})
train_y = train['label'].values

data = make_feat(train,'offline')



predictors = [c for c in data.columns if c not in ['label']]

t0 = time.time()
print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 32,
    'learning_rate': 0.02,
    'subsample': 0.8715623,
    'colsample_bytree': 0.9497036,
    'reg_alpha': 0.04,
    'reg_lambda': 0.073,
    'min_split_gain': 0.0222415,
    'min_child_weight':40,
    'verbose': -1,
    'seed': 66,
}
lgb_train = lgb.Dataset(data[predictors], train_y)

gbm = lgb.cv(params,
                lgb_train,
                num_boost_round=10000,
                verbose_eval = 50,
                early_stopping_rounds=100)
print('CV训练用时{}秒'.format(time.time() - t0))


