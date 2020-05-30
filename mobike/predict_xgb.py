import xgboost
from mobike.feat2 import *


xgb_train_pred_path = '../cache2/xgb_train_pred09242.csv'
xgb_test_pred_path = '../cache2/xgb_test_pred09242.csv'



train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train, test])
del train, test

train_feat_path = cache_path + 'train_feat.hdf'
train_feat = pd.read_hdf(train_feat_path,'w')
gc.collect()
predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
                                          'bikeid', 'starttime', 'geohashed_start_loc'])


train_pred = None
for i in [0,1,2,3,4]:
    n_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
    print('###################第{}轮训练####################'.format(i))
    orderid_sample = make_sample(data.orderid.tolist(), n_sub=5, seed=66)
    print('train_feat.shape: {}'.format(train_feat.shape))
    test_eval = train_feat[train_feat['orderid'].isin(orderid_sample[n_dict[i]])]
    train_pred_temp = test_eval[['orderid', 'geohashed_end_loc']]
    print('test_eval shape: {}'.format(test_eval.shape))
    train_eval = train_feat[~train_feat['orderid'].isin(orderid_sample[n_dict[i]])]
    print('train_eval shape: {}'.format(train_eval.shape))

    xgb_train = xgboost.DMatrix(train_eval[predictors], train_eval.label)
    xgb_eval = xgboost.DMatrix(test_eval[predictors],test_eval.label)

    xgb_path = cache_path + 'xgb_model_{}.model'.format(i)
    if (not os.path.exists(xgb_path)):
        xgb_params = {
            "objective"         : "reg:logistic"
            ,"eval_metric"      : "logloss"
            ,"eta"              : 0.05
            ,"max_depth"        : 12
            ,"min_child_weight" : 4
            ,"gamma"            :0.70
            ,"subsample"        :0.76
            ,"colsample_bytree" :0.95
            ,"alpha"            :2e-05
            ,"lambda"           :10
            ,'silent'           :1
        }

        gbm = xgboost.train(params=xgb_params,dtrain=xgb_train,num_boost_round=600)
        pickle.dump(gbm, open(xgb_path, 'wb+'))
    gbm = pickle.load(open(xgb_path, 'rb+'))
    preds_temp = gbm.predict(xgb_eval)
    train_pred_temp.loc[:, 'pred'] = preds_temp
    if train_pred is None:
        train_pred = train_pred_temp
    else:
        train_pred = train_pred.append(train_pred_temp)
    gc.collect()
train_pred.to_csv(xgb_test_pred_path, index=False)




# cv5折预测test_pred
data_list = pd.date_range('2017-05-25 00:00:00', '2017-06-02 00:00:00')
test_pred = None
for i in range(len(data_list) - 1):
    print('###############{}###############'.format(str(data_list[i])))
    gc.collect()
    data_temp = data.copy()
    data_temp.loc[(data_temp['starttime'] >= str(data_list[i])) & (
    data_temp['starttime'] < str(data_list[i + 1])), 'geohashed_end_loc'] = np.nan
    test_data = data_temp[
        (data_temp['starttime'] >= str(data_list[i])) & (data_temp['starttime'] < str(data_list[i + 1]))].copy()
    test_feat_temp = make_train_set(data_temp, test_data).fillna(-1)
    test_pred_temp = test_feat_temp[['orderid', 'geohashed_end_loc']].copy()
    xgb_eval = xgboost.DMatrix(test_feat_temp[predictors],test_feat_temp.label)
    test_pred_temp.loc[:, 'pred'] = 0
    for j in range(5):
        gbm = pickle.load(open(cache_path + 'xgb_model_{}.model'.format(i), 'rb+'))
        preds_temp = gbm.predict(xgb_eval)
        test_pred_temp.loc[:, 'pred'] += preds_temp
    test_pred_temp.loc[:, 'pred'] = test_pred_temp.loc[:, 'pred']/5.0
    if test_pred is None:
        test_pred = test_pred_temp
    else:
        test_pred = test_pred.append(test_pred_temp)
test_pred.to_csv(xgb_train_pred_path,  index=False)




