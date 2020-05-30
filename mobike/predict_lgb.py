from mobike.feat12 import *

lgb_train_pred_path = '../cache2/lgb_train_pred.csv'
lgb_test_pred_path = '../cache2/lgb_test_pred.csv'



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


    lgb_path = cache_path + 'lgb_model_{}.model'.format(i)
    lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label, categorical_feature=['holiday', 'biketype'])
    lgb_eval = lgb.Dataset(test_eval[predictors], test_eval.label, categorical_feature=['holiday', 'biketype'])
    if (not os.path.exists(lgb_path)):
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 8,
            'num_leaves': 150,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'verbose': 0,
            'seed': 66,
        }
        print('Start training...')
        gbm = lgb.train(params,lgb_train,1500)
        pickle.dump(gbm, open(lgb_path, 'wb+'))
    gbm = pickle.load(open(lgb_path, 'rb+'))
    preds_temp = gbm.predict(test_eval[predictors])
    train_pred_temp.loc[:, 'pred'] = preds_temp
    if train_pred is None:
        train_pred = train_pred_temp
    else:
        train_pred = train_pred.append(train_pred_temp)
    gc.collect()
train_pred.to_csv(lgb_train_pred_path, index=False)




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
    test_pred_temp.loc[:, 'pred'] = 0
    for j in range(5):
        gbm = pickle.load(open(cache_path + 'lgb_model_{}.model'.format(i), 'rb+'))
        preds_temp = gbm.predict(test_feat_temp[predictors])
        test_pred_temp.loc[:, 'pred'] += preds_temp
    test_pred_temp.loc[:, 'pred'] = test_pred_temp.loc[:, 'pred']/5.0
    if test_pred is None:
        test_pred = test_pred_temp
    else:
        test_pred = test_pred.append(test_pred_temp)
test_pred.to_csv(lgb_test_pred_path,  index=False)



