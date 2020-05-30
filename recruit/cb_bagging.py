from recruit.feat2 import *
import datetime
import catboost as cb
import xgboost as xgb
from tqdm import tqdm
import lightgbm as lgb
from catboost import Pool
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error



start_date = '2017-03-12'
periods = [[14,28,56,1000],
           [3,15,40,200],
           [5,17,45,230],
            [7,19,50,260],
            [9,21,55,290],
            [11,23,60,320],
            [13,25,65,350],
            [20,50,150,1000],
            [10,40,100,1000],
            [10,25,80,300],
          ]
preds = []
for period in periods:
    print(period)
    train_feat = pd.DataFrame()
    for i in range(58):
        train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),39,period)
        train_feat = pd.concat([train_feat,train_feat_sub])
    for i in range(1,6):
        train_feat_sub = make_feats(date_add_days(start_date,i*(7)),42-(i*7),period)
        train_feat = pd.concat([train_feat,train_feat_sub])
    test_feat = make_feats(date_add_days(start_date, 42),39,period)


    predictors = [f for f in test_feat.columns if f not in (['id','store_id','visit_date','end_date','air_area_name','visitors'])]


    t0 = time.time()
    cb_train = Pool(train_feat[predictors], train_feat['visitors'])
    cb_test = Pool(test_feat[predictors], test_feat['visitors'])
    cb_model = cb.CatBoostRegressor(iterations=830, depth=7, learning_rate=0.06, eval_metric='RMSE',
                                    od_type='Iter', od_wait=20, random_seed=42,
                                    bagging_temperature=0.85, rsm=0.85, verbose=False)

    cb_model.fit(cb_train)
    cb_eval_pred = cb_model.predict(cb_test)
    preds.append(cb_eval_pred)
    gc.collect()

print('训练用时{}秒'.format(time.time() - t0))
subm = pd.DataFrame({'id':test_feat.store_id + '_' + test_feat.visit_date,'visitors':np.expm1(np.mean(preds,axis=0))})
subm = submission[['id']].merge(subm,on='id',how='left').fillna(0)
subm[['id','visitors']].to_csv(r'C:\Users\csw\Desktop\python\recruit\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False,  float_format='%.4f')