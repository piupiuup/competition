# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from matplotlib.pylab import rcParams



def predict_final(train, test,  silent=1,alg=XGBClassifier(learning_rate=0.05,n_estimators=150)):

   train = train.fillna(-1)
   test = test.fillna(-1)
   target = 'subsidy'
   IDcol = 'studentid'
   predictors = ['studentid', 'var_of_consume', 'consume_of_day', 'loc248', 'year', 'mean_of_meal', 'loc829', 'std_of_meal', 'std_meal_perday',
 'sum_quar_4', 'mean_of_storage', 'loc72', 'percent_of_week', 'sum_of_bath', 'loc526', 'loc6',
 'loc118', 'in5', 'loc257', 'loc2260', 'percent_of_bath', 'loc232', 'skew_of_wash', 'loc283', 'in4', 'loc2031', 'loc2151',
 'loc277', 'loc2324', 'min_of_consume', 'std_of_storage', 'sum_of_consume', 'in3', 'mean_of_over_storage', 'sum_quar_1', 'rank_of_relative',
 'count_of_storage', 'loc996', 'loc251', 'skew_of_meal', 'loc91', 'loc217', 'percent_of_meal', 'loc269', 'loc531', 'loc254', 'mean_quar_3',
 'loc818', 'loc246', 'loc272', 'day_of_borrow', 'loc306', 'percent_of_>1000', 'loc1113', 'loc915', 'count_quar_3', 'loc2107',
 'mean_of_library', 'loc105', 'count_of_borrow', 'loc57', 'loc2119', 'loc664', 'loc179', 'loc1562', 'out2', 'mean_of_print',
 'loc1368', 'day_of_meal', 'kurt_of_consume', 'out1', 'loc227', 'loc1167', 'loc2043',  'loc263', 'count_quar_4',
 'count_quar_1', 'count_of_meal', 'loc812', 'loc784', 'std_of_bath', 'loc832', 'loc585', 'max_of_bus', 'loc88', 'loc144',
 'loc957', 'day_of_shopping', 'sum_of_water', 'loc527', 'sum_of_bus', 'loc300', 'loc286', 'skew_of_consume', 'loc193', 'loc1997',
 'mean_quar_2',  'loc1576', 'loc226', 'loc213', 'loc844', 'loc1049', 'loc1683', 'loc1042', 'mean_of_shopping', 'loc661',
 'sum_of_meal', 'loc349', 'loc340', 'loc929', 'loc204', 'loc2279', 'loc237', 'kurt_of_bath', 'day_of_water', 'loc113', 'count_of_print',
 'loc2171', 'loc307', 'loc326', 'loc1047', 'loc478', 'loc1545', 'count_of_loc', 'loc552', 'loc214', 'loc250', 'skew_of_bus',
 'sum_quar_3', 'loc840', 'max_of_print', 'loc794', 'loc841', 'loc846', 'loc863', 'sum_of_print']
   predictors2 = ['loc72', 'loc283', 'kurt_of_bus', 'mean_quar_2', 'consume_of_day', 'std_consum_week', 'loc996', 'mean_of_storage', 'count_of_bath',
 'loc2107', 'skew_consum_day', 'loc105', 'percent_of_>1000', 'min_of_consume', 'mean_of_shopping', 'sum_of_bath', 'loc2119', 'loc282',
 'skew_of_bath', 'skew_consum_week', 'skew_consum_quarter', 'std_consum_day', 'loc263', 'max_of_consume', 'var_of_consume', 'kurt_of_bath',
 'loc1479', 'loc397', 'skew_of_wash', 'count_dorm_perday', 'sum_of_library', 'skew_consum_month', 'loc228', 'mean_of_meal', 'skew_of_meal',
 'max_consum_month', 'mean_of_over_storage', 'loc27', 'loc269', 'loc679', 'percent_of_bath', 'kurt_of_consume', 'loc346', 'loc1076',
 'count_of_loss', 'loc168', 'loc2031', 'loc191', 'count_of_library', 'mean_consum_day', 'loc258', 'loc1368', 'loc1985',
 'day_of_bath', 'count_of_meal', 'loc796', 'loc91', 'loc113', 'null', 'count_of_permeal', 'studentid']
   predictors3 = ['count_of_storage', 'percent_of_>1500', 'loc251', 'count_quar_3', 'count_quar_4', 'loc2043', 'count_of_bath', 'loc213',
 'loc226', 'std_of_meal', 'min_of_consume', 'difference_of_consume', 'loc257', 'day_of_borrow', 'loc241', 'loc124', 'max_of_library',
 'kurt_consum_month', 'skew_of_bus', 'skew_consum_week', 'loc192', 'loc219', 'loc182', 'sum_quar_3', 'mean_quar_1', 'loc254',
 'rank_of_relative', 'percent_of_print', 'percent_of_shopping', 'percent_difference_consume', 'studentid']

   train_1 = train.copy()
   train_1['subsidy'] = train_1['subsidy'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

   train_2 = train.copy()
   train_2 = train_2[train['subsidy'] > 0]
   train_2['subsidy'] = train_2['subsidy'].map({1000: 0, 1500: 1, 2000: 1})

   train_3 = train.copy()
   train_3 = train_3[train['subsidy'] > 1000]
   train_3['subsidy'] = train_3['subsidy'].map({1500: 0, 2000: 1})

   # model of selecting id which big 0
   alg1 = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,  n_estimators=160, nthread=-1, subsample=0.6)
   alg1.fit(train_1[predictors], train_1[target])
   result1 = alg1.predict_proba(test[predictors])
   result1 = pd.DataFrame({1:result1[:, 1], 'studentid':test['studentid'].values})
   result1.sort_values(1,inplace=True,ascending=False)
   n1000 = int(len(test)*0.24)
   test_id_1000 = list(result1.head(n1000)['studentid'].values)

   # model of select id which big 1000
   test_2_id = [i in test_id_1000 for i in test['studentid'].values]
   test_2 = test[test_2_id]
   alg2 = XGBClassifier(learning_rate=0.05,n_estimators=300,max_depth=3,subsample=1)
   alg2.fit(train_2[predictors2], train_2[target])
   result2 = alg2.predict_proba(test_2[predictors2])
   result2 = pd.DataFrame({1:result2[:, 1], 'studentid':test_2['studentid'].values})
   result2.sort_values(1,inplace=True,ascending=False)
   n1500 = int(len(test)*0.09)
   test_id_1500 = list(result2.head(n1500)['studentid'].values)

   # model of select id which big 1000
   test_3_id = [i in test_id_1500 for i in test['studentid'].values]
   test_3 = test[test_3_id]
   alg3 = XGBClassifier(learning_rate=0.05,n_estimators=250,max_depth=3,subsample=0.96)
   alg3.fit(train_3[predictors3], train_3[target])
   result3 = alg3.predict_proba(test_3[predictors3])
   result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]})
   result3.sort_values(1,inplace=True,ascending=False)
   n2000 = int(len(test)*0.03)
   test_id_2000 = list(result3.head(n2000)['studentid'].values)

   # 将id对应的助学金合并起来
   subsidy = []
   for x in test['studentid']:
       if x in test_id_1000:
           if x in test_id_1500:
               if x in test_id_2000:
                   subsidy.append(2000)
               else:
                   subsidy.append(1500)
           else:
               subsidy.append(1000)
       else:
           subsidy.append(0)

   result = pd.DataFrame({'studentid': test['studentid'], 'subsidy': subsidy})

   if silent == 0: print result['subsidy'].value_counts()

   return result