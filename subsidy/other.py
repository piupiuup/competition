import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA,KernelPCA
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold,ShuffleSplit,KFold
from sklearn.metrics import classification_report,f1_score,accuracy_score
from collections import defaultdict

# 读取数据
select = pd.read_excel('xiaosi1_consume.xlsx')
f1314_train = select.loc[(select.year==0) & (select.subsidy.notnull())]  # 2365 rows × 65 columns

f1415_train = select.loc[(select.year==1) & (select.subsidy.notnull())]  # 4699 rows × 65 columns

f1314_test = select.loc[(select.year==0) & (select.subsidy.isnull())]    # 2278 rows × 65 columns

f1415_test =  select.loc[(select.year==1) & (select.subsidy.isnull())]   # 4628 rows

#突然觉得，1415年的数据算不上不平衡。或者说，可以用扩充样本的方式全部带入SVC、RFC、GBC等。
# 而1314年的数据需要先分为0和非0，再非0类分类。
f1314_traind  = f1314_train.copy()
f1415_traind = f1415_train.copy()
f1314_testc  = f1314_test.copy()
f1415_testc  = f1415_test.copy()
####    1314年训练集分为0类和非0类，并把subsidy标签改为1和-1  #########
f1314_traind.set_value(f1314_traind.subsidy==0,'subsidy',1)
f1314_traind.set_value(f1314_traind.subsidy!=1,'subsidy',-1)
#####   扩充样本   #####################
f1314_traind_n0 = f1314_traind.loc[f1314_traind.subsidy==-1]
for i in range(5):
    f1314_traind = f1314_traind.append(f1314_traind_n0)
### 加入随机扰动##########
f1314_traind['sum_total']=f1314_traind.sum_total.values + [round(1*np.random.randn(),2) for i in range(len(f1314_traind))]
# #####  1314进行特征选择#####
predictors = [x for x in f1314_traind.columns if x not in ['subsidy','year','sum_shower','num_laundry','sum_total','sum_library','sum_checkget','num_checkget']]
sel_1314_train = f1314_traind[predictors]
sel_1314_test = f1314_testc[predictors]
######  准备训练集  ##################
x_1314_train = np.array(sel_1314_train)
x_1314_test = np.array(sel_1314_test)
y_1314_train = np.array(f1314_traind['subsidy'])
##############################################    RFC 1314年 0和非0类 训练及预测  #####################
X1_train, X1_test, Y1_train, Y1_test = train_test_split(x_1314_train, y_1314_train, test_size=0.3,random_state=0)
rfc = RandomForestClassifier(criterion='entropy', max_features='auto' ,\
                             bootstrap=True, oob_score=False, n_jobs=1,class_weight = 'balanced_subsample', random_state=0, verbose=0)
tuned_parameters = {'n_estimators': [250], 'max_depth':  [11],'max_features':['auto'],'min_samples_leaf':[7]}
grid_search1 = GridSearchCV(rfc, param_grid=tuned_parameters, verbose=3, n_jobs=-1,scoring='f1_macro',\
              cv=ShuffleSplit(len(Y1_train) ,n_iter=5, test_size=0.3,random_state=0)).fit(X1_train, Y1_train)
#### 1314  0和非0类的预测及处理#####
predict_1314 = grid_search1.best_estimator_.predict(x_1314_test)
result_1314 = pd.DataFrame({'ID':f1314_test['ID'].as_matrix(), 'subsidy':predict_1314.astype(np.int32)})
result_1314.set_value(result_1314.subsidy==1,'subsidy',0)


###############  先回到未处理过的数据集   ################
f1314_traind  = f1314_train.copy()
f1415_traind = f1415_train.copy()
f1314_testc  = f1314_test.copy()
f1415_testc  = f1415_test.copy()
################   1314 非0类再分类  单独用rfc训练一个模型进行预测  ###############
f1314_n0 = f1314_traind.loc[f1314_traind.subsidy!=0] #527
################   扩充样本   ###################
f1314_n0_2000 = f1314_n0.loc[f1314_n0.subsidy==2000]
f1314_n0_1500 = f1314_n0.loc[f1314_n0.subsidy==1500]
f1314_n0_1000 = f1314_n0.loc[f1314_n0.subsidy==1000]
for i in range(2):
    f1314_n0 = f1314_n0.append(f1314_n0_2000)
for i in range(1):
    f1314_n0 = f1314_n0.append(f1314_n0_1500)
# for i in range(3):
#     f1314_n0 = f1314_n0.append(f1314_n0_1000)
### 加入随机扰动##########
f1314_n0['sum_total']=f1314_n0.sum_total.values + [round(1*np.random.randn(),2) for i in range(len(f1314_n0))]

#####  进行特征选择#####
predictors = [x for x in f1314_traind.columns if x not in ['subsidy','year','sum_checkget','num_checkget']]##
sel_1314_train = f1314_n0[predictors]
sel_1314_test = f1314_testc[predictors]
######  准备训练集  ##################
x_1314_train = np.array(sel_1314_train)
y_1314_train = np.array(f1314_n0['subsidy'])
x_1314_test = np.array(sel_1314_test)  # 真正用于测试的数据集还需要根据上一步训练进行筛选

##############################################    RFC 1314年 非0类 训练及预测  #####################
X1_train, X1_test, Y1_train, Y1_test = train_test_split(x_1314_train, y_1314_train, test_size=0.3,random_state=0)
# X1_train, X1_test, Y1_train, Y1_test = train_test_split(sel_1314_train, y_1314_train, test_size=0.3,random_state=0)
rfc = RandomForestClassifier(criterion='entropy', max_features='auto' ,\
                             bootstrap=True, oob_score=False, n_jobs=1,class_weight = 'balanced_subsample', random_state=0, verbose=0)
tuned_parameters = {'n_estimators': [400], 'max_depth':  [11],'max_features':['auto'],'min_samples_leaf':[2]}
grid_search1 = GridSearchCV(rfc, param_grid=tuned_parameters, verbose=3, n_jobs=-1,scoring='f1_macro',\
              cv=ShuffleSplit(len(Y1_train) ,n_iter=5, test_size=0.3,random_state=0)).fit(X1_train, Y1_train)
#### 1314 非0类的预测和处理######
index = result_1314.loc[result_1314.subsidy!=0].index
predict_1314 = grid_search1.best_estimator_.predict(x_1314_test[index])
result1314_n0= pd.DataFrame({'ID':result_1314.loc[result_1314.subsidy!=0].ID.values, 'subsidy':predict_1314.astype(np.int32)})
result1314_0 = result_1314.loc[result_1314.subsidy==0]
result1314_n0.subsidy.value_counts()
###############  先回到未处理过的数据集   ################
f1314_traind  = f1314_train.copy()
f1415_traind = f1415_train.copy()
f1314_testc  = f1314_test.copy()
f1415_testc  = f1415_test.copy()
# ####    1415年训练集分为0类和非0类，并把subsidy标签改为1和-1   #########
f1415_traind.set_value(f1415_traind.subsidy==0,'subsidy',1)
f1415_traind.set_value(f1415_traind.subsidy!=1,'subsidy',-1)
#####   扩充样本   #####################
f1415_traind_n0 = f1415_traind.loc[f1415_traind.subsidy==-1]
for i in range(1):
    f1415_traind = f1415_traind.append(f1415_traind_n0)

### 加入随机扰动##########

f1415_traind['sum_total'] = f1415_traind.sum_total.values + [round(1 * np.random.randn(), 2) for i in
                                                             range(len(f1415_traind))]
#####  1415进行特征选择#####
predictors = [x for x in f1415_traind.columns if
              x not in ['subsidy', 'year', 'sum_deposit', 'sum_checkget', 'sum_total', 'single_POS', 'num_checkget',
                        'num_lost', 'single_library', 'sum_food']]  ##
sel_1415_train = f1415_traind[predictors]
sel_1415_test = f1415_testc[predictors]
##  准备训练集######
x_1415_train = np.array(sel_1415_train)
x_1415_test = np.array(sel_1415_test)
y_1415_train = np.array(f1415_traind['subsidy'])
##############################################    RFC  1415年  0和非0   训练及预测 #####################
X2_train, X2_test, Y2_train, Y2_test = train_test_split(x_1415_train, y_1415_train, test_size=0.3, random_state=0)
rfc = RandomForestClassifier(criterion='entropy', max_features='auto', \
                             bootstrap=True, oob_score=False, n_jobs=-1, class_weight='balanced_subsample',
                             random_state=0, verbose=0)
# tuned_parameters = {'n_estimators': [100,150,200], 'max_depth':  [4,7,10,15],'max_features':['auto'],'min_samples_leaf':[2,4,5,7]}
tuned_parameters = {'criterion': ['entropy'], 'n_estimators': [250], 'max_depth': [11], 'max_features': ['auto'],
                    'min_samples_leaf': [4]}
grid_search2 = GridSearchCV(rfc, param_grid=tuned_parameters, verbose=0, n_jobs=1, scoring='f1_macro', \
                            cv=ShuffleSplit(len(Y2_train), n_iter=5, test_size=0.3, random_state=0)).fit(X2_train,
                                                                                                         Y2_train)

#### 1415  0和非0类的预测及处理#####
predict_1415 = grid_search2.best_estimator_.predict(x_1415_test)
result_1415 = pd.DataFrame({'ID': f1415_test['ID'].as_matrix(), 'subsidy': predict_1415.astype(np.int32)})
result_1415.set_value(result_1415.subsidy == 1, 'subsidy', 0)


###############  先回到未处理过的数据集   ################
f1314_traind  = f1314_train.copy()
f1415_traind = f1415_train.copy()
f1314_testc  = f1314_test.copy()
f1415_testc  = f1415_test.copy()
################   1415 非0类再分类  单独用rfc训练一个模型进行预测  ###############
f1415_n0 = f1415_traind.loc[f1415_traind.subsidy!=0] #527

### 加入不敏感随机扰动##########
f1415_n0['sum_total']=f1415_n0.sum_total.values + [round(1*np.random.randn(),2) for i in range(len(f1415_n0))]

####  进行特征选择#####
predictors = [x for x in f1415_traind.columns if x not in ['subsidy','year','sum_laundry','num_laundry']]##
sel_1415_train = f1415_n0[predictors]
sel_1415_test = f1415_testc[predictors]
######  准备训练集  ##################
x_1415_train = np.array(sel_1415_train)
y_1415_train = np.array(f1415_n0['subsidy'])
x_1415_test = np.array(sel_1415_test)  # 真正用于测试的数据集还需要根据上一步训练进行筛选
##############################################    RFC  1415年  非0类  训练及预测 #####################
X2_train, X2_test, Y2_train, Y2_test = train_test_split(x_1415_train, y_1415_train, test_size=0.3,random_state=0)
rfc = RandomForestClassifier(criterion='entropy', max_features='auto' ,\
                             bootstrap=True, oob_score=False, n_jobs=-1,class_weight = 'balanced_subsample', random_state=0, verbose=0)
# tuned_parameters = {'n_estimators': [100,150,200], 'max_depth':  [4,7,10,15],'max_features':['auto'],'min_samples_leaf':[2,4,5,7]}
tuned_parameters = {'criterion':['entropy'] , 'n_estimators': [300], 'max_depth':  [11],'max_features':['auto'],'min_samples_leaf':[2]}
grid_search2 = GridSearchCV(rfc, param_grid=tuned_parameters, verbose=0, n_jobs=1,scoring='f1_macro',\
              cv=ShuffleSplit(len(Y2_train) ,n_iter=5, test_size=0.3,random_state=0)).fit(X2_train, Y2_train)

#### 1415 非0类的预测和处理######
index = result_1415.loc[result_1415.subsidy!=0].index
predict_1415 = grid_search2.best_estimator_.predict(x_1415_test[index])
result1415_n0= pd.DataFrame({'ID':result_1415.loc[result_1415.subsidy!=0].ID.values, 'subsidy':predict_1415.astype(np.int32)})
result1415_0 = result_1415.loc[result_1415.subsidy==0]
result1415_n0['subsidy'].value_counts()

result1 = pd.concat([result1415_n0,result1415_0,result1314_n0,result1314_0],join='outer',axis=0,ignore_index=True)
result1.sort_index(by='ID',inplace=True)
result1 = result1.rename(columns={'ID':'studentid'})
result1.to_csv('result1.csv',index=False)