from __future__ import print_function
import numpy as np
import pandas as pd
np.random.seed(1337) # for reproducibility
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold

# load user modules
import models

def StackModels(train, test, y, clfs, n_folds, scaler=None): # train data (pd data frame), test data (pd date frame), Target data,
                                                # list of models to stack, number of folders, boolean for scaling
# StackModels() performs Stacked Aggregation on data: it uses n different classifiers to get out-of-fold 
# predictions for target data. It uses the whole training dataset to obtain signal predictions for test.
# This procedure adds n meta-features to both train and test data (where n is number of models to stack).

    print("Generating Meta-features")
    num_class = np.unique(y).shape[0]
    skf = list(StratifiedKFold(y, n_folds))
    # print(skf)
    if scaler:
        scaler = preprocessing.StandardScaler().fit(train)
        train_sc = scaler.transform(train)
        test_sc = scaler.transform(test)
    else:
        train_sc = train
        test_sc = test
    blend_train = np.zeros((train.shape[0], num_class*len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((test.shape[0], num_class*len(clfs)))   # Number of testing data x Number of classifiers   
    for j, clf in enumerate(clfs):
        print ('Training classifier [%s]' % (j))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = train.iloc[tr_index,:]
            Y_train = y[tr_index]
            X_cv = train.iloc[cv_index,:]
            if scaler:
               scaler_cv = preprocessing.StandardScaler().fit(X_train)
               X_train=scaler_cv.transform(X_train)
               X_cv=scaler_cv.transform(X_cv)

            clf.fit(X_train, Y_train)
            pred = clf.predict_proba(X_cv)
            blend_train[cv_index, j*num_class:(j+1)*num_class] = pred
                
        print('stacking test data')        
        clf.fit(train_sc, y)
        pred = clf.predict_proba(test_sc)

        blend_test[:, j*num_class:(j+1)*num_class] = pred

    return blend_train, blend_test 

def Trees_stacking(train, test, y):
    
    
    clf1=models.XGBoost_multilabel(nthread=2, eta=0.08 ,gamma=0.1, max_depth=15,
                           min_child_weight=2,
                           max_delta_step=None,
                           subsample=0.7, colsample_bytree=0.3,
                           silent=1, seed=1301,
                           l2_reg=1.8, l1_reg=0.15, num_round=300)
                          
    clf2=RandomForestClassifier(n_estimators=250, criterion='entropy', max_depth=15, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.6,
                            max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=2,
                            random_state=1301, verbose=0)
                          
    clf3=ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_depth=15,
                             min_samples_split=2, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features=0.5,
                             max_leaf_nodes=None, bootstrap=False, oob_score=False,
                             n_jobs=2, random_state=1301, verbose=0)  
                          
    clf4 = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features=0.4, random_state=1301),
                             n_estimators=300, learning_rate=0.07, random_state=1301)                           
    
    clfs = [clf1, clf2, clf3, clf4]
    train_probs, test_probs = StackModels(train, test, y, clfs, 3)  #n_folds=3
    
    return train_probs, test_probs

# read file
train_test = pd.read_csv('../input/feature.csv')

train = train_test[train_test['money'].notnull()]
train['money'] = train['money'].replace([1000,1500,2000],[1,2,3])
test = train_test[train_test['money'].isnull()]

train = train.fillna(0)
test = test.fillna(0)

target = 'money'
IDcol = 'id'
ids = test['id'].values
predictors = [x for x in train.columns if x not in [target]]
y = train[target]

train_raw = train[predictors]
test_raw = test[predictors]

# print(train_raw.shape)
# print(test_raw.shape)

meta_trees_train, meta_trees_test = Trees_stacking(train_raw, test_raw, y)
print(meta_trees_train)
print(meta_trees_test)

clf_xgb = models.XGBoost_multilabel(nthread=6, eta=0.016 ,gamma=1, max_depth=9,
                           min_child_weight=11,
                           max_delta_step=None,
                           subsample=1, colsample_bytree=0.75,
                           silent=1, seed=1301,
                           l2_reg=3, l1_reg=0.2, num_round=800)  
clf_xgb.fit(meta_trees_train, y)
preds_xgb = clf_xgb.predict_proba(meta_trees_test)

preds_subm = pd.DataFrame(preds_xgb)
preds_subm.columns = ['s0','s1000','s1500','s2000']
preds_subm['studentid'] = ids
print('Xgb done!')

preds_subm.to_csv('../temp/r1.csv',index=False)
result = pd.read_csv('../temp/r1.csv')

list0 = result[['studentid']]
list0['subsidy'] = 0

result = result.sort_values(by='s1000',ascending=False)
list1000 = pd.DataFrame(result.head(2250).studentid)
list1000['subsidy'] = 1000

result = result.sort_values(by='s1500',ascending=False)
list1500 = pd.DataFrame(result.head(900).studentid)
list1500['subsidy'] = 1500

result = result.sort_values(by='s2000',ascending=False)
list2000 = pd.DataFrame(result.head(600).studentid)
list2000['subsidy'] = 2000

result = pd.concat([list2000,list1500,list1000,list0])
result.index = range(len(result))
result = result.drop_duplicates(['studentid'])
result = result.sort_values('studentid')

print('1000--'+str(len(result[result.subsidy==1000])) + ':741')
print('1500--'+str(len(result[result.subsidy==1500])) + ':465')
print('2000--'+str(len(result[result.subsidy==2000])) + ':354')
result.to_csv('../output/stacking1.csv',index=False)

