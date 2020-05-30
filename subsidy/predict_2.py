import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

#助学金预测代码
def predict_2(train,test,alg=XGBClassifier,coefficient=1,silent=1):


    # F1预测分数
    def predict_f1(proba, n, proportion):
        proba = sorted(proba,reverse=True)
        TP = sum(proba[0:n])  # 预测为a类且正确的数量
        MP = int(len(proba) * proportion)  # a类实际的数量
        MN = n  # 预测为a类的数量
        return 2 * TP / (MP + MN)

    #判断非零的个数
    def count1000(result):
        result = sorted(result,reverse=True)
        TP = 0.
        MN = 0
        MP = 0.1432*len(result)
        for x in result:
            TP = TP+x
            MN = MN+1
            if (x<(TP/(MP+MN))):
                break

        return MN


    # 判断2000的个数
    def count2000(result):
        count = len(result)
        score = 0
        n = 0
        for x in xrange(count):
            temp = predict_f1(result[:,0], x, 0.0427)+predict_f1(result[:,1], count-x, 0.0325)
            if score > temp:
                n = count-x+1
                break
            score = temp

        return n


    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'subsidy'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    test_1 = test.copy()
    train_1['subsidy'] = train_1['subsidy'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_2= train.copy()
    test_2 = test.copy()
    train_2 = train_2[train['subsidy']>0]
    train_2['subsidy'] = train_2['subsidy'].map({0: 0, 1000: 0, 1500: 1, 2000: 1})

    train_3 = train.copy()
    test_3 = test.copy()
    train_3 = train_3[train['subsidy']>1000]
    train_3['subsidy'] = train_3['subsidy'].map({0: 0, 1000: 0, 1500: 0, 2000: 1})


    # model of select id which big 0
    alg1 = alg()
    alg1.fit(train_1[predictors], train_1[target])
    result1 = alg1.predict_proba(test_1[predictors])
    n1000 = int(count1000(result1[:,1])*coefficient)
    if silent==0 : print '大于0的个数为：', n1000
    result1 = pd.DataFrame({'studentid': test_1['studentid'].values, 1: result1[:, 1]}).sort_values(by=[1],ascending=False)
    test_id_1000 = list(result1.head(n1000)['studentid'])

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test_2['studentid']]
    test_2 = test_2[test_2_id]
    alg2 = alg()
    alg2.fit(train_2[predictors], train_2[target])
    result2 = alg2.predict_proba(test_2[predictors])
    n1500 = int(len(test)*0.096*coefficient)
    if silent==0: print '大于1000的个数为：',n1500
    result2 = pd.DataFrame({'studentid': test_2['studentid'].values, 1: result2[:, 1]}).sort_values(by=[1],ascending=False)
    test_id_1500 = list(result2.head(n1500)['studentid'])

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test_3['studentid']]
    test_3 = test_3[test_3_id]
    alg3 = alg()
    alg3.fit(train_3[predictors], train_3[target])
    result3 = alg3.predict_proba(test_3[predictors])
    n2000 = int(count2000(result3)*coefficient)
    if silent==0: print '2000的个数：', n2000
    result3 = pd.DataFrame({'studentid': test_3['studentid'].values, 1: result3[:, 1]}).sort_values(by=[1],ascending=False)
    test_id_2000 = list(result3.head(n2000)['studentid'])

    #将id对应的助学金合并起来
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

    return pd.DataFrame({'studentid':test['studentid'],'subsidy':subsidy})



