import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def predict_1(train,test):

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'overdue'
    IDcol = 'userid'
    predictors = [x for x in train.columns if x not in [target,IDcol]]

    # 这里用Logistic回归
    lr_model = LogisticRegression(C = 1.0, penalty = 'l2')
    lr_model.fit(train[predictors], train[target])
    pred = lr_model.predict_proba(test[predictors])

    result = pd.DataFrame({'userid':test['userid'].values,'probability':pred[:,1]})
    result.sort_index(ascending=False,axis=1,inplace=True)
    return result