import numpy as np
import pandas as pd

#定义ks评分函数
def KS(y_true,y_pred):
    #对测评数据整形排序
    data  = pd.DataFrame({'true':list(y_true),'pred':list(y_pred)})
    data.sort_values('pred',inplace=True)

    count_of_0,count_of_1 = data['true'].value_counts()
    score = []
    num0 = 0.
    num1 = 0.
    for true_value in data['true'].values:
        if true_value:
            num1 += 1
        else:
            num0 += 1
        score.append(abs(num0/count_of_0-num1/count_of_1))

    return np.max(score)

def CV(train,predict,n=5):
    # 分割训练集
    def split(data, split_n=5, random_state=None):
        N = len(data)
        row = np.random.permutation(N)
        n = N / split_n
        train = data.iloc[row[:-n]]
        test = data.iloc[row[-n:]]

        return train, test

    # 线下评分程序
    def score(y_true, y_pred):
        # 对测评数据整形排序
        data = pd.DataFrame({'true': list(y_true), 'pred': list(y_pred)})
        data.sort_values('pred', inplace=True)

        count_of_0, count_of_1 = data['true'].value_counts()
        score = []
        num0 = 0.
        num1 = 0.
        for true_value in data['true'].values:
            if true_value:
                num1 += 1
            else:
                num0 += 1
            score.append(abs(num0 / count_of_0 - num1 / count_of_1))

        return np.max(score)

    output = []
    for x in range(n):
        train_sub,test_sub = split(train,split_n=5)
        result = predict(train_sub,test_sub)
        s = score(test_sub['overdue'],result['probability'])
        output.append(s)

    return output