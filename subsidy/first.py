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

train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\output\people_info_train.csv')
test = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\output\people_info_test.csv')

train.columns = ['studentid','consume','consume_meal','count_meal','eva_meal','percent_meal','college','people_college','order_true','order','money']
test.columns = ['studentid','consume','consume_meal','count_meal','eva_meal','percent_meal','college','people_college','order_true','order']

#对train的行进行重新排序
train.index = np.random.permutation(10885)
train = train.sort_index(axis=0)

train_1 = train.copy()
test_1 = test.copy()
train_1 = train_1[train_1['studentid'] < 22073]
test_1 = test_1[test_1['studentid'] < 22073]


train_1['money'] = train_1['money'].map({0:0, 1000:1, 1500:1, 2000:1})

train_1 = train_1.fillna(-1)
test_1 = test_1.fillna(-1)

target = 'money'
predictors = [x for x in train.columns if x not in ['id','people_college','order_true','money']]


# model
clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)
clf = clf.fit(train_1[predictors],train_1[target])
result = clf.predict_proba(test_1[predictors])

test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = test_1['id'].values
test_result.subsidy = result[:,1]
test_result = pd.merge(left=test[['id']], right=test_result, how='left', by='studentid').fillna(0)
test_result.sort(['subsidy'],ascending=False)

#添加label的列，大于22073为1，小于为零
test_1['label'] = (test_1['studentid']>22073).map({False:0,True:1})

select = test_result.head(2653)['id'].values


# 输入数据，用简单的模型做预测
def predict_1(train, test):
    train_1 = train.copy()
    test_1 = test.copy()
    train_1['money'] = train_1['money'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_1 = train_1.fillna(-1)
    test_1 = test_1.fillna(-1)
    target = 'money'
    IDcol = 'studentid'
    predictors = [x for x in train_1.columns if x not in [target]]

    # model
    clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf.fit(train_1[predictors], train_1[target])
    result1 = clf.predict_proba(test_1[predictors])
    result1 = pd.DataFrame({'studentid':test_1['studentid'].values,'proba_of_1':result1[:,1]}).sort(['proba_of_1'],ascending=False)

    select_id = [id in result1.head(2800)['studentid'].values for id in test_1['studentid']]
    clf2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf2.fit(train_1[train['money'] != 0][predictors], train[train_1['money'] != 0][target])
    result2 = clf2.predict(test_1[select_id][predictors])
    result2 = pd.DataFrame({'studentid': test_1[select_id]['studentid'].values, 'subsidy': result2})

    result = pd.merge(result1[['studentid']],result2,how='left',on='studentid').fillna(0)
    print result.groupby(result['subsidy']).size()
    return result

#最佳映射
def mapping(data):
    #F1预测分数
    def predict_f1(proba,isin,proportion):
        TP = proba[list(isin)].sum()    #预测为a类且正确的数量
        MP = int(proba.size*proportion)    #a类实际的数量
        MN = proba[list(isin)].size     #预测为a类的数量
        return 2*TP/(MP+MN)

    #DC评分函数
    def predict_DC(proba_data,prodict):
        #1000的f1值
        score1000 = predict_f1(proba_data[1000], (prodict == 1000), 0.068)
        score1500 = predict_f1(proba_data[1500], (prodict == 1500), 0.0427)
        score2000 = predict_f1(proba_data[2000], (prodict == 2000), 0.0325)
        return 0.068*score1000+0.0427*score1500+0.0325*score2000

    #判断改变预测值后，评分是否有提高，若提高则改变，否则不改变
    def if_improved(id,value):
        prodict_copy = prodict.copy()
        prodict_copy[id] = value
        score2 = predict_DC(data,prodict_copy)
        return (score1,prodict) if score2<score1 else (score2,prodict_copy)

    prodict = pd.Series(0, name='subsidy', index=data.index)
    score1 = predict_DC(data, prodict)
    for column in [0,1000,1500,2000,0,1000,1500,0,1000]:
        proba_col = data[column].copy()
        proba_col.sort_values(inplace=True,ascending=False)
        for index in proba_col.index:
            if prodict[index] == proba_col.name:
                continue
            score1,prodict = if_improved(index, proba_col.name)
        print 'score1:',score1

    print prodict.value_counts()
    return prodict


#输出预测的概率函数
# 输入数据，用简单的模型做预测(效果一般)
def predict_2(train, test):
    train_1 = train.copy()
    test_1 = test.copy()
    train_1['money'] = train_1['money'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_1 = train_1.fillna(-1)
    test_1 = test_1.fillna(-1)
    target = 'money'
    IDcol = 'studentid'
    predictors = [x for x in train_1.columns if x not in [target]]

    # model
    clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200)
    clf.fit(train_1[predictors], train_1[target])
    result1 = clf.predict_proba(test_1[predictors])

    clf2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200)
    clf2.fit(train_1[train['money'] != 0][predictors], train[train_1['money'] != 0][target])
    result2 = clf2.predict_proba(test_1[predictors])
    result2 = result2*result1[:,1].reshape(len(result1),1)

    result = pd.DataFrame({'studentid':test_1['studentid'] , 0:result1[:,0] , 1000:result2[:,0] , 1500:result2[:,1] , 2000:result2[:,2]})
    result.set_index('studentid',inplace=True)
    return result
output1 = mapping1(data1)
output2 = mapping2(data2)
output3 = mapping2(data3)

# 输入单个特征和标签，对特征进行分析
def analyse(data,feature,label,n=10,delect=None):
    # 删除无用数据
    data = data[data[feature] != delect]
    data = data[data[feature].notnull()]

    #对数据分类
    factor = pd.cut(data[feature], n)
    result = data.groupby(factor).apply(lambda group : group[label].value_counts())
    s = result.apply(lambda x: x.sum(),axis=1)
    p = result.icol(1)/s
    result['sum'] = s
    result['persent'] = p
    result.fillna(0, inplace=True)

    #绘图
    p.plot()

    return result

for column in train_1.columns:
    analyse(train_1,column,'money',n=20,delect=-1)


#模型预测2
def predict_2(train, test):
    train_1 = train.copy()
    test_1 = test.copy()
    train_1['money'] = train_1['money'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_1 = train_1.fillna(-1)
    test_1 = test_1.fillna(-1)
    target = 'money'
    IDcol = 'studentid'
    predictors = [x for x in train_1.columns if x not in [target]]

    # model
    clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf.fit(train_1[predictors], train_1[target])
    result1 = clf.predict_proba(test_1[predictors])

    clf2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60, max_depth=3,min_samples_leaf=6,min_samples_split=150,subsample=1,random_state=30)
    clf2.fit(train[train['money'] != 0][predictors], train[train['money'] != 0][target])
    result2 = clf2.predict_proba(test[predictors])

    result3 = result2*(result1[:,1].reshape(len(result),1))
    result3 = pd.DataFrame({'studentid':test_1['studentid'],0:result1[:,0],1000:result3[:,0],1500:result3[:,1],2000:result3[:,2]})
    result3.set_index('studentid',inplace=True)
    return result3


#当期望得分最高时，判断subsidy为非零的个数
def predict_2(train,test,coefficient=1):


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
    target = 'money'
    IDcol = 'studentid'
    predictors = [x for x in train.columns if x not in [target]]

    train_1 = train.copy()
    test_1 = test.copy()
    train_1['money'] = train_1['money'].map({0: 0, 1000: 1, 1500: 1, 2000: 1})

    train_2= train.copy()
    test_2 = test.copy()
    trian_2 = train_2[train['money']>0]
    train_2['money'] = train_2['money'].map({0: 0, 1000: 0, 1500: 1, 2000: 1})

    train_3 = train.copy()
    test_3 = test.copy()
    train_3 = train_3[train['money']>1000]
    train_3['money'] = train_3['money'].map({0: 0, 1000: 0, 1500: 0, 2000: 1})


    # model of select id which big 0
    clf1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=3, min_samples_leaf=6,
                                 min_samples_split=150, subsample=1, random_state=30)
    clf1.fit(train_1[predictors], train_1[target])
    result1 = clf1.predict_proba(test_1[predictors])
    n1000 = int(count1000(result1[:,1])*coefficient)
    print '大于0的个数为：', n1000
    result1 = pd.DataFrame({'studentid': test_1['studentid'].values, 1: result1[:, 1]}).sort_values(by=[1],ascending=False)
    test_id_1000 = list(result1.head(n1000)['studentid'])

    # model of select id which big 1000
    test_2_id = [i in test_id_1000 for i in test_2['studentid']]
    test_2 = test_2[test_2_id]
    clf2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=3, min_samples_leaf=6,
                                 min_samples_split=150, subsample=1, random_state=30)
    clf2.fit(train_2[predictors], train_2[target])
    result2 = clf2.predict_proba(test_2[predictors])
    n1500 = int(len(test)*0.096*coefficient)
    print '大于1000的个数为：',n1500
    result2 = pd.DataFrame({'studentid': test_2['studentid'].values, 1: result2[:, 1]}).sort_values(by=[1],ascending=False)
    test_id_1500 = list(result2.head(n1500)['studentid'])

    # model of select id which big 1000
    test_3_id = [i in test_id_1500 for i in test_3['studentid']]
    test_3 = test_3[test_3_id]
    clf3 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=3, min_samples_leaf=6,
                                 min_samples_split=150, subsample=1, random_state=30)
    clf3.fit(train_3[predictors], train_3[target])
    result3 = clf3.predict_proba(test_3[predictors])
    n2000 = int(count2000(result3)*coefficient)
    print '2000的个数：', n2000
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

#线下评分程序
def score(y_true,y_pred):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    count = y_true.value_counts()

    score1000 = f1_score(y_true == 1000, y_pred == 1000) * count[1000] / y_true.size
    score1500 = f1_score(y_true == 1500, y_pred == 1500) * count[1500] / y_true.size
    score2000 = f1_score(y_true == 2000, y_pred == 2000) * count[2000] / y_true.size

    return score1000+score1500+score2000

#分割训练集
def split(data,split=3,random_state=None):
    N = len(data)
    row = np.random.permutation(N)
    n = N/split
    train = data.irow(row[:-n])
    test = data.irow(row[-n:])

    return train,test

#交叉验证
def CV(train,predict,n=5):
    # 分割训练集
    def split(data, split_n=3, random_state=None):
        N = len(data)
        row = np.random.permutation(N)
        n = N / split_n
        train = data.iloc[row[:-n]]
        test = data.iloc[row[-n:]]

        return train, test

    # 线下评分程序
    def score(y_true, y_pred):
        y_true = pd.Series(list(y_true))
        y_pred = pd.Series(list(y_pred))
        count = y_true.value_counts()

        score1000 = f1_score(y_true == 1000, y_pred == 1000) * count[1000] / y_true.size
        score1500 = f1_score(y_true == 1500, y_pred == 1500) * count[1500] / y_true.size
        score2000 = f1_score(y_true == 2000, y_pred == 2000) * count[2000] / y_true.size

        return score1000 + score1500 + score2000

    output = []
    for x in range(n):
        train_sub,test_sub = split(train,split_n=3)
        result = predict(train_sub,test_sub)
        s = score(test_sub['subsidy'],result['subsidy'])
        output.append(s)

    return output


'''
助学金获得奖学金按数量分类器
data.columns = [0,1000,1500,2000]
data.index = 'studentid'
'''
def count_classify(data,p0=0.87,p1000=0.14,p1500=0.06,p2000=0.03):

    sum_of_p  = float(p0+p1000+p1500+p2000)
    sum_of_n = len(data)
    p0 = p0 / sum_of_p
    p1000 = p1000 / sum_of_p
    p1500 = p1500 / sum_of_p
    p2000 = p2000 / sum_of_p

    count1000 = int(sum_of_n * p1000)
    count1500 = int(sum_of_n * p1500)
    count2000 = int(sum_of_n * p2000)
    count_n = pd.Series({1000:count1000, 1500:count1500, 2000:count2000})
    print '规定数量：\n',count_n

    value_count = data[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1).value_counts()
    #print value_count
    coefficient = pd.Series({1000:0.2, 1500:0.2, 2000:0.2})
    while True:
        value_count_temp = value_count
        for i in [1000,1500,2000]:
            if value_count[i] > count_n[i]:
                if coefficient[i] > 0:
                    coefficient[i] = coefficient[i] * (-0.5)
            elif value_count[i] < count_n[i]:
                if coefficient[i] < 0:
                    coefficient[i] = coefficient[i] * (-0.5)
            else:
                continue
            data[i] = data[i] * (1+coefficient[i])

        #选择概率最大值作为分类
        value_count = data[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1).value_counts()
        #print value_count
        if value_count.equals(value_count_temp):
            break
    print '最后数量：\n',value_count
    return data[[0, 1000, 1500, 2000]].apply(lambda x: x.argmax(), axis=1)

#已知第一年求第二年 的先验概率矩阵 和 后验概率矩阵
def get_proba_matrix(trian):
    train_temp = train[train['studentid']<22072][['studentid','subsidy']].copy()
    train_temp['year'] = train_temp['studentid']%2
    train_temp['id'] = train_temp['studentid']//2
    train_1 = train_temp[train_temp['year']==0]
    train_2 = train_temp[train_temp['year']==1]
    train_12 = pd.merge(train_1,train_2,on='id',how='inner',   suffixes=('_1','_2'))
    matrix = train_12.groupby(['subsidy_x','subsidy_y']).size().unstack()
    prior = matrix.divide(matrix.sum(axis=1),axis=0).T
    posterior = matrix.divide(matrix.sum(axis=0),axis=1)
    return prior,posterior


#双模型交叉验证
def CV2(train,feature_1,feature_2,predict_1,predict_2,n=5):
    # 分割训练集
    def split(data, split_n=3, random_state=None):
        N = len(data)
        row = np.random.permutation(N)
        n = N / split_n
        train = data.iloc[row[:-n]]
        test = data.iloc[row[-n:]]

        return train, test

    # 线下评分程序
    def score(y_true, y_pred):
        y_true = pd.Series(list(y_true))
        y_pred = pd.Series(list(y_pred))
        count = y_true.value_counts()

        score1000 = f1_score(y_true == 1000, y_pred == 1000) * count[1000] / y_true.size
        score1500 = f1_score(y_true == 1500, y_pred == 1500) * count[1500] / y_true.size
        score2000 = f1_score(y_true == 2000, y_pred == 2000) * count[2000] / y_true.size

        return score1000 + score1500 + score2000

    def mapping(data):
        n = len(data)

        # 筛选出大于0的id
        data1 = data.copy()
        data1.sort_values(['>0'], inplace=True, ascending=False)
        test_id_1000 = list(data1['studentid'].head(int(n * 0.23)).values)

        # 筛选出大于1000的id
        data2 = data1[[idx in test_id_1000 for idx in list(data1['studentid'].values)]].copy()
        data2.sort_values(['>1000'], inplace=True, ascending=False)
        test_id_1500 = list(data2['studentid'].head(int(n * 0.09)).values)

        # 筛选出大于1500的id
        data3 = data2[[idx in test_id_1500 for idx in list(data2['studentid'].values)]].copy()
        data3.sort_values(['>1500'], inplace=True, ascending=False)
        test_id_2000 = list(data3['studentid'].head(int(n * 0.03)).values)

        # 将id对应的助学金合并起来
        subsidy = []
        for x in data['studentid']:
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

        return pd.DataFrame({'studentid': list(data.index), 'subsidy': subsidy})

    output = []
    for x in range(n):
        train_sub,test_sub = split(train,split_n=3)
        result1 = predict_1(train_sub[feature_1], test_sub[feature_1])
        result2 = predict_2(train_sub[feature_2], test_sub[feature_2])
        result = (result1+result2)/2.
        result = mapping(result)
        s = score(test_sub['subsidy'], result['subsidy'])
        output.append(s)

    return output


#调参
def for_gsearch(data,gsearch,n=1):
    target = 'subsidy'
    predictors = [x for x in data.columns if x not in [target]]

    grid_scores_ = []
    for i in range(n):
        data.index = np.random.permutation(len(train.index))
        data = data.sort_index(axis=0)
        gsearch.fit(data[predictors], data[target])
        if i==0:                                                #第一次添加
            for j in range(len(gsearch.grid_scores_)):
                grid_scores_.append([gsearch.grid_scores_[j][0],list(gsearch.grid_scores_[j][2])])
        else:
            for j in range(len(gsearch.grid_scores_)):
                grid_scores_[j][1].extend(list(gsearch.grid_scores_[j][2]))

    result = []
    for line in grid_scores_:
        result.append('mean : %.4f, std : %.4f, params : {'%(np.mean(line[1]),np.std(line[1]))+line[0].keys()[0]+':'+str(line[0].values()[0])+'}')

    return result


        #gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_



# 模型整天调参
def for_cv(train,predict,params,n=5):

    # 分割训练集
    def split(data, split_n=3, random_state=None):
        N = len(data)
        row = np.random.permutation(N)
        n = N / split_n
        train1 = data.iloc[row[:-n]]
        test1 = data.iloc[row[-n:]]
        train2 = data.iloc[row[n:]]
        test2 = data.iloc[row[:n]]
        train3 = pd.concat([data.iloc[row[:n]],data.iloc[row[-n:]]])
        test3 = data.iloc[row[n:-n]]

        return ((train1, test1), (train2, test2), (train3, test3))

    # 线下评分程序
    def score(y_true, y_pred):
        y_true = pd.Series(list(y_true))
        y_pred = pd.Series(list(y_pred))
        count = y_true.value_counts()

        score1000 = f1_score(y_true == 1000, y_pred == 1000) * count[1000] / y_true.size
        score1500 = f1_score(y_true == 1500, y_pred == 1500) * count[1500] / y_true.size
        score2000 = f1_score(y_true == 2000, y_pred == 2000) * count[2000] / y_true.size

        return score1000 + score1500 + score2000

    output = []
    for param in params:
        output.append([])
    for x in range(n):
        data_split = split(train,split_n=3)
        for i in range(len(params)):
            for train_sub,test_sub in data_split:
                result = predict(train_sub,test_sub,params[i])
                s = score(test_sub['subsidy'],result['subsidy'])
                output[i].append(s)

    return output

#对id进行加噪声
def noise(li,n):
    l1 = np.random.permutation(len(li))
    l2 = np.random.permutation(len(li))
    for i in range(n):
        li[l1[i]] = li[l2[i]]
    return li

#改建KFlod
def split(y, n_splits=3, random_state=None):
    n_sub = int(y/n_splits)
    row = np.random.permutation(y)



    return train, test