# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from matplotlib.pylab import rcParams
import os



#------------------------测评函数-------------------------#
def score(y_true,y_pred):

    y_true = pd.DataFrame(y_true).fillna(0)
    y_pred = pd.DataFrame(y_pred).fillna(0)
    y_pred.columns = y_true.columns
    y_pred.index = y_true.index

    y_true.sort_index(ascending=True,inplace=True)
    y_pred.sort_index(ascending=True,inplace=True)
    if y_pred[y_pred<0].sum().sum()<0:
        print '预测结果出现负数!请重新检查你的结果！'
    y_pred = y_pred - y_pred[y_pred < 0].fillna(0)
    shape = y_true.shape
    dif = y_true-y_pred
    summation = (y_true+y_pred).replace(0,np.nan)

    return dif.divide(summation).sum().sum()/(shape[0]*shape[1])




def get_xy(data,day,sep=7):
    week = day%7
    day_x = range(day-week-sep,day-week)
    result_x = data[day_x]
    result_y = data[day]

    return result_x,result_y



#客流量异常店铺检测,输入的天数大于14天
def abnormal(data,valve=0.25):
    data = pd.DataFrame(data)
    result1 = []
    result2 = []
    for i,row in data.iterrows():
        for j in range(len(row)-13):
            sum1 = sum(row[j:j+8])
            sum2 = sum(row[j+8:j+15])
            if abs(sum1-sum2)/(sum1+sum2)>valve:
                result1.append(i)
                result_list = []
                for k in [8,15,22]:
                    result_list.append(False if sum(row[k-8:k])/sum(row)<0.25 else True)
                result2.append(result_list)
                break
    print '异常店铺个数：' + str(len(result1))
    result = pd.DataFrame(result2,index=result1)

    return result


#检测提交结果是否异常
def inspect(url):
    import pandas as pd
    import numpy as np

    data = pd.read_csv(url,header=None)
    result = data
    flag = True
    if data.shape != (2000,15):
        print '数据缺失或多余！请重新检查。'
        flag = False
    if data.abs().sum().sum() != data.sum().sum():
        print '结果中出现负数！已用0代替。'
        result = data - data[data < 0].fillna(0)
        flag = False
    for tp in data.dtypes.values:
        if tp.type is not np.int64:
            print '数据不是整数！已替换为整数。'
            flag = False
            break
    if True in (data.isnull().values):
        print '数据中包含空值，已替换为零。'
    result = result.fillna(0).astype(int)
    if flag == True:
        print 'Great！数据完整，不存在空值、负数和零。'

    return result


#使用本周其他6天数据预测 本周的,data是DataFrame格式的，返回的是的是一个data的变异副本
def getFeatrue_1(data):

    data = data.fillna(0)
    result = pd.DataFrame()
    for i in range(7):
        clf = LinearRegression()
        train_x = data[range(0,i)+range(i+1,6)]
        train_y = data[i]
        clf.fit(train_x, train_y)
        result[i] = clf.predict(train_x)
    result.index = data.index

    return result




























