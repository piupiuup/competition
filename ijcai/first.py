# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.ensemble import ExtraTreesRegressor
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from matplotlib.pylab import rcParams
from sklearn.cluster import KMeans
import os

shop_info_path = 'C:\\Users\\CSW\\Desktop\\python\\IJCAI\\data\\shop_info.txt'
user_pay_path = 'C:\\Users\\CSW\\Desktop\\python\\IJCAI\\data\\user_pay.txt'

#########################################读取文件#########################################
#读取shop_info 文件
shop_info = pd.read_csv(shop_info_path,header=None,names=['shopid','city_name','location_id',
                    'per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name',
                      'cate_3_name'])

#读取user_pay 文件
pay_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\IJCAI\data\user_pay.txt',header=None,names=['uid','shopid','time'])
pay_train['date'] = pay_train['time'].apply(lambda x: x[:10])

#整理时间客流量表
count_of_day = pay_train.groupby(['shopid','date'])['date'].count().unstack().iloc[:,1:].fillna(0)
count_of_day.sort_index(axis=1,inplace=True)
count_of_day.columns = range(492)

weekA = count_of_day[range(471,478)]
weekA.columns = range(0,7)
weekB = count_of_day[range(478,485)]
weekB.columns = range(0,7)
weekC = count_of_day[range(485,492)]
weekC.columns = range(0,7)

clf1 = LinearRegression()
clf1.fit(weekA,weekC)
clf2 = LinearRegression()
clf2.fit(pd.concat([weekA,weekB]),pd.concat([weekB,weekC]))

weekD_pred1 = pd.DataFrame(clf1.predict(weekB),index=range(1,2001))
weekE_pred1 = pd.DataFrame(clf1.predict(weekC),index=range(1,2001))
weekD_pred2 = pd.DataFrame(clf2.predict(weekC),index=range(1,2001))

weekD_pred = weekD_pred1*0.25+weekD_pred2*0.25+weekA*0.15+weekB*0.15+weekC*0.2
weekE_pred = weekE_pred1*0.4+weekA*0.15+weekB*0.2+weekC*0.25+5


#################################把天气信息整理到两个表中#################################
url = r'C:\Users\CSW\Desktop\python\IJCAI\data\train\Weather_Date'
city_list = os.listdir(url)
weather = pd.DataFrame(columns=['city','date','max','min','weather','direction','level'])
for city in city_list:
    url2 = url + '\\' + city
    weather_sub = pd.read_csv(url2,header=None,names=['date','max','min','weather','direction','level'])
    weather_sub['city'] = city
    weather = pd.concat([weather,weather_sub])

