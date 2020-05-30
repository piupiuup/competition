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


#使用线性回归训练预测
def predict_1(train_x,train_y,test_x):

    train_x = train_x.fillna(0)
    train_y = train_y.fillna(0)
    test_x = test_x.fillna(0)

    clf = LinearRegression()
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)
    result = pd.DataFrame(result,index=test_x.index)
    result = result - result[result < 0].fillna(0)

    return result






