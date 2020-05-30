import numpy as np
import pandas as pd


#建立评分函数(y_true,y_pred都是pd.Series的格式)
def score(y_true,y_pred):
    #RMSE
    def RMSE(y_true,y_pred):
        return (y_true-y_pred).apply(lambda x: x*x).sum()**0.5

    #RMSE MAX
    def MAX_RMSE(y_true,y_pred):
        return y_true.apply(lambda x: x**2 if x>2.5 else (5-x)**2).sum()**0.5

    return (MAX_RMSE(y_true,y_pred)-RMSE(y_true,y_pred))/MAX_RMSE(y_true,y_pred)*10



