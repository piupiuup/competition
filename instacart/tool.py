import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

# f1评分公式
def f1(y_true,y_pred):
    TP = len(set(y_true) & set(y_pred))         #预测为a类且正确的数量
    MP = len(y_true)                            #a类实际的数量
    MN = len(y_pred)                            #预测为a类的数量
    return 2*TP/(MP+MN)

# 最终评分公式
def instacart_grade(y_true,y_pred):
    return np.mean([f1(x, y) for x, y in zip(y_true, y_pred)])