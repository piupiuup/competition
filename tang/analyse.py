import numpy as np
import pandas as pd

data_path = 'C:/Users/csw/Desktop/python/tang/data/'

test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb2312')
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb2312')