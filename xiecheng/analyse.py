
import pandas as pd
import scipy
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb


#################################读取表格信息############################
product_info = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\product_info.csv')
evaluation_category = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\evaluation_category.csv')
sample_result = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\submit\example.txt')
product_quantity = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\product_quantity.csv')



#############################对读取的数据进行整理##########################
evaluation_category['category'] = evaluation_category['category'].map({'A' :1 ,'B' :2})
product_quantity['product_month'] = product_quantity['product_date'].map(lambda x :x[:7])
product_quantity['product_month'] = product_quantity['product_month'] + '-01'
product_quantity_unstack = product_quantity.groupby(['product_id', 'product_month'])['ciiquantity'].sum().unstack()


