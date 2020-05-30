import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



#读取id_list
sample_result = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\submit\example.txt')

#读取文件
product_info = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\product_info.csv')
product_quantity = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\product_quantity.csv')
product_quantity['month'] = product_quantity['product_date'].apply(lambda x:x[:7])
evaluation_category = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\evaluation_category.csv')


#处理文件
product_quantity_unstack = product_quantity.groupby(['product_id','month'])['ciiquantity'].agg('sum').unstack().fillna(0)
temp = pd.DataFrame(np.arange(1,4001).reshape(4000,1),index=range(1,4001))
product_quantity_unstack = pd.merge(product_quantity_unstack,temp,left_index=True,right_index=True,how='right').fillna(0).iloc[:,:23]
result1 = (product_quantity_unstack*1.6).stack().reset_index()
result1.columns = ['product_id','product_month','ciiquantity_month']
result1['product_month'] = result1['product_month'].map(lambda x:x+'-01')

n = 9
result2 = (product_quantity_unstack.apply(lambda x: np.average(x[n:],weights=range(n,len(x))),axis=1)*1.18).reset_index()
result2.columns = ['product_id','ciiquantity_month']
result = pd.merge(sample_result.iloc[:,:2],result2,on=['product_id'],how='outer').replace(0,130)
result['ciiquantity_month'] = result['ciiquantity_month'].astype('int')
result.to_csv(r'C:\Users\CSW\Desktop\python\xiecheng\submit\0310(1).csv',index=False)



#分析用程序
n = 11
product_quantity_unstack['result1'] = (product_quantity_unstack.apply(lambda x: np.average(x[n:23],weights=range(n,23)),axis=1)*1.0)
product_quantity_unstack['result2'] = (product_quantity_unstack.apply(lambda x: adjust(x[0:23]),axis=1)*1.0)
product_quantity_unstack['percent_of_one'] = product_quantity_unstack.apply(lambda x: percent_of_one(x[0:23]),axis=1)*1.0


#每月所占比例
sum_of_month = product_quantity_unstack.sum()
month = [7,8,9,10,11,12,1,2,3,4,5,6]
percent_of_month = {}
for i in range(12):
    percent_of_month[month[i]] = sum_of_month[i+6]/sum_of_month[i:i+12].sum()
percent_of_month = pd.Series(percent_of_month)