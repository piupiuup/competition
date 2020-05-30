import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#读取表格信息
product_info = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\product_info.csv')
evaluation_category = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\evaluation_category.csv')
evaluation_category.category = evaluation_category.category.map({'A':1,'B':2})
sample_result = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\submit\example.txt')
product_quantity = pd.read_csv(r'C:\Users\CSW\Desktop\python\xiecheng\product_quantity.csv')
train_day=product_quantity.groupby(['product_id','product_date']).sum()['ciiquantity'].unstack()
product_quantity.product_date = product_quantity.product_date.map(lambda x:x[:7])
product_quantity.product_date = product_quantity.product_date + '-01'
product_quantity_unstack = product_quantity.groupby(['product_id','product_date']).sum()['ciiquantity'].unstack()
product_quantity_unstack = product_quantity_unstack.reset_index()
product_quantity_unstack = pd.merge(product_quantity_unstack,evaluation_category,on='product_id',how='right')
del product_quantity_unstack['category']
product_quantity_unstack = product_quantity_unstack.set_index('product_id')


#第一年的比例
def percent_of_one(arr):
    if np.sum(arr[-11:])==0:
        return 0
    return (sum(arr[:11]))/(sum(arr[:11])+sum(arr[-11:]))


#标准差太大用均值预测，标准差不大用去年同期预测
#
#同地区同类做聚类分析(分析每月走势)
#分析走势调整大小
def adjust(arr):
    n_del = 3
    n = 12
    c = 0.87
    valve = 170
    arr_temp = []
    for i in range(len(arr)):
        if arr[i]==0:
            continue
        else:
            arr_temp = arr[(i+1):]
            break
    if len(arr_temp)==0 or (len(arr_temp) < n_del and np.max(arr_temp)<valve):
        return 130
    elif len(arr_temp) < n_del:
        return (np.max(arr_temp)+arr_temp[-1])/2.
    else:
        arr_temp = arr_temp[(n_del-1):]
        arr_temp = arr_temp[-n:]
        coef = 1
        sum = 0.
        sum_of_coef = 0.
        for i in list(reversed(range(-len(arr_temp),0))):
            sum += arr_temp[i]*coef
            sum_of_coef += coef
            coef = coef*c
        ave = sum/sum_of_coef
        return ave


#过滤掉增长的n个月
def filter(df):
    result = pd.DataFrame(np.zeros(df.shape))
    result.iloc[:,:] = np.nan
    j = 0
    for name,row in df.iterrows():
        arr_temp = []
        for i in range(len(row.values)):
            if row.values[i] == 0:
                continue
            else:
                result.iloc[j:(j+1),(i + 1):] = row.values[(i + 1):]
                break
    result.columns = df.columns
    return result


#通过非空id，计算每个月的变化
train_temp = product_quantity_unstack[~product_quantity_unstack['2014-01-01'].isnull()]
sum_of_month = train_temp.sum()
percent_of_year = []
for i in range(12):
    percent_of_year.append(sum_of_month[i+6]/sum(sum_of_month[i:i+12]))
index = ['-07-','-08-','-09-','-10-','-11-','-12-','-01-','-02-','-03-','-04-','-05-','-06-']
percent_of_year = pd.DataFrame({'percent_of_year':percent_of_year},index=index).sort_index()
percent_of_year['percent_of_year'].plot()

#对比非空月份和 全部数据均值的变换
product_quantity_unstack.sum().plot()
train_day.sum().plot()

# 获取每个月的占全年的比重
def get_weight(data,district_id):
    sum_of_month = data.loc[:,'2014-01-01':'2015-11-01'].sum(axis=0)
    percent_of_year = []
    for i in range(12):
        percent_of_year.append(sum_of_month[i+6]/sum(sum_of_month[i:i+12]))
    index = [7,8,9,10,11,12,1,2,3,4,5,6]
    percent_of_year = pd.DataFrame({'percent_of_year':percent_of_year}).sort_index()
    percent_of_year = percent_of_year.T
    percent_of_year['district_id'] = [district_id]
    return percent_of_year

#获取商店聚类的id
def get_id_dict(data,col_name):
    data_notnull = data[~data['2014-01-01'].isnull()]
    id_dict = data_notnull[col_name].value_counts()
    id_dict = list(id_dict[id_dict>15].index)
    if -1 in id_dict:
        id_dict.remove(-1)
    id_dict = data[data[col_name].isin(id_dict)][['product_id',col_name]]
    id_dict.rename(columns={col_name:'district_id'},inplace=True)
    #获取每个月的比重
    grouped = data_notnull.groupby(col_name)
    percent_of_year = None
    for name, group in grouped:
        if percent_of_year is None:
            percent_of_year = get_weight(group,name)
        else:
            percent_of_year = pd.concat([percent_of_year,get_weight(group,name)])
    #获取每个城市的重心坐标
    grouped = data[~data['lat'].isin([0,-1])].groupby(col_name)
    location = None
    for name, group in grouped:
        temp = group.mean()[['lat','lon',col_name]]
        temp = pd.DataFrame(temp).T
        temp.rename(columns = {col_name:'district_id'},inplace=True)
        if location is None:
            location = temp
        else:
            location = pd.concat([location,temp])
    location['district_id'] = location['district_id'].astype('int')
    id_dict = pd.merge(id_dict, percent_of_year, on='district_id',how='left')
    id_dict = pd.merge(id_dict, location, on='district_id', how='left')
    return id_dict

#根据距离就算孤点的变化规律
def get_id_dict_other(id_other,id_dict):
    other = id_other[['product_id','lat','lon']].copy()
    dict = id_dict.copy()
    del dict['product_id']
    dict.drop_duplicates(inplace=True)
    ids = []
    for row in other.values:
        dict['lat_other'] = row[1]
        dict['lon_other'] = row[2]
        dict['distance'] = dict.apply(lambda x: (x['lat']-x['lat_other'])**2 + (x['lon']-x['lon_other'])**2,axis=1)
        district_id = dict['district_id'][dict['distance'].argmin()]
        try:
            district_id = district_id.values[0]
        except:
            pass
        ids.append([row[0],district_id])
    result = pd.DataFrame(ids,columns=['product_id','district_id'])
    result = pd.merge(result,dict,on='district_id',how='left')
    del result['lat_other'],result['lon_other'],result['distance']
    return result
#计算每个二级域名的坐标中心
info = pd.merge(product_quantity_unstack.reset_index(),product_info,on='product_id',how='right')
id_dict3 = get_id_dict(info,'district_id3')
id_dict2 = get_id_dict(info,'district_id2')
id_dict = pd.concat([id_dict3,id_dict2])
location_dict = id_dict.iloc[:,2:].drop_duplicates()
id_dict.drop_duplicates('product_id',inplace=True)
id_other = info[~info['product_id'].isin(id_dict['product_id'])]
id_dict1 = get_weight(id_other[(id_other['lat']==0) & (~id_other['2014-01-01'].isnull())],0)
id_dict0 = get_weight(id_other[(id_other['lat']==-1) & (~id_other['2014-01-01'].isnull())],-1)
id_dict1['lat'] = -1
id_dict1['lon'] = -1
id_dict0['lat'] = 0
id_dict0['lon'] = 0
id_dict1 = pd.merge(id_dict1,info[['product_id','lat']],on='lat',how='left')
id_dict0 = pd.merge(id_dict0,info[['product_id','lat']],on='lat',how='left')
id_dict = pd.concat([id_dict,id_dict1,id_dict0])
id_dict.drop_duplicates('product_id',inplace=True)
id_other = info[~info['product_id'].isin(id_dict['product_id'])]
id_dict_other = get_id_dict_other(id_other,id_dict)
id_dict = pd.concat([id_dict,id_dict_other])
id_dict['product_id'] = id_dict['product_id'].astype(int)
del id_dict['lon'],id_dict['lat'],id_dict['district_id']
id_dict.set_indx('product_id',inplace=True)
percent_of_year = id_dict.stack()
percent_of_year = pd.DataFrame(percent_of_year).reset_index()
percent_of_year.columns = ['product_id','month','percent_of_year']