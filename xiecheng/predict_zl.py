# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

months = ['2014-01-01','2014-02-01','2014-03-01','2014-04-01','2014-05-01',
          '2014-06-01','2014-07-01','2014-08-01','2014-09-01','2014-10-01',
          '2014-11-01','2014-12-01','2015-01-01','2015-02-01','2015-03-01',
          '2015-04-01','2015-05-01','2015-06-01','2015-07-01','2015-08-01',
          '2015-09-01','2015-10-01','2015-11-01',]
#计算两个月份的差值
def diff_of_month(month1,month2):
    try:
        years = int(month1[:4])-int(month2[:4])
        months = int(month1[5:7])-int(month2[5:7])
        months = months + years*12
        return months
    except:
        return(-30)


def get_start_month(data):
    result = data.copy()
    result.reset_index(inplace=True)
    rows = data.loc[:,'2014-01-01':'2015-11-01'].values
    start_months = []
    for row in rows:
        flag = False
        start = 0
        for i in row:
            if flag:
                pass
            else:
                if ~np.isnan(i):
                    flag = True
                    start_months.append(months[start])
            if (~flag) & (start==22):
                start_months.append(np.nan)
            start += 1

    result['start_month'] = start_months
    return result[['product_id','start_month']]

#计算均值
def get_mean(row):
    k1 = 0.9; k2 = 1.0558
    sum = 0; n = 0; j = 0
    for i in reversed(row):
        if ~np.isnan(i):
            sum += i * (k1**j) * (k2**j)
            n += k1**j
            j += 1
    if n == 0:
        mean = np.nan
    else:
        mean = sum/n
    return mean



def analyze(data):
    """
    返回np带的统计值
    """
    result = pd.DataFrame(columns=['product_id'])
    result['product_id'] = data.index
    result.index = data.index
    result['mean1'] = data.apply(get_mean,axis=1)
    result['mean0'] = data.mean(axis=1)
    result['max'] = data.max(skipna=True ,axis=1)
    result['min'] = data.min(skipna=True ,axis=1)
    result['jicha'] = result['max'] - result['min']
    result['head_first'] = data.iloc[:, -1:]
    result['head_two'] = data.iloc[:, -2:-1]
    return result

# 获取每个月的占全年的比重
def get_weight(data,district_id):
    sum_of_month = data.loc[:,'2014-01-01':'2015-11-01'].sum(axis=0)
    percent_of_year = []
    for i in range(12):
        percent_of_year.append(sum_of_month[i+6]/sum(sum_of_month[i:i+12]))
    index = [7,8,9,10,11,12,1,2,3,4,5,6]
    percent_of_year = pd.DataFrame({'percent_of_year':percent_of_year},index=index).sort_index()
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
def get_percent_of_year(product_quantity_unstack,product_info):
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
    id_dict.set_index('product_id',inplace=True)
    percent_of_year = id_dict.stack()
    percent_of_year = pd.DataFrame(percent_of_year).reset_index()
    percent_of_year.columns = ['product_id','month','percent_of_year']

    return percent_of_year


#修改空值部分 和 月份的变化趋势
def rule(product_quantity_unstack,df):
    data = df.copy()
    id_null = list(product_quantity_unstack[product_quantity_unstack.sum(axis=1)<1].index)
    weight_null = 105/(data[data['product_id'].isin(id_null)]['ciiquantity_month'].mean())
    weight_null = round((weight_null-0.005),2)
    data['coe'] = data.apply(lambda x: (int(x[1][3]) * 12 + int(x[1][5:7]) - 71) / 7.5 * weight_null if x[0] in id_null else 1, axis=1)
    data['ciiquantity_month'] = data['ciiquantity_month'] * data['coe']
    del data['coe']

    months = ['2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01',
              '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01', '2016-09-01',
              '2016-10-01', '2016-11-01', '2016-12-01', '2017-01-01']
    weights_all = [124, 129, 145, 136, 158, 152, 166, 198, 219, 167, 202, 145, 158, 173]
    coe = pd.DataFrame({'product_month':months,'weight_all':weights_all})
    mean = data.groupby('product_month',as_index=False)['ciiquantity_month'].mean()
    mean.rename(columns={'ciiquantity_month':'mean'},inplace=True)
    data = pd.merge(data, coe, on='product_month',how='left')
    data = pd.merge(data, mean, on='product_month', how='left')
    data['ciiquantity_month'] = data['ciiquantity_month'] * data['weight_all'] / data['mean']
    del data['weight_all'],data['mean']

    return data

product_info_path = r'C:\Users\CSW\Desktop\python\xiecheng\product_info.csv'
evaluation_category_path = r'C:\Users\CSW\Desktop\python\xiecheng\evaluation_category.csv'
sample_result_path = r'C:\Users\CSW\Desktop\python\xiecheng\submit\example.txt'
product_quantity_path = r'C:\Users\CSW\Desktop\python\xiecheng\product_quantity.csv'


#################################读取表格信息############################
product_info = pd.read_csv(product_info_path)
evaluation_category = pd.read_csv(evaluation_category_path)
sample_result = pd.read_csv(sample_result_path)
product_quantity = pd.read_csv(product_quantity_path)


#############################对读取的数据进行整理##########################
evaluation_category['category'] = evaluation_category['category'].map({'A' :1 ,'B' :2})
product_quantity['product_month'] = product_quantity['product_date'].map(lambda x :x[:7])
product_quantity['product_month'] = product_quantity['product_month'] + '-01'
product_quantity_unstack = product_quantity.groupby(['product_id', 'product_month'])['ciiquantity'].sum().unstack()
#补充没有销售记录的id
product_quantity_unstack = product_quantity_unstack.reset_index()
product_quantity_unstack = pd.merge(product_quantity_unstack ,evaluation_category ,on='product_id' ,how='right')
product_quantity_unstack.set_index('product_id',inplace=True)
del product_quantity_unstack['category']

# 开始营业的时间
start_month = get_start_month(product_quantity_unstack)

# 修补数据
product_info.loc[173 ,'cooperatedate'] = '2013-03-15'
product_info.loc[1212 ,'cooperatedate'] = '2013-03-15'
product_info.loc[2164 ,'cooperatedate'] = '2014-03-15'
product_info.loc[3444 ,'cooperatedate'] = '2015-05-15'
product_info.loc[2317 ,'cooperatedate'] = '2015-02-15'


#通过非空id，计算每个月的变化
train_temp = product_quantity_unstack[~product_quantity_unstack['2014-01-01'].isnull()]
sum_of_month = train_temp.sum()
percent_of_year = []
for i in range(12):
    percent_of_year.append(sum_of_month[i+6]/sum(sum_of_month[i:i+12]))
index = [7,8,9,10,11,12,1,2,3,4,5,6]
percent_of_year = pd.Series(percent_of_year,index=index).sort_index()



data_quantity = product_quantity_unstack.fillna(0).stack().reset_index()
data_quantity.columns = ['product_id', 'product_month', 'ciiquantity_month']
all_sample = pd.concat([data_quantity, sample_result])
del all_sample['ciiquantity_month']
all_sample['year'] = all_sample['product_month'].map(lambda x: x[:4]).astype(int)
all_sample['month'] = all_sample['product_month'].map(lambda x: x[5:7]).astype(int)
# 距离合作日期天数 & 合并info信息
all_sample = pd.merge(all_sample, product_info, on='product_id', how='left')
all_sample['diff_of_day'] = (pd.to_datetime(all_sample['product_month']) -
                             pd.to_datetime(all_sample['cooperatedate'])).apply(lambda x: x.days)
#去除增长趋势后，每个月所占全年的比重
all_sample['percent_of_year_all'] = all_sample['month'].map(dict(percent_of_year))
#添加category信息
all_sample = pd.merge(all_sample,evaluation_category,on='product_id',how='left')


#根据销量类型给商品分类
product_ord1 = product_quantity.groupby(['product_id','orderattribute1'])['ciiquantity'].count().unstack().apply(lambda x:x.argmax(),axis=1).reset_index()
product_ord2 = product_quantity.groupby(['product_id','orderattribute2'])['ciiquantity'].count().unstack().apply(lambda x:x.argmax(),axis=1).reset_index()
product_ord3 = product_quantity.groupby(['product_id','orderattribute3'])['ciiquantity'].count().unstack().apply(lambda x:x.argmax(),axis=1).reset_index()
product_ord4 = product_quantity.groupby(['product_id','orderattribute4'])['ciiquantity'].count().unstack().apply(lambda x:x.argmax(),axis=1).reset_index()
product_ord1.columns = ['product_id','ord1']
product_ord2.columns = ['product_id','ord2']
product_ord3.columns = ['product_id','ord3']
product_ord4.columns = ['product_id','ord4']
product_ord1['gs'] = 1
product_ord1 = product_ord1.groupby(['product_id','ord1'])['gs'].count().unstack().fillna(0).add_prefix('ord1_').reset_index()
all_sample = pd.merge(all_sample,product_ord1,on='product_id',how='left')
all_sample = pd.merge(all_sample,product_ord2,on='product_id',how='left')
all_sample = pd.merge(all_sample,product_ord3,on='product_id',how='left')
all_sample = pd.merge(all_sample,product_ord4,on='product_id',how='left')


#按地区提取商家变化趋势
percent_of_year = get_percent_of_year(product_quantity_unstack,product_info)
all_sample = pd.merge(all_sample,percent_of_year,on=['product_id','month'],how='left')



##########################线下训练集########################
#统计前5月的均值，最大值，最小值，极差
train_quantity = pd.DataFrame.copy(product_quantity_unstack.iloc[:, 4:9])
train_quantity_feature = analyze(train_quantity)
train_y = pd.DataFrame.copy(product_quantity_unstack.iloc[:, 9:21])
train_y = train_y.stack().reset_index()
train_y.columns = sample_result.columns
train = pd.merge(train_y, train_quantity_feature, on='product_id', how='left')
train = pd.merge(train,all_sample,on=['product_id','product_month'],how='left')
train['long_of_month'] = train['product_month'].apply(lambda x:diff_of_month(x,'2014-09-01'))
train = pd.merge(train,start_month,on='product_id',how='left')
train['start_month'] = train.apply(lambda x: diff_of_month(x['product_month'],x['start_month']),axis=1)
train = train[train['start_month']>0]
train = train.fillna(0)

#########################线下测试集#########################
test_quantity = pd.DataFrame.copy(product_quantity_unstack.iloc[:, 16:21])
test_quantity_feature = analyze(test_quantity)
test_y = pd.DataFrame.copy(product_quantity_unstack.iloc[:, -2:])
test_y = test_y.stack().reset_index()
test_y.columns = sample_result.columns
test = pd.merge(test_y, test_quantity_feature, on='product_id', how='left')
test = pd.merge(test,all_sample,on=['product_id','product_month'],how='left')
test['long_of_month'] = test['product_month'].apply(lambda x:diff_of_month(x,'2015-09-01'))
test = pd.merge(test,start_month,on='product_id',how='left')
test['start_month'] = test.apply(lambda x: diff_of_month(x['product_month'],x['start_month']),axis=1)
test = test.fillna(0)


#从备选特征中选取有用的特征
feature_label = [ 'product_id','long_of_month','percent_of_year','month',
'percent_of_year_all','head_first','head_two',
'mean0', 'max', 'min', 'jicha', 'category',  'ord1_1',  'ord1_2', 'ord1_3', 'ord2',
 'ord3', 'ord4', 'district_id1', 'district_id2', 'district_id3', 'district_id4',
'lat', 'lon', 'eval', 'eval2', 'eval4', 'voters', 'maxstock']

xgtrain_x = xgb.DMatrix(train[feature_label], train['ciiquantity_month'])
xgtrain_y = xgb.DMatrix(test[feature_label], test['ciiquantity_month'])

# xgtest = xgb.DMatrix(test[feature_label])

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 7,
          # 'lambda':100,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'min_child_weight': 10,  # 8~10
          'eta': 0.02,
          'seed':66,
          # 'nthread':12
          }
params['silent'] = 1
watchlist = [(xgtrain_x, 'train'), (xgtrain_y, 'eval')]
model = xgb.train(params, xgtrain_x, 5000, watchlist, early_stopping_rounds=100)
zycs = model.best_ntree_limit

##########################线上训练集########################
#统计前5月的均值，最大值，最小值，极差
train = pd.concat([train,test])


##########################线上测试集########################
#统计前5月的均值，最大值，最小值，极差
test_quantity = pd.DataFrame.copy(product_quantity_unstack.iloc[:, 18:])
test_quantity_feature = analyze(test_quantity)
test_y = sample_result
test = pd.merge(test_y, test_quantity_feature, on='product_id', how='left')
test = pd.merge(test,all_sample,on=['product_id','product_month'],how='left')
test['long_of_month'] = test['product_month'].apply(lambda x:diff_of_month(x,'2015-11-01'))
test = test.fillna(0)



xgtest = xgb.DMatrix(test[feature_label])
xgtrain = xgb.DMatrix(train[feature_label], label=train['ciiquantity_month'])
watchlist = [ (xgtrain,'train')]
model = xgb.train(params, xgtrain, zycs, watchlist, early_stopping_rounds=50)
test_y = model.predict(xgtest)
sample_result.ciiquantity_month = test_y
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x:x if x > 0 else 5)
#对最后两个月乘以1.3




###########################使用xgb增加单个月份#######################
sample_result = pd.DataFrame(columns = ['product_id', 'product_month', 'ciiquantity_month'])
for i in tqdm(range(1,13)):
    train_sub = train[train['month']==i]
    train_temp = pd.concat([train,train_sub])
    #for j in range(0,1):
    #    train_temp = pd.concat([train_temp, train_sub])
    xgtrain = xgb.DMatrix(train_temp[feature_label], label=train_temp['ciiquantity_month'])
    test_temp = test[test['month']==i]
    xgtest = xgb.DMatrix(test_temp[feature_label])
    model = xgb.train(params, xgtrain, zycs)
    test_y = model.predict(xgtest)
    sample_result_sub = pd.DataFrame({'product_id':test_temp.product_id,
                                      'product_month':test_temp.product_month,
                                      'ciiquantity_month':test_y})
    sample_result = pd.concat([sample_result,sample_result_sub])

sample_result_sub.sort_values([ 'product_month','product_id'],inplace=True)
sample_result.index = list(range(sample_result.shape[0]))
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x: x if x > 0 else 5)
sample_result = sample_result.loc[:,['product_id', 'product_month', 'ciiquantity_month']]
sample_result['product_id'] = sample_result['product_id'].astype(int)
sample_result_xgb = sample_result
result_xgb = rule(product_quantity_unstack,sample_result_xgb)


###########################随机森林增加月份权重#######################
sample_result = pd.DataFrame(columns = ['product_id', 'product_month', 'ciiquantity_month'])
for i in tqdm(range(1,13)):
    train_sub = train[train['month']==i]
    train_temp = pd.concat([train,train_sub])
    #for j in range(0,1):
    #    train_temp = pd.concat([train_temp, train_sub])
    test_temp = test[test['month']==i]
    model = RandomForestRegressor(n_estimators=2000, criterion='mse', max_depth=None,
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, max_features=6,
                              max_leaf_nodes=None, bootstrap=True,
                              oob_score=False, n_jobs=-1, random_state=66,
                              verbose=0, warm_start=False)
    model = model.fit(train_temp[feature_label], train_temp['ciiquantity_month'])
    test_y = model.predict(test_temp[feature_label])
    sample_result_sub = pd.DataFrame({'product_id':test_temp.product_id,
                                      'product_month':test_temp.product_month,
                                      'ciiquantity_month':test_y})
    sample_result = pd.concat([sample_result,sample_result_sub])

sample_result_sub.sort_values([ 'product_month','product_id'],inplace=True)
sample_result.index = list(range(sample_result.shape[0]))
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x: x if x > 0 else 5)
sample_result = sample_result.loc[:,['product_id', 'product_month', 'ciiquantity_month']]
sample_result['product_id'] = sample_result['product_id'].astype(int)
sample_result_rf = sample_result
result_rf = rule(product_quantity_unstack,sample_result_rf)


###########################GBRT增加月份权重#######################
sample_result = pd.DataFrame(columns = ['product_id', 'product_month', 'ciiquantity_month'])
for i in tqdm(range(1,13)):
    train_sub = train[train['month']==i]
    train_temp = pd.concat([train,train_sub])
    #for j in range(0,1):
    #    train_temp = pd.concat([train_temp, train_sub])
    test_temp = test[test['month']==i]
    model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.01,loss='ls',
                                  max_depth=7,criterion='friedman_mse',
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, subsample=0.7, max_features=5,
                              max_leaf_nodes=None, random_state=66)
    model = model.fit(train_temp[feature_label], train_temp['ciiquantity_month'])
    test_y = model.predict(test_temp[feature_label])
    sample_result_sub = pd.DataFrame({'product_id':test_temp.product_id,
                                      'product_month':test_temp.product_month,
                                      'ciiquantity_month':test_y})
    sample_result = pd.concat([sample_result,sample_result_sub])

sample_result_sub.sort_values([ 'product_month','product_id'],inplace=True)
sample_result.index = list(range(sample_result.shape[0]))
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x: x if x > 0 else 5)
sample_result = sample_result.loc[:,['product_id', 'product_month', 'ciiquantity_month']]
sample_result['product_id'] = sample_result['product_id'].astype(int)
sample_result_gbrt = sample_result
result_gbrt = rule(product_quantity_unstack,sample_result_gbrt)

###########################ET增加月份权重#######################
sample_result = pd.DataFrame(columns = ['product_id', 'product_month', 'ciiquantity_month'])
for i in tqdm(range(1,13)):
    train_sub = train[train['month']==i]
    train_temp = pd.concat([train,train_sub])
    #for j in range(0,1):
    #    train_temp = pd.concat([train_temp, train_sub])
    test_temp = test[test['month']==i]
    model = ExtraTreesRegressor(n_estimators=2000, n_jobs=-1, min_samples_split=2,
                             min_samples_leaf=1, max_depth=20, max_features=13,
                            criterion='mse',random_state=66)
    model = model.fit(train_temp[feature_label], train_temp['ciiquantity_month'])
    test_y = model.predict(test_temp[feature_label])
    sample_result_sub = pd.DataFrame({'product_id':test_temp.product_id,
                                      'product_month':test_temp.product_month,
                                      'ciiquantity_month':test_y})
    sample_result = pd.concat([sample_result,sample_result_sub])
    print(i, end=',')

sample_result_sub.sort_values([ 'product_month','product_id'],inplace=True)
sample_result.index = list(range(sample_result.shape[0]))
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x: x if x > 0 else 5)
sample_result = sample_result.loc[:,['product_id', 'product_month', 'ciiquantity_month']]
sample_result['product_id'] = sample_result['product_id'].astype(int)
sample_result_et = sample_result
result_et = rule(product_quantity_unstack,sample_result_et)


#融合
sample_result['ciiquantity_month'] = (result_xgb['ciiquantity_month']*0.25
                                      + result_rf['ciiquantity_month']*0.25
                                      + result_gbrt['ciiquantity_month']*0.25
                                      + result_et['ciiquantity_month']*0.25)




#测评函数
def grade(y_true,y_pred):
    from sklearn.metrics import mean_squared_error
    result = mean_squared_error(y_true,y_pred)
    result = result**0.5
    return result



##############################使用随机森林预测##################################
model = RandomForestRegressor(n_estimators=2000, criterion='mse', max_depth=None,
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, max_features=6,
                              max_leaf_nodes=None, bootstrap=True,
                              oob_score=False, n_jobs=-1, random_state=66,
                              verbose=0, warm_start=False)
model = model.fit(train[feature_label], train['ciiquantity_month'])
test_y = model.predict(test[feature_label])
grade(test['ciiquantity_month'],test_y)
sample_result.ciiquantity_month = test_y
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x:x if x > 0 else 5)



##############################使用GBDT预测##################################
model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.01,loss='ls',
                                  max_depth=9,criterion='friedman_mse',
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, subsample=0.7, max_features=5,
                              max_leaf_nodes=None, random_state=66)
model = model.fit(train[feature_label], train['ciiquantity_month'])
test_y = model.predict(test[feature_label])
grade(test['ciiquantity_month'],test_y)
sample_result.ciiquantity_month = test_y
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x:x if x > 0 else 5)

map3得分
##############################使用ET预测##################################
model = ExtraTreesRegressor(n_estimators=2000, n_jobs=-1, min_samples_split=2,
                             min_samples_leaf=1, max_depth=20, max_features=13,
                            criterion='mse',random_state=66)
model = model.fit(train[feature_label], train['ciiquantity_month'])
test_y = model.predict(test[feature_label])
grade(test['ciiquantity_month'],test_y)
sample_result.ciiquantity_month = test_y
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x:x if x > 0 else 5)


##############################使用lgb预测##################################
lgb_train = lgb.Dataset(train[feature_label], train['ciiquantity_month'])
lgb_test = lgb.Dataset(test[feature_label], test['ciiquantity_month'])
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_test,
                early_stopping_rounds=50)

test_y = gbm.predict(test[feature_label], num_iteration=gbm.best_iteration)
grade(test['ciiquantity_month'],test_y)
sample_result.ciiquantity_month = test_y
sample_result.ciiquantity_month = sample_result.ciiquantity_month.map(lambda x:x if x > 0 else 5)


