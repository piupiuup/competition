import pandas as pd
import numpy as np
import gc
import os
import pickle

action_2_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Action_201602.csv"
action_3_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Action_201603.csv"
action_4_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Action_201604.csv"
comment_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Comment.csv"
product_path = r"C:\Users\csw\Desktop\python\JData\data\JData_Product.csv"
user_path = r"C:\Users\csw\Desktop\python\JData\data\JData_User.csv"

dates = ['2016-01-31', '2016-02-01', '2016-02-02', '2016-02-03', '2016-02-04', '2016-02-05', '2016-02-06',
     '2016-02-07', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-02-13',
     '2016-02-14', '2016-02-15', '2016-02-16', '2016-02-17', '2016-02-18', '2016-02-19', '2016-02-20',
     '2016-02-21', '2016-02-22', '2016-02-23', '2016-02-24', '2016-02-25', '2016-02-26', '2016-02-27',
     '2016-02-28', '2016-02-29', '2016-03-01', '2016-03-02', '2016-03-03', '2016-03-04', '2016-03-05',
     '2016-03-06', '2016-03-07', '2016-03-08', '2016-03-09', '2016-03-10', '2016-03-11', '2016-03-12',
     '2016-03-13', '2016-03-14', '2016-03-15', '2016-03-16', '2016-03-17', '2016-03-18', '2016-03-19',
     '2016-03-20', '2016-03-21', '2016-03-22', '2016-03-23', '2016-03-24', '2016-03-25', '2016-03-26',
     '2016-03-27', '2016-03-28', '2016-03-29', '2016-03-30', '2016-03-31', '2016-04-01', '2016-04-02',
     '2016-04-03', '2016-04-04', '2016-04-05', '2016-04-06', '2016-04-07', '2016-04-08', '2016-04-09',
     '2016-04-10', '2016-04-11', '2016-04-12', '2016-04-13', '2016-04-14', '2016-04-15']

#获取最近发生过关系的  ID
def action_id(data,days):
    data_sub = data[(data['date'].isin(days))]
    ids = []
    grouped = grouped = data_sub.groupby(['user_id', 'sku_id'])
    for name, group in grouped:
        ids.append(name)
    print ('发生过关系的ID个数为：' + str(len(ids)) + '个')
    result = pd.DataFrame(ids, columns=['user_id', 'sku_id'])
    return result


#获取最近有具体关系的ID
def get_id(data,days,type=0):
    mapping = {0: '过', 1: '浏览', 2: '加入购物车', 3: '购物车删除', 4: '下单', 5: '关注', 6: '点击'}
    data_sum = None
    if type==0:
        data_sub = data[(data['date'].isin(days))]
    else:
        data_sub = data[(data['date'].isin(days)) & (data['type']==type)]
    ids = []
    grouped = grouped = data_sub.groupby(['user_id','sku_id'])
    for name,group in grouped:
        ids.append(name)
    print ('有'+mapping[type]+'关系的ID个数为：'+str(len(ids))+'个')
    result = pd.DataFrame(ids,columns=['user_id','sku_id'])
    return result

#读取action数据内容
def get_action():
    dump_path = r'C:\Users\csw\Desktop\python\JData\data\action.pkl'
    if os.path.exists(dump_path):
        action = pickle.load(open(dump_path,'rb'))
    else:
        action_2 = pd.read_csv(action_2_path)
        action_3 = pd.read_csv(action_3_path)
        action_4 = pd.read_csv(action_4_path)
        action = pd.concat([action_2,action_3,action_4])
        action['date'] = action['time'].apply(lambda x: x[:10])
        action['time'] = pd.to_datetime(action['time'])
        pickle.dump(action, open(dump_path, 'wb+'))
    return action

#读取数据
print('读取数据')
user = pd.read_csv(user_path,encoding='ANSI')
product = pd.read_csv(product_path)
comment = pd.read_csv(comment_path)
action = get_action()



#######################提取最终购买记录#######################
print('提取最终的购买记录')
endtime = -2*5
y_true = get_id(action[action['cate']==8],dates[endtime:None if endtime==-5 else (endtime+5)],4)
y_true['label'] = 1
#选取片段，清理内存
print('选取片段，清理内存')
dates_temp = dates[30+endtime:None if endtime==0 else endtime]
action = action[action['date'].isin(dates_temp)]
gc.collect()


######################行为特征#########################
print('提取行为特征')
columns = ['user_id', 'sku_id', 'time_of_action', 'time_of_1', 'time_of_2', 'time_of_3', 'time_of_4',
 'time_of_5', 'time_of_6', 'count_of_1', 'count_of_2', 'count_of_3', 'count_of_4', 'count_of_5',
           'count_of_6','count_of_action']
#提取训练用的标签ID（前5天）
ids = get_id(action[action['cate']==8],dates_temp[-5:])
action_id = pd.merge(ids, action, on=['user_id','sku_id'],how='left')
features = []
for id, group in action_id.groupby(['user_id','sku_id']):
    sample = [id[0],id[1]]                                                      #添加id
    sample.append((pd.Timestamp(dates_temp[-1])-group['time'].max()).days+1)   #最后一次action时间
    time_of_action = pd.Timestamp(dates_temp[-1])-group.groupby('type')['time'].max()
    sample.append(-1 if 1 not in time_of_action.index else time_of_action[1].days+1)#最后一次浏览的时间
    sample.append(-1 if 2 not in time_of_action.index else time_of_action[2].days+1)#最后一次加入购物车的时间
    sample.append(-1 if 3 not in time_of_action.index else time_of_action[3].days+1)# 最后一次删除购物车的时间
    sample.append(-1 if 4 not in time_of_action.index else time_of_action[4].days+1)# 最后一次购买的时间
    sample.append(-1 if 5 not in time_of_action.index else time_of_action[5].days+1)# 最后一次关注的时间
    sample.append(-1 if 6 not in time_of_action.index else time_of_action[6].days+1)# 最后一次点击的时间
    sample.append(group[group['type'] == 1].shape[0])  # 浏览次数
    sample.append(group[group['type'] == 2].shape[0])  # 加入购物车
    sample.append(group[group['type'] == 3].shape[0])  # 删除购物车
    sample.append(group[group['type'] == 4].shape[0])  # 购买
    sample.append(group[group['type'] == 5].shape[0])  # 关注
    sample.append(group[group['type'] == 6].shape[0])  # 点击
    sample.append(group.shape[0])   #总action次数
    features.append(sample)

del action_id
action_feature = pd.DataFrame(features,columns=columns)




######################人物特征#########################
print('提取人物特征')
user_feature = user.copy()
user_feature.set_index('user_id',inplace=True)
#年龄
user_feature['age'] = user_feature['age'].map({'-1':-1,'15岁以下':1,
                               '16-25岁':2,'36-45岁':3,'46-55岁':4,'56岁以上':5}).fillna(-1)
#注册日期
user_feature['user_reg_tm'] = (pd.Timestamp('2016-04-20')-
                       pd.to_datetime(user_feature['user_reg_tm'])).apply(lambda x:-10000 if type(x) is pd.tslib.NaTType else x.days)
#购买次数
user_feature['count_buy_user'] = action[action['type']==4].groupby('user_id')['cate'].count()
user_feature['count_buy_user'] = user_feature['count_buy_user'].fillna(0)
#action总次数
user_feature['count_action_user'] = action.groupby('user_id')['cate'].count()
user_feature['count_action_user'] = user_feature['count_action_user'].fillna(0)
#每成交一次的action次数
user_feature['action_per_user'] = user_feature['count_action_user']/user_feature['count_buy_user']
user_feature['action_per_user'] = user_feature['action_per_user'].replace(np.inf,-1)
#最近浏览cate8的总次数
#user_feature['count_action_cate8'] = action[action['cate']==8].groupby('user_id')['sku_id'].count()
#最近浏览cate8的种类数
#user_feature['count_product_cate8'] = action[action['cate']==8].groupby('user_id')['sku_id'].apply(lambda x:x.nunique())

user_feature.reset_index(inplace=True)



######################商品特征#########################
print('提取商品特征')
product_feature = comment[comment['dt']=='2016-04-11']
del product_feature['dt'],product_feature['has_bad_comment']
product_feature = pd.merge(product[['sku_id','a1','a2','a3','cate','brand']],
                           product_feature,on='sku_id',how='left')
product_feature.set_index('sku_id',inplace=True)
action_id = action[action['sku_id'].isin(product_feature.index)]
#商品点击率
product_feature['count_of_people'] = action_id.groupby('sku_id')['user_id'].nunique()
product_feature['count_of_people'] = product_feature['count_of_people'].fillna(0)
#商品销售数量
product_feature['count_of_sale'] = action_id[action_id['type']==4].groupby('sku_id')['sku_id'].count()
product_feature['count_of_sale'] = product_feature['count_of_sale'].fillna(0)
#商品转化率
product_feature['conversion'] = product_feature['count_of_sale']/product_feature['count_of_people']
#当天购买的比例

product_feature.reset_index(inplace=True)






######################品牌特征##########################
#品牌销量
#品牌购物车转化率
#当天购买的比例


#######################合并添加标签#############################
feature = pd.merge(action_feature,user_feature,on='user_id',how='left')
feature = pd.merge(feature,product_feature,on='sku_id',how='left')


feature = pd.merge(feature,y_true,on=['user_id','sku_id'],how='left')
feature['label'] = feature['label'].fillna(0)
feature = feature.fillna(-1)

print('feature'+str(endtime/-5)+'生成')




浏览数、收藏数、购物车加入、购物车删除、购买数、关注数、点击数、
平均活跃天数、最后活跃天数距离最终时间的天数、
年龄段、性别、用户等级、用户注册时间长度、
商品属性1、商品属性2、商品属性3、品类、 品牌
累计评分数、是否有差评、差评比率、平均累计评分数、平均是否有差评、平均差评比率

