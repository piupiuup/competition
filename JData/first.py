import pandas as pd
import numpy as np


#读取数据
user = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_User.csv',encoding='ANSI')
product = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Product.csv')
comment = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Comment.csv')
def get_action():
    action_2 = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Action_201602.csv')
    action_3 = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Action_201603.csv')
    action_3_extra = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Action_201603_extra.csv')
    action_4 = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Action_201604.csv')
    action = pd.concat([action_2,action_3,action_3_extra,action_4])
    return action
action = get_action()
action['date'] = action['time'].apply(lambda x:x[:10])
action['time'] = pd.to_datetime(action['time'])


#建立测试集
test_dates = ['2016-04-11','2016-04-12','2016-04-13','2016-04-14','2016-04-15']
test = action[(action['date'].isin(test_dates)) & (action['type']==4)][['user_id','sku_id']]

pred_dates = ['2016-04-06','2016-04-07','2016-04-08','2016-04-09','2016-04-10']

######################分析数据######################
#加入购物车后购买的概率
#当天直接购买的概率

labels
dates = pd.date_range('20160311','20160410')
user_recall = []
ui_recall = []
for i,date in enumerate(reversed(dates)):
    data_temp = action[(action['time']>str(date)) & (action['time']<'2016-04-11')]
    user_dict = data_temp['user_id'].unique()
    ui_dict = data_temp[['user_id','sku_id']].drop_duplicates()
    user_recall.append(sum(labels['user_id'].isin(user_dict)))
    ui_recall.append(pd.merge(ui_dict,labels,on=['user_id','sku_id'],how='inner').shape[0])

recall = pd.DataFrame({'user_recall':user_recall,'ui_recall':ui_recall})
recall = recall/1380


# 分析不同属性和品牌的转化率
product = pd.read_csv(r'C:\Users\csw\Desktop\python\JData\data\JData_Product.csv')
actions = get_cate8('2016-01-31','2016-04-16')
buy = actions[actions['type']==4].drop_duplicates(['user_id','sku_id'])
buy['label'] = 1
actions = actions.drop_duplicates(['user_id','sku_id'])
actions = pd.merge(actions,buy[['user_id','sku_id','label']],on=['user_id','sku_id'],how='left').fillna(0)
actions = pd.merge(actions,product,on='sku_id',how='left')
key = 'a1'
data_buy = actions.groupby(key,as_index=False)['label'].agg({'buy':'sum'})
data_look = actions.groupby(key,as_index=False)['label'].agg({'look':'count'})
data = pd.merge(data_look,data_buy,on=key).fillna(0)
data['percent'] = data['buy']/data['look']