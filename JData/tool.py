import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pylab as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#F11公式
def grade_F11(y_true,y_pred,silent=0):
    label_pred = y_pred['user_id'].copy().drop_duplicates()
    label_true = y_true['user_id'].copy().drop_duplicates()

    a = float(sum(label_pred.isin(label_true.values)))  # 正确的个数
    m = len(label_pred)                                 # 预测的个数
    n = len(label_true)                                 # 实际的个数
    score = 6 * a / (5 * m + n)
    if silent != 1:
        print ('label正确的个数为：' + str(a))
        print ('label实际的个数为：' + str(n) + '，  label召回率为：' + str(a / n))
        print ('label预测的个数为：' + str(m) + '，  label准确率为：' + str(a / (1*m)))
        print ('F11得分：' + str(score))
    return score


# F12公式
def grade_F12(y_true, y_pred, silent=0):

    a = (y_pred[['user_id','sku_id']].merge(y_true[['user_id', 'sku_id']], on=['user_id', 'sku_id'])).shape[0]   # 正确的个数
    m = len(y_pred)                                  # 预测的个数
    n = len(y_true)                                  # 实际的个数
    score = 5 * a / (2 * m + 3 * n)
    if silent != 1:
        print ('pred正确的个数为：' + str(a))
        print ('pred实际的个数为：' + str(n) + '，  pred召回率为：' + str(a / n))
        print ('pred预测的个数为：' + str(m) + '，  pred准确率为：' + str(a / (1*m)))
        print ('predF12得分：' + str(score))
    return score


#测评函数
def grade(y_true,y_pred,silent=0):
    if 'sku_id' not in y_true.columns:
        y_true['sku_id'] = 0
    if 'sku_id' not in y_pred.columns:
        y_pred['sku_id'] = 0
    y_pred = y_pred[['user_id','sku_id']].drop_duplicates()
    y_true = y_true[['user_id', 'sku_id']].drop_duplicates()


    #计算F11
    F11 = grade_F11(y_pred, y_true, silent)
    F12 = grade_F12(y_pred, y_true, silent)
    score = 0.4*F11 + 0.6*F12
    if silent!=1:
        print ('最终得分：' + str(score))
    return score


#提交去重
def select(data,valve=0.1):
    data.sort_values('label', ascending=False, inplace=True)
    data = data.loc[:,['user_id', 'sku_id', 'label']]
    result = []
    for user_id,group in data.groupby('user_id'):
        result.append(group.values[0])
    result = pd.DataFrame(result,columns=['user_id', 'sku_id', 'label'])
    result = result.loc[:,[ 'user_id',  'sku_id', 'label']]
    result = result[result['label']>valve].iloc[:,:2]
    result = result.astype(int)
    print ('筛选出'+str(result.shape[0])+'个ID')
    return result



#删除购买过的user的action
def delete(data):
    return data[~data['user_id'].isin(data[(data['type']==4) & (data['cate']==8)]['user_id'].unique())]



#auc评分
def grade_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

#判断购物车里是否还有cate8
def get_if_exist(data):
    temp = data[(data['cate']==8) & (data['type'].isin([2,3]))]
    def if_exist(x):
        x_temp = x.drop_duplicates(['user_id','sku_id'],keep='last')
        return 1 if 2 in x_temp['type'].values else 0
    result = temp.groupby('user_id').apply(lambda x:if_exist(x))
    result = set(result[result==1].index)

    return result

#有过删除购物车行为的最后又添加
def if_del_add(data):
    temp = data[data['cate']==8]
    data_del = temp[temp['type']==3]
    data_add = temp[temp['type'].isin([2,3])]
    data_add.drop_duplicates('user_id',keep='last',inplace=True)
    data_add = data_add[data_add['type']==2]
    result = set(data_add[data_add['user_id'].isin(data_del['user_id'].values)]['user_id'].values)

    return result

#有过5行为的
def n5(data):
    temp = data[data['cate'] == 8]
    result = temp.groupby('user_id')['type'].agg({'n_type':'nunique'})
    result = set(result[result['n_type']==5].index)
    return result

#获取最后一次type行为
def get_ast_type(data, type=3):
    temp = data[data['cate'] == 8]
    temp = temp.drop_duplicates('user_id',keep='last')
    temp = temp[temp['type']==type]
    result = set(temp.user_id)
    return result

# 规则筛选
def get_user(user,n=700):
    data = user.copy()
    data = data[(data['type4']!=1) & (data['user_lv_cd'].isin([3,4,5]))]
    data.sort_values('user_first_tm',ascending=True,inplace=True)
    return data.head(n)

result_final = pd.merge(result,result_first_tm,on='user_id',how='left')
result_final['label'] = result_final['probability_x'] + (0.5*result_final['probability_y'])
result_final.sort_values('label',ascending=False,inplace=True)
result_final = result_final[(result_final['type4']!=1) & (result_final['user_lv_cd']>2)]
result_final['sku_id'] = 678
output_temp = result_final.head(700)[['user_id','sku_id']]
sum(output_temp['user_id'].isin(output['user_id'].values))