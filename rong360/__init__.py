# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np

#金融塞
def analyse(data,feature,label,n=10,delect=None):
    # 删除无用数据
    data = data[data[feature] != delect]
    data = data[data[feature].notnull()]

    #对数据分类
    factor = pd.cut(data[feature], n)
    result = data.groupby(factor).apply(lambda group : group[label].value_counts())
    s = result.apply(lambda x: x.sum(),axis=1)
    p = result.icol(1)/s
    result['sum'] = s
    result['persent'] = p
    result.fillna(0, inplace=True)

    #绘图
    p.plot()

    return result

#读取表格
loan_time_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\train\loan_time_train.txt',header=None,names=['userid','loan_time'])
#loan_time_train['loan_time'] = loan_time_train['loan_time'] // 86400
user_info_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\train\user_info_train.txt',header=None,names=["userid", "sex", "profession", "education", "merge", "registered"])
bill_detail_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\train\bill_detail_train.txt',header=None,names=["userid", "time", "bank", "上期账单金额", "上期还款金额", "信用卡额度", "本期账单额度", "本期账单最低还款额", "消费笔数", "本期账单金额", "调整金额", "循环利息", "可用余额", "预借现金额度", "state"])
bill_detail_train['time'].replace(0,np.nan,inplace=True)
#bill_detail_train['time'] = bill_detail_train['time']//86400
overdue_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\train\overdue_train.txt',header=None,names=['userid','overdue'])

#融合表格
bill_loan_train = pd.merge(bill_detail_train,loan_time_train,on='userid',how='left')
train = pd.merge(overdue_train,user_info_train,on='userid',how='left')
train = pd.merge(train,loan_time_train,on='userid',how='left')


bill_loan_train = bill_loan_train[[u'userid',u'time',u'loan_time',u'state']]

t1=bill_loan_train[(bill_loan_train['time']>bill_loan_train['loan_time'])].groupby("userid",as_index=False)
t1=t1['time'].agg({'t1' : 'count'})
data = pd.merge(overdue_train,t1,on='userid',how='left')
data.t1.fillna(0,inplace=True)
analyse(data,'t1','overdue',delect=np.nan,n=6)
