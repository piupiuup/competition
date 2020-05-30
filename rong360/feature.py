import numpy as np
import pandas as pd
import time
import datetime
import math


def get_feature(mold):

    #读取放款时间文件
    loan_time_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\%s\loan_time_%s.txt'%(mold,mold), header=None,names=['userid', 'loan_time'])
    start_day = loan_time_data['loan_time'].min() // 86400
    loan_time_data['loan_time'] = loan_time_data['loan_time'] // 86400 - start_day

    #读取用户信息列表
    user_info_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\%s\user_info_%s.txt'%(mold,mold), header=None, names=['userid', '用户性别', '用户职业', '用户教育程度', '用户婚姻状态', '用户户口类型'])
    data = user_info_data.copy()
    #读取信用卡信息
    bill_detail_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\%s\bill_detail_%s.txt'%(mold,mold), header=None,
                        names=['userid', 'time', '银行标识', '上期账单金额', '上期还款金额', '信用卡额度', '本期账单余额', '本期账单最低还款额',
                        '消费笔数', '本期账单金额', '调整金额', '循环利息', '可用余额', '预借现金额度', '还款状态'])
    bill_detail_data['time'] = bill_detail_data['time'] // 86400 - start_day

    #读取用户是否逾期
    if mold == 'train':
        overdue_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\%s\overdue_%s.txt'%(mold,mold),header=None,names=['userid','overdue'])
    else:
        overdue_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\rong360\%s\usersID_%s.txt' % (mold, mold), header=None, names=['userid', 'overdue'])
    data = pd.merge(overdue_data,user_info_data,on='userid',how='left')
    #将放款时间和信用卡时间融合在一个表中作对比
    bill_detail_data = pd.merge(bill_detail_data, loan_time_data, on='userid', how='left')

    #删除时间异常行
    temp = bill_detail_data[bill_detail_data['time']>-2000].copy()
    temp['diff'] = temp['loan_time']-temp['time']

    #老段子特征1
    data.set_index('userid', inplace=True)
    data['count_of_day1'] = temp[temp['time'] > temp['loan_time']].groupby('userid')['time'].agg('nunique')
    data['count_of_day2'] = temp[temp['time'] > temp['loan_time']].groupby('userid')['time'].agg('count')

    data.reset_index(inplace=True)

    return data


