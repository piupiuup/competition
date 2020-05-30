# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime,time

'''
特征挖掘
'''


#读取文件
#选取学号，和助学经作为merge的标签

train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\subsidy_train.txt',header=None,names=['studentid','subsidy'])
train.set_index('studentid',inplace=True)
#读取card表提取信息
card_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\card_train.txt',header=None,names=['studentid','消费类别','消费地点','消费方式','消费时间','消费金额','剩余金额'])
card_train['消费地点'] = card_train['消费地点'].apply(lambda x:x if x is np.nan else x.replace('地点','loc'))
card_train['消费时间'] = card_train['消费时间'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
card_train['day'] = card_train['消费时间'].apply(lambda x: x.date())
#读取院系和成绩信息
score_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\score_train.txt',header=None,names=['studentid','college','rank'])
score_train.set_index('studentid',inplace=True)
#读取寝室门禁文件
dorm_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\dorm_train.txt',header=None,names=['studentid','time','IO'])
dorm_train['time'] = dorm_train['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
dorm_train['day'] = dorm_train['time'].apply(lambda x: x.date())
#读取图书馆图书借阅记录
borrow_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\borrow_train.txt',header=None,names=['studentid','time','book_name','bookid'])


#选取年份信息作为特征，13~14为0,14~15为1
train['year'] = train.index % 2
#寝室门禁进出次数
train['count_of_dorm'] = dorm_train.groupby('studentid').agg('size')
#寝室门禁记录的天数(默认在校天数)
train['day_of_dorm'] = dorm_train.groupby(['studentid'])['day'].agg(lambda x:len(x.unique()))
#平均每天进出次数
train['count_dorm_perday'] = train['count_of_dorm']/train['day_of_dorm']

#添加院系信息
train['college'] = score_train['college']
#添加院系人数（后期可删除）
count_of_student = score_train.groupby('college')['rank'].max()+1
score_train['count_of_student'] = score_train.apply(lambda x:count_of_student[int(x['college'])],axis=1)
train['count_of_student'] = score_train['count_of_student']
#添加排名（后期可删除）
train['rank'] = score_train['rank']
#排名归一化
train['rank_of_relative'] = train['rank']/train['count_of_student']

#每个同学卡消费总额
train['sum_of_consume'] = card_train[card_train['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('sum')
#每个同学卡消费平均值
train['mean_of_consume'] = card_train[card_train['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('mean')
#每个同学卡消费标准差
train['var_of_consume'] = card_train[card_train['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('var')
# 每个同学卡消费最大值
train['max_of_consume'] = card_train[card_train['消费类别'] == 'POS消费'].groupby('studentid')['消费金额'].agg('max')
# 每个同学卡消费标准差
train['min_of_consume'] = card_train[card_train['消费类别'] == 'POS消费'].groupby('studentid')['消费金额'].agg('min')
#每个同学卡消费次数
train['count_of_consume'] = card_train[card_train['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('size')
#每天消费总额
train['consume_of_day'] = card_train[card_train['消费类别']=='POS消费'].groupby(['studentid'])['day'].agg(lambda x:len(x.unique()))
#在校时校园卡使用频率
day_of_dorm = pd.DataFrame(dorm_train.groupby(['studentid'])['day'].agg(lambda x:set(x)))
day_of_consume = pd.DataFrame(card_train.groupby(['studentid'])['day'].agg(lambda x:set(x)))
dorm_and_consume = pd.merge(day_of_dorm,day_of_consume,left_index=True,right_index=True)
train['percent_of_day'] = dorm_and_consume.apply(lambda x:len(x[0].intersection(x[1])),axis=1)/day_of_dorm['day'].apply(len)

#每个同学吃饭消费总额
train['sum_of_meal'] = card_train[card_train['消费方式']=='食堂'].groupby('studentid')['消费金额'].agg('sum')
#每个同学吃饭消费的次数(对于时间间隔小于两个小时的合并为一次)
card_train_temp = list(card_train[card_train['消费方式']=='食堂'].values)
data = []
row_temp = card_train_temp.pop(0)
for row in card_train_temp:
    if np.abs((row[4]-row_temp[4]).total_seconds())<7200:
        row[5] = row_temp[5]+row[5]
    else:
        data.append(row_temp)
    row_temp = row
data.append(row_temp)
data = pd.DataFrame(data,columns=card_train.columns)
train['count_of_meal'] = data.groupby('studentid').size()
#每个同学的平均一次的饭钱
train['mean_of_meal'] = train['sum_of_meal']/train['count_of_meal']
#每个同学平均每顿饭的点餐次数
train['count_of_permeal'] = card_train[card_train['消费方式']=='食堂'].groupby('studentid').size()/train['count_of_meal']
#每位同学吃饭所占校园卡消费的总量
train['percent_of_meal'] = train['sum_of_meal']/train['sum_of_consume']
#每个同学吃饭的天数
train['day_of_meal'] = card_train[card_train['消费方式']=='食堂'].groupby('studentid')['day'].agg('nunique')
#在校吃饭天数占在校天数的比例
train['percent_of_atSchool'] = train['day_of_meal']/train['day_of_dorm']
#平均每天在校吃饭次数
train['count_meal_perday'] = train['count_of_meal']/train['day_of_meal']
#每天吃饭金额的标准差
temp  = pd.DataFrame(card_train[card_train['消费方式']=='食堂'].groupby(['studentid','day'])['消费金额'].sum())
temp.reset_index(inplace=True)
train['std_meal_perday'] = temp.groupby('studentid')['消费金额'].std()

#每个同学卡余额关于时间的平均值
card_train_temp = card_train.copy()
mean_of_overage = {}
for name,student in card_train_temp.groupby('studentid'):
    student = list(student.values)
    row_temp = student.pop(0)
    sum_of_seconds = 0
    sum_of_overage = 0
    for row in student:
        seconds = abs((row[4]-row_temp[4]).total_seconds())
        sum_of_seconds = sum_of_seconds + seconds
        sum_of_overage = sum_of_overage + seconds*row_temp[6]
        row_temp = row
    mean = sum_of_overage/sum_of_seconds if sum_of_seconds!=0 else None
    mean_of_overage[name] = mean
train['mean_of_overage'] = pd.Series(mean_of_overage)


#卡充值次数
card_train_temp = card_train[card_train['消费类别'].isin(['圈存转账','卡充值'])]
train['count_of_storage'] = card_train_temp.groupby('studentid')['消费类别'].agg('size')
#卡平均充值金额
train['mean_of_storage'] = card_train_temp.groupby('studentid')['消费金额'].agg('mean')
#卡最大充值金额
train['max_of_storage'] = card_train_temp.groupby('studentid')['消费金额'].agg('max')
# 卡最小充值金额
train['min_of_storage'] = card_train_temp.groupby('studentid')['消费金额'].agg('min')
# 卡充值金额标准差
train['std_of_storage'] = card_train_temp.groupby('studentid')['消费金额'].agg('std')
#充值时卡上的余额的平均值
train['mean_of_over_storage'] = card_train_temp.groupby('studentid')['剩余金额'].agg('mean')-card_train_temp.groupby('studentid')['消费金额'].agg('mean')

#每个同学开水消费总额
train['sum_of_water'] = card_train[card_train['消费方式']=='开水'].groupby('studentid')['消费金额'].agg('sum')
#每个同学开水消费总次数
train['count_of_meal'] = card_train[card_train['消费方式']=='开水'].groupby('studentid')['消费金额'].agg('count')
#每个同学开水消费平均值
train['mean_of_water'] = card_train[card_train['消费方式']=='开水'].groupby('studentid')['消费金额'].agg('mean')
#每个同学开水消费的最大值
train['max_of_water'] = card_train[card_train['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('max')
#每个同学使用开水的天数
train['day_of_water'] = card_train[card_train['消费方式'] == '开水'].groupby('studentid')['day'].agg('nunique')
#每个同学开水消费总额所占花销比例
train['percent_of_water'] = train['sum_of_water']/train['sum_of_consume']

#每个同学淋浴消费总额
train['sum_of_bath'] = card_train[card_train['消费方式']=='淋浴'].groupby('studentid')['消费金额'].agg('sum')
#每个同学淋浴消费总次数
train['count_of_bath'] = card_train[card_train['消费方式']=='淋浴'].groupby('studentid')['消费金额'].agg('count')
#每个同学淋浴消费平均值
train['mean_of_bath'] = card_train[card_train['消费方式']=='淋浴'].groupby('studentid')['消费金额'].agg('median')
#淋浴消费的天数
train['day_of_bath'] = card_train[card_train['消费方式']=='淋浴'].groupby('studentid')['day'].agg('nunique')
#每个同学淋浴消费总额所占花销比例
train['percent_of_bath'] = train['sum_of_bath']/train['sum_of_consume']

#每个同学校车消费总额
train['sum_of_bus'] = card_train[card_train['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('sum')
#每个同学校车消费总次数
train['count_of_bus'] = card_train[card_train['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('count')
#每个同学校车消费平均值
train['mean_of_bus'] = card_train[card_train['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('mean')
#校车消费的天数
train['day_of_bus'] = card_train[card_train['消费方式']=='校车'].groupby('studentid')['day'].agg('nunique')

#每个同学超市消费总额
train['sum_of_shopping'] = card_train[card_train['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('sum')
#每个同学超市消费总次数
train['count_of_shopping'] = card_train[card_train['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('count')
#每个同学超市消费平均值
train['mean_of_shopping'] = card_train[card_train['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('mean')
#超市消费的天数
train['day_of_shopping'] = card_train[card_train['消费方式']=='超市'].groupby('studentid')['day'].agg('nunique')
#每个同学超市消费总额所占花销比例
train['percent_of_shopping'] = train['sum_of_shopping']/train['sum_of_consume']

#每个同学洗衣房消费总额
train['sum_of_wash'] = card_train[card_train['消费方式']=='洗衣房'].groupby('studentid')['消费金额'].agg('sum')
#每个同学洗衣房消费总次数
train['count_of_wash'] = card_train[card_train['消费方式']=='洗衣房'].groupby('studentid')['消费金额'].agg('count')
#每个同学洗衣房消费平均值
train['mean_of_wash'] = card_train[card_train['消费方式']=='洗衣房'].groupby('studentid')['消费金额'].agg('mean')
#洗衣房消费的天数
train['day_of_wash'] = card_train[card_train['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg('mean')
#每个同学洗衣房消费总额所占花销比例
train['percent_of_wash'] = train['sum_of_wash']/train['sum_of_consume']

#每个同学图书馆消费总额
train['sum_of_library'] = card_train[card_train['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('sum')
#每个同学图书馆消费总次数
train['count_of_library'] = card_train[card_train['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('count')
#每个同学图书馆消费平均值
train['mean_of_library'] = card_train[card_train['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('mean')
#每个同学图书馆消费总额所占花销比例
train['percent_of_library'] = train['sum_of_library']/train['sum_of_consume']

#每个同学文印中心消费总额
train['sum_of_print'] = card_train[card_train['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('sum')
#每个同学文印中心消费总次数
train['count_of_print'] = card_train[card_train['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('count')
#每个同学文印中心消费平均值
train['mean_of_print'] = card_train[card_train['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('mean')
#每个同学文印中心消费总额所占花销比例
train['percent_of_print'] = train['sum_of_print']/train['sum_of_consume']

#每个同学教务处消费总额
train['sum_of_training'] = card_train[card_train['消费方式']=='教务处'].groupby('studentid')['消费金额'].agg('sum')
#每个同学教务处消费总次数
train['count_of_training'] = card_train[card_train['消费方式']=='教务处'].groupby('studentid')['消费金额'].agg('count')

#每个同学医院消费总额
train['sum_of_hospital'] = card_train[card_train['消费方式']=='医院'].groupby('studentid')['消费金额'].agg('sum')
#每个同学医院消费总次数
train['count_of_hospital'] = card_train[card_train['消费方式']=='医院'].groupby('studentid')['消费金额'].agg('count')

#每个同学其他消费总额
train['sum_of_other'] = card_train[card_train['消费方式']=='其他'].groupby('studentid')['消费金额'].agg('sum')
#每个同学其他消费总次数
train['count_of_other'] = card_train[card_train['消费方式']=='其他'].groupby('studentid')['消费金额'].agg('count')

#对比两年的变化
train_temp1 = train[train.index%2==1]
train_temp2 = train[train.index%2==0]
train_temp1.index = train_temp1.index-1
train_temp2.index = train_temp2.index+1
train_temp = pd.concat([train_temp1,train_temp2])
train_temp3 = (train+train_temp)/2
#计算第二年校园卡消费比第一年增加量
train['differenct_of_consume'] = train_temp['sum_of_consume']-train['sum_of_consume']
#计算第二年校园卡消费比第一年增加量所占百分比
train['percent_differenct_consume'] = train['differenct_of_consume'] /(train_temp3['sum_of_consume']+1)
#计算第二年校园卡食堂消费比第一年增加量
train['differenct_of_meal'] = train_temp['sum_of_meal']-train['sum_of_meal']
#计算第二年校园卡食堂消费比第一年增加量所占百分比
train['percent_differenct_meal'] = train['differenct_of_meal'] /(train_temp3['sum_of_meal']+1)
#计算第二年校园卡非食堂消费比第一年增加量
train['differenct_of_nomeal'] = train['differenct_of_consume']-train['differenct_of_meal']
#计算第二年校园卡非消费比第一年增加量所占百分比
train['percent_differenct_noconsume'] = train['differenct_of_nomeal'] /(train_temp3['sum_of_consume']-train_temp3['sum_of_meal']+1)


train.reset_index(inplace=True)

return train





#按季度统计消费特征
card_train_temp['quarter'] = card_train_temp['消费时间'].apply(lambda x: x.quarter)
sum_quarter_consume = pd.DataFrame(card_train_temp.groupby(['studentid','quarter'])['消费金额'].sum()).unstack()
sum_quarter_consume_percent = sum_quarter_consume.div(sum_quarter_consume.sum(axis=1),axis=0)













